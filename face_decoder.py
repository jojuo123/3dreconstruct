import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
import numpy as np
from scipy.io import loadmat
from options import Option
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_rotation,
    PointLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TensorProperties,
    specular
)
#import option as opt
import math as m

class PointLightsNew(TensorProperties):
    def __init__(
        self,
        diffuse_color_per_vertex,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device: str = "cpu",
    ):
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: torch.device on which the tensors should be located
        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            diffuse_color_per_vertex=diffuse_color_per_vertex,
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        # pyre-fixme[16]: `PointLights` has no attribute `location`.
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        # pyre-fixme[16]: `PointLights` has no attribute `location`.
        direction = self.location - points
        # pyre-fixme[16]: `PointLights` has no attribute `diffuse_color`.
        return self.diffuse_color_per_vertex
    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        # pyre-fixme[16]: `PointLights` has no attribute `location`.
        direction = self.location - points
        return specular(
            points=points,
            normals=normals,
            # pyre-fixme[16]: `PointLights` has no attribute `specular_color`.
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )

def _validate_light_properties(obj):
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = "Expected %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))


class BFM():
    def __init__(self, model_path='./BFM/BFM_model_front.mat'):
        model = loadmat(model_path)
        self.opt = Option()
        self.meanshape = torch.tensor(model['meanshape']).to(self.opt.device)
        self.idBase = torch.tensor(model['idBase']).to(self.opt.device)
        self.exBase = torch.tensor(model['exBase'].astype(np.float32)).to(self.opt.device)
        self.meantex = torch.tensor(model['meantex']).to(self.opt.device)
        self.texBase = torch.tensor(model['texBase']).to(self.opt.device)
        self.point_buf = torch.tensor(model['point_buf']).to(self.opt.device)
        self.face_buf = torch.tensor(model['tri']).to(self.opt.device)
        self.front_mask_render = torch.squeeze(torch.tensor(model['frontmask2_idx'])).to(self.opt.device)
        self.mask_face_buf = torch.tensor(model['tri_mask2']).to(self.opt.device)
        self.skin_mask = torch.squeeze(torch.tensor(model['skinmask'])).to(self.opt.device)
        self.keypoints = torch.squeeze(torch.tensor(model['keypoints'])).to(self.opt.device)

class Face3D(nn.Module):
    def __init__(self):
        super().__init__()
        facemodel = BFM()
        self.facemodel = facemodel
        self.rasterization_setting = RasterizationSettings(
            image_size=224
        )
        self.mesh_rasterizer = MeshRasterizer(cameras=None, raster_settings=self.rasterization_setting)
        self.shader = SoftPhongShader()

        self.renderer = MeshRenderer(self.mesh_rasterizer, self.shader)
        self.opt = Option()

    def Reconstruction_Block(self, coeff, opt):
        id_coeff,ex_coeff,tex_coeff,angles,translation,gamma,camera_scale,f_scale = self.Split_coeff(coeff)

        face_shape = self.Shape_formation_block(id_coeff, ex_coeff, self.facemodel)
        face_texture = self.Texture_formation_block(tex_coeff, self.facemodel)
        rotation = self.Compute_rotation_matrix(angles)
        face_norm = self.Compute_norm(face_shape, self.facemodel)
        norm_r = torch.matmul(face_norm, rotation)

        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)

        face_landmark_t = self.Compute_landmark(face_shape_t, self.facemodel)
        landmark_p = self.Projection_block(face_landmark_t, camera_scale, f_scale)

        face_color = self.Illumination_block(face_texture, norm_r, gamma)
        #need opt
        render_block = self.Render_block(face_shape_t, norm_r, face_color, camera_scale, f_scale, self.facemodel, opt.batch_size, opt.is_train)

        self.id_coeff = id_coeff
        self.ex_coeff = ex_coeff
        self.tex_coeff = tex_coeff
        self.f_scale = f_scale
        self.gamma = gamma
        self.face_shape = face_shape
        self.face_shape_t = face_shape_t
        self.face_texture = face_texture
        self.face_color = face_color
        self.landmark_p = landmark_p
        self.render_block = render_block
        #self.render_imgs = render_imgs
        #self.img_mask = img_mask
        #self.img_mask_crop = img_mask_crop


    def Split_coeff(self, coeff):
        id_coeff = coeff[:, :80]
        ex_coeff = coeff[:, 80:144]
        tex_coeff = coeff[:, 144:224]
        angles = coeff[:, 224:227]
        gamma = coeff[:, 227:254]
        translation = coeff[:, 254:257]
        camerascale = torch.ones((int(coeff.shape[0]), 1)).to(self.opt.device)
        f_scale = torch.ones((int(coeff.shape[0]), 1)).to(self.opt.device)
        return id_coeff, ex_coeff, tex_coeff, angles, translation, gamma, camerascale, f_scale
    
    def Shape_formation_block(self, id_coeff, ex_coeff, facemodel):
        face_shape = torch.einsum('ij,aj->ai', facemodel.idBase, id_coeff) + torch.einsum('ij,aj->ai', facemodel.exBase, ex_coeff) + facemodel.meanshape

        face_shape = torch.reshape(face_shape, (int(face_shape.shape[0]), -1, 3))
        face_shape = face_shape - torch.reshape((torch.mean(torch.reshape(facemodel.meanshape, (-1, 3)) ,0)), (1,1,3))

        return face_shape
    
    def Compute_norm(self, face_shape, facemodel):
        shape = face_shape
        face_id = facemodel.face_buf
        point_id = facemodel.point_buf

        face_id = (face_id - 1).type(torch.int32)
        point_id = (point_id - 1).type(torch.int32)

        # print(face_id.shape)
        # print(shape.shape)
        vv1 = shape[:, face_id[:, 0].type(torch.long), :]
        vv2 = shape[:, face_id[:, 1].type(torch.long), :]
        vv3 = shape[:, face_id[:, 2].type(torch.long), :]

        # v1 = torch.gather(shape, 1, face_id[:, 0])
        # v2 = torch.gather(shape, 1, face_id[:, 1])
        # v3 = torch.gather(shape, 1, face_id[:, 2])
        e1 = vv1 - vv2
        e2 = vv2 - vv3
        face_norm = torch.cross(e1, e2)

        face_norm = F.normalize(face_norm, p=2, dim=2)
        face_norm = torch.cat((face_norm, 
        torch.zeros((int(face_shape.shape[0]), 1, 3)).to(self.opt.device)
        ), 1)

        #print(face_norm.shape)
        tmp = face_norm[:, point_id.type(torch.long), :]
        #print(tmp.shape)

        v_norm = torch.sum(tmp, 2)
        #v_norm = torch.sum(torch.gather(face_norm, 1, point_id), 2)
        v_norm = F.normalize(v_norm, dim=2)

        return v_norm

    def Texture_formation_block(self, tex_coeff, face_model):
        # print(tex_coeff.shape)
        # print(face_model.texBase.shape)
        face_texture = torch.einsum('ij,aj->ai', face_model.texBase, tex_coeff) + face_model.meantex
        face_texture = torch.reshape(face_texture, (
            int(face_texture.shape[0]),
            -1,
            3
        ))

        return face_texture

    def Compute_rotation_matrix(self, angles):
        n_data = int(angles.shape[0])
        rotation_X = torch.cat((
            torch.ones((n_data, 1)).to(self.opt.device),
            torch.zeros((n_data, 3)).to(self.opt.device),
            torch.reshape(torch.cos(angles[:, 0]), (n_data, 1)),
            -torch.reshape(torch.sin(angles[:, 0]), (n_data, 1)),
            torch.zeros((n_data, 1)).to(self.opt.device),
            torch.reshape(torch.sin(angles[:, 0]), (n_data, 1)),
            torch.reshape(torch.cos(angles[:, 0]), (n_data, 1)),
        ), 1).to(self.opt.device)

        rotation_Y = torch.cat(
            (torch.reshape(torch.cos(angles[:,1]),(n_data,1)),
			torch.zeros((n_data,1)).to(self.opt.device),
			torch.reshape(torch.sin(angles[:,1]),(n_data,1)),
			torch.zeros((n_data,1)).to(self.opt.device),
			torch.ones((n_data,1)).to(self.opt.device),
			torch.zeros((n_data,1)).to(self.opt.device),
			-torch.reshape(torch.sin(angles[:,1]),(n_data,1)),
			torch.zeros((n_data,1)).to(self.opt.device),
			torch.reshape(torch.cos(angles[:,1]),(n_data,1))),
			1
		)

        rotation_Z = torch.cat(
            (torch.reshape(torch.cos(angles[:,2]),(n_data,1)),
			-torch.reshape(torch.sin(angles[:,2]),(n_data,1)),
			torch.zeros((n_data,1)).to(self.opt.device),
			torch.reshape(torch.sin(angles[:,2]),(n_data,1)),
			torch.reshape(torch.cos(angles[:,2]),(n_data,1)),
			torch.zeros([n_data,3]).to(self.opt.device),
			torch.ones((n_data,1)).to(self.opt.device)),
			1
		)

        rotation_X = torch.reshape(rotation_X,(n_data,3,3))
        rotation_Y = torch.reshape(rotation_Y,(n_data,3,3))
        rotation_Z = torch.reshape(rotation_Z,(n_data,3,3))

        rotation = torch.matmul(torch.matmul(rotation_Z, rotation_Y), rotation_X)

        rotation = torch.transpose(rotation, 2, 1)

        return rotation

    def Projection_block(self, face_shape, camera_scale, f_scale):
        focal = torch.tensor(1015.0)
        focal = focal * f_scale
        focal = torch.reshape(focal, (-1, 1))
        batchsize = int(focal.shape[0])

        camera_pos = torch.reshape(torch.tensor([0.0, 0.0, 10.0]).to(self.opt.device), (1, 1, 3)) * torch.reshape(camera_scale, (-1, 1, 1))
        reverse_z = torch.tile(torch.reshape(torch.tensor([1.0,0,0,0,1,0,0,0,-1.0]), (1, 3, 3)), (int(face_shape.shape[0]), 1, 1)).to(self.opt.device)

        p_matrix = torch.cat(
            (
                focal,
                torch.zeros((batchsize, 1)).to(self.opt.device),
                112.0 * torch.ones((batchsize, 1)).to(self.opt.device),
                torch.zeros((batchsize, 1)).to(self.opt.device),
                focal,
                112.0 * torch.ones((batchsize, 1)).to(self.opt.device),
                torch.zeros((batchsize, 2)).to(self.opt.device),
                torch.ones((batchsize, 1)).to(self.opt.device),
            ),
            1
        )
        p_matrix = torch.reshape(p_matrix, (-1, 3, 3))

        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, torch.transpose(p_matrix, 2, 1))

        face_projection = aug_projection[:, :, 0:2] / torch.reshape(aug_projection[:, :, 2], (int(face_shape.shape[0]), int(aug_projection.shape[1]), 1))

        return face_projection

    def Compute_landmark(self, face_shape, facemodel):

        keypoints_idx = facemodel.keypoints
        keypoints_idx = (keypoints_idx - 1).type(torch.long)
        #print(face_shape.shape)
        #face_landmark = torch.gather(face_shape, 1, keypoints_idx)
        face_landmark = face_shape[:, keypoints_idx, :]
        return face_landmark

    def Illumination_block(self, face_texture, norm_r, gamma):
        n_data = int(gamma.shape[0])
        n_point = int(norm_r.shape[1])
        gamma = torch.reshape(gamma, (n_data, 3, 9))

        init_lit = torch.tensor([0.8,0,0,0,0,0,0,0,0]).to(self.opt.device)
        gamma = gamma + torch.reshape(init_lit, (1, 1, 9))

        a0 = m.pi
        a1 = 2*m.pi / torch.sqrt(torch.tensor(3.0).type(torch.float32)).to(self.opt.device)
        a2 = 2*m.pi / torch.sqrt(torch.tensor(8.0).type(torch.float32)).to(self.opt.device)
        c0 = 1/torch.sqrt(torch.tensor(4 * m.pi).type(torch.float32)).to(self.opt.device)
        c1 = torch.sqrt(torch.tensor(3.0)) / torch.sqrt(torch.tensor(4 * m.pi)).to(self.opt.device)
        c2 = 3 * torch.sqrt(torch.tensor(5.0)) / torch.sqrt(torch.tensor(12 * m.pi)).to(self.opt.device)

        Y = torch.cat(
            (
                torch.tile(torch.reshape(a0 * c0, (1, 1, 1)), (n_data, n_point, 1)),
                torch.unsqueeze(-a1*c1*norm_r[:,:,1], 2),
                torch.unsqueeze(a1*c1*norm_r[:,:,2], 2),
			    torch.unsqueeze(-a1*c1*norm_r[:,:,0], 2),
			    torch.unsqueeze(a2*c2*norm_r[:,:,0]*norm_r[:,:,1], 2),
			    torch.unsqueeze(-a2*c2*norm_r[:,:,1]*norm_r[:,:,2], 2),
                torch.unsqueeze(a2*c2*0.5/torch.sqrt(torch.tensor(3.0))*(3*torch.square(norm_r[:,:,2])-1),2),
			    torch.unsqueeze(-a2*c2*norm_r[:,:,0]*norm_r[:,:,2],2),
                torch.unsqueeze(a2*c2*0.5*(torch.square(norm_r[:,:,0])-torch.square(norm_r[:,:,1])),2)
            ),
            2
        )

        color_r = torch.squeeze(torch.matmul(Y,torch.unsqueeze(gamma[:,0,:],2)),axis = 2)
        color_g = torch.squeeze(torch.matmul(Y,torch.unsqueeze(gamma[:,1,:],2)),axis = 2)
        color_b = torch.squeeze(torch.matmul(Y,torch.unsqueeze(gamma[:,2,:],2)),axis = 2)

        face_color = torch.stack([color_r*face_texture[:,:,0],color_g*face_texture[:,:,1],color_b*face_texture[:,:,2]], 2)

        return face_color
    
    def Rigid_transform_block(self, face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + torch.reshape(translation, (int(face_shape.shape[0]), 1, 3))

        return face_shape_t

    def Render_block(self, face_shape, face_norm, face_color, camera_scale, f_scale, facemodel, batchsize, is_train=True):
        

        n_vex = int(facemodel.idBase.shape[0]/3)

        fov_y = 2 * torch.atan(112.0 / (1015.0 * f_scale)) * 180.0 / m.pi
        # print(fov_y)
        # fov_y = torch.reshape(fov_y, (batchsize, 1))
        fov_y = float(fov_y[0, 0])

        face_shape = torch.reshape(face_shape, (batchsize, n_vex, 3))
        face_norm = torch.reshape(face_norm, (batchsize, n_vex, 3))
        face_color = torch.reshape(face_color, (batchsize, n_vex, 3))

        # mask_face_shape = torch.gather(face_shape, 1, (facemodel.front_mask_render - 1).type(torch.int32))
        # mask_face_norm = torch.gather(face_norm, 1, (facemodel.front_mask_render - 1).type(torch.int32))
        # mask_face_color = torch.gather(face_color, 1, (facemodel.front_mask_render - 1).type(torch.int32))

        mask_face_shape = face_shape[:, (facemodel.front_mask_render - 1).type(torch.long), :]
        mask_face_norm = face_norm[:, (facemodel.front_mask_render - 1).type(torch.long), :]
        mask_face_color = face_color[:, (facemodel.front_mask_render - 1).type(torch.long), :]

        camera_position = torch.tensor([0, 0, 10.0]).to(self.opt.device) * torch.reshape(camera_scale, (-1, 1))
        camera_look_at = torch.tensor([0, 0, 0.0]).to(self.opt.device)
        camera_up = torch.tensor([0, 1.0, 0]).to(self.opt.device)

        camera_look_at = torch.reshape(camera_look_at, (1, 3))
        camera_up = torch.reshape(camera_up, (1, 3))

        light_positions = torch.tile(torch.reshape(torch.tensor([0, 0, 1e5]).to(self.opt.device), (1, 1, 3)), (batchsize, 1, 1))
        light_intensities = torch.tile(torch.reshape(torch.tensor([0.0, 0.0, 0.0]).to(self.opt.device), (1, 1, 3)), (batchsize, 1, 1))
        ambient_color = torch.tile(torch.reshape(torch.tensor([1.0, 1, 1]), (1, 3)), (batchsize, 1)).to(self.opt.device)

        #goi 3dpytorch

        return {
            'n_vex': n_vex,
            'fov_y': fov_y,
            'face_shape': face_shape,
            'face_norm': face_norm,
            'face_color': face_color,
            'mask_face_shape': mask_face_shape,
            'mask_face_norm': mask_face_norm,
            'mask_face_color': mask_face_color,
            'camera_position': camera_position,
            'camera_look_at': camera_look_at,
            'camera_up': camera_up,
            'light_positions': light_positions,
            'light_intensities': light_intensities,
            'ambient_color': ambient_color
        }

    def forward(self, x):
        coeff = x['coeff']
        opt = x['opt']
        self.Reconstruction_Block(coeff, opt)
        #goi 3dtorch
        # print(self.render_block['camera_position'].shape)
        # print(self.render_block['camera_look_at'].shape)
        # print(self.render_block['camera_up'].shape)
        camera_look_at_rotation = look_at_rotation(self.render_block['camera_position'], self.render_block['camera_look_at'], self.render_block['camera_up'])

        camera = FoVPerspectiveCameras(fov=self.render_block['fov_y'], znear=0.01, zfar=50.0, R=camera_look_at_rotation, device=self.opt.device)

        lights = PointLightsNew(diffuse_color_per_vertex=self.render_block['face_color'], ambient_color=self.render_block['ambient_color'])

        lights_mask = PointLightsNew(diffuse_color_per_vertex=self.render_block['mask_face_color'], ambient_color=self.render_block['ambient_color'])

        faces = (self.facemodel.face_buf-1).type(torch.long)
        faces_tmp = torch.unsqueeze(faces, 0)
        # tmp = -1*torch.ones((3, faces.shape[0], faces.shape[1])).to(self.opt.device)
        # #print(faces.shape)
        # #print(tmp)
        # faces_tmp = torch.cat([faces_tmp, tmp], dim=0).type(torch.long)
        #print(faces_tmp.shape)
        faces_tmp2 = torch.unsqueeze(faces, 0)
        faces_tmp2 = faces_tmp2.repeat(4, 1, 1)
        #print(faces_tmp2.shape)

        #print(self.render_block['mask_face_shape'].shape)
        #print(faces.shape)

        #print(faces_tmp.shape)
        #print(self.render_block['mask_face_shape'].shape)

        faces_mask = (self.facemodel.mask_face_buf-1).type(torch.long)
        faces_tmp_mask = torch.unsqueeze(faces_mask, 0)
        #print(faces_tmp.shape)
        faces_tmp2_mask = torch.unsqueeze(faces_mask, 0)
        faces_tmp2_mask = faces_tmp2_mask.repeat(4, 1, 1)

        meshes_world = Meshes(verts=self.face_shape, faces=faces_tmp2)

        meshes_world_mask = Meshes(verts=self.render_block['mask_face_shape'], faces=faces_tmp2_mask)

        out_img_raw = self.renderer(meshes_world, cameras=camera, lights=lights)

        img = out_img_raw[:, :, :, :3]
        mask = out_img_raw[:, :, :, 3:]

        img = torch.flip(img, 3).type(torch.float32)
        mask = mask.type(torch.float32)

        if (self.opt.is_train):
            mask_crop = self.renderer(meshes_world_mask, cameras=camera, lights=lights_mask)
            mask_f = mask_crop[:, :, :, 3:]
            mask_f = mask_f.type(torch.float32)

            return {
                'rendered_img': img,
                'img_mask': mask,
                'img_mask_crop': mask_f,
                'landmark_p': self.landmark_p,
                'id_coeff': self.id_coeff,
                'ex_coeff': self.ex_coeff,
                'tex_coeff': self.tex_coeff,
                'face_texture': self.face_texture,
                'facemodel': self.facemodel,
                'gamma': self.gamma
            }


        return {
            'rendered_img': img,
            'img_mask': mask,
            'img_mask_crop': mask,
            'landmark_p': self.landmark_p,
            'id_coeff': self.id_coeff,
            'ex_coeff': self.ex_coeff,
            'tex_coeff': self.tex_coeff,
            'face_texture': self.face_texture,
            'facemodel': self.facemodel,
            'gamma': self.gamma
        }