import torch 
import torch.nn.functional as F
from scipy.io import loadmat, savemat

def Photo_loss(input_imgs, render_imgs, img_mask):
    input_imgs = input_imgs.type(torch.float32)
    #image = [batchsize, channel, height, width]
    (img_mask[:, 0, :, :]).detach()

    
    photo_loss = torch.sqrt(torch.sum(torch.square(input_imgs - render_imgs), dim=1)) * img_mask/255

    photo_loss = torch.sum(photo_loss) / torch.maximum(torch.sum(img_mask), torch.tensor(1.0))

    return photo_loss

def Perceptual_loss(id_feature, id_label):
    id_feature = F.normalize(id_feature, dim=1)
    id_label = F.normalize(id_label, dim=1)
    sim = torch.sum(id_feature*id_label, dim=1)
    loss = torch.sum(torch.maximum(torch.tensor(0.0), 1.0 - sim)) / (torch.tensor(id_feature.shape[0])).type(torch.float32)
    return loss

def Landmark_loss(landmark_p, landmark_label):
    landmark_weight = torch.cat(
        (
            torch.ones((1, 28)),
            20 * torch.ones((1, 3)),
            torch.ones((1, 29)),
            20 * torch.ones((1, 8))
        ),
        1
    )
    landmark_weight = torch.tile(landmark_weight, (int(landmark_p.shape[0]), 1))

    landmark_loss = torch.sum(torch.sum(torch.square(landmark_p - landmark_label), dim=2) * landmark_weight) / (68.0 * (torch.tensor(landmark_p.shape[0])).type(torch.float32))

    return landmark_loss

def Regulation_loss(id_coeff, ex_coeff, tex_coeff, opt):
    w_ex = opt.w_ex
    w_tex = opt.w_tex

    regulation_loss = F.mse_loss(id_coeff, torch.zeros(id_coeff.shape)) + w_ex * F.mse_loss(ex_coeff, torch.zeros(ex_coeff.shape)) + w_tex * F.mse_loss(tex_coeff, torch.zeros(tex_coeff.shape))
    regulation_loss = 2 * regulation_loss / (torch.tensor(id_coeff.shape[0])).type(torch.float32)

    return regulation_loss

def Reflectance_loss(face_texture, facemodel):
    skin_mask = facemodel.skin_mask
    skin_mask = torch.reshape(skin_mask, (1, skin_mask.shape[0], 1))

    texture_mean = torch.sum(face_texture * skin_mask, dim=1) / torch.sum(skin_mask)
    texture_mean = torch.unsqueeze(texture_mean, 1)

    reflectance_loss = torch.sum(torch.square((face_texture - texture_mean) * skin_mask / 255.0)) / ((torch.tensor(face_texture.shape[0])).type(torch.float32) * torch.sum(skin_mask))

    return reflectance_loss

def Gamma_loss(gamma):
    gamma = torch.reshape(gamma, (-1, 3, 9))
    gamma_mean = torch.mean(gamma, dim=1, keepdim=True)
    gamma_loss = torch.mean(torch.square(gamma-gamma_mean))
    return gamma_loss