import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
import face_decoder
import network
import inception_resnet_v1
import losses

class Reconstruction_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Face3D = face_decoder.Face3D()
        self.opt = opt
        self.r_net = network.R_net()

    def forward(self, x):
        x_image = x['image']
        x_lm = x['landmark']
        x_mask = x['mask']
        x_image_coeff = self.r_net(x_image)
        next_input = {
            'coeff': x_image_coeff,
            'opt': self.opt
        }
        x_rendered = self.Face3D(next_input)
        # id_label = self.perceptual(x_image)
        # id_features = self.perceptual(x_rendered['render_imgs'])

        return {
            'rendered_img': x_rendered['img'],
            'attention_mask': x_mask,
            'landmark': x_lm,
            'img': x_image,
            'img_mask_crop': x_rendered['img_mask_crop'],
            'rendered_landmark': x_rendered['landmark_p'],
            'id_coeff': x_rendered['id_coeff'],
            'ex_coeff': x_rendered['ex_coeff'],
            'tex_coeff': x_rendered['tex_coeff'],
            'face_texture': x_rendered['face_texture'],
            'facemodel': x_rendered['facemodel'],
            'opt': self.opt,
            'gamma': x_rendered['gamma']
        }

class Loss_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = network.Perceptual_Net()
        self.perceptual.train(False)
    
    def forward(self, x):
        rendered_img = x['rendered_img']
        img = x['img']
        id_labels = self.perceptual(img)
        id_features = self.perceptual(rendered_img)

        x['id_labels'] = id_labels
        x['id_features'] = id_features

        return x

class FullModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.reconstruction = Reconstruction_model(self.opt)
        self.loss_layer = Loss_layer()
    
    def forward(self, x):
        x = self.reconstruction(x)
        x = self.loss_layer(x)
        x = self.loss(x)
        return x
    
    def loss(self, x):
        photo_loss = losses.Photo_loss(x['img'], x['rendered_img'], x['img_mask_crop'] * x['attention_mask'])
        landmark_loss = losses.Landmark_loss(x['rendered_landmark'], x['landmark'])
        perceptual_loss = losses.Perceptual_loss(x['id_features'], x['id_features'])
        reg_loss = losses.Regulation_loss(x['id_coeff'], x['ex_coeff'], x['tex_coeff'], x['opt'])
        reflect_loss = losses.Reflectance_loss(x['face_texture'], x['facemodel'])
        gamma_loss = losses.Gamma_loss(x['gamma'])

        opt = x['opt']
        loss = opt.w_photo*photo_loss + opt.w_lm*landmark_loss + opt.w_id*perceptual_loss + opt.w_reg*reg_loss + opt.w_ref*reflect_loss + opt.w_gamma*gamma_loss

        return loss
    
