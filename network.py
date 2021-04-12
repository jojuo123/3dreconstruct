import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
import inception_resnet_v1
import resnet_50_v1

def weight_init(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)

class R_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet_50_v1.Resnet_50_v1(pretrain=True)
        self.layer_id = nn.Conv2d(2048, 80, (1, 1))
        self.layer_ex = nn.Conv2d(2048, 64, (1, 1))
        self.layer_tex = nn.Conv2d(2048, 80, (1, 1))
        self.layer_angles = nn.Conv2d(2048, 80, (1, 1))
        self.layer_tex = nn.Conv2d(2048, 3, (1, 1))
        self.layer_gamma = nn.Conv2d(2048, 27, (1, 1))
        self.layer_t_xy = nn.Conv2d(2048, 2, (1, 1))
        self.layer_t_z = nn.Conv2d(2048, 1, (1, 1))
    def forward(self, x):
        x = self.resnet(x)

        net_id = self.layer_id(x)
        net_ex = self.layer_ex(x)
        net_tex = self.layer_tex(x)
        net_angles = self.layer_angles(x)
        net_gamma = self.layer_gamma(x)
        net_t_xy = self.layer_t_xy(x)
        net_t_z = self.layer_t_z(x)

        net_id = torch.squeeze(net_id)
        net_ex = torch.squeeze(net_ex)
        net_tex = torch.squeeze(net_tex)
        net_angles = torch.squeeze(net_angles)
        net_gamma = torch.squeeze(net_gamma)
        net_t_xy = torch.squeeze(net_t_xy)
        net_t_z = torch.squeeze(net_t_z)

        out = torch.cat((net_id, net_ex, net_tex, net_angles, net_gamma, net_t_xy, net_t_z), 1)

        return out
class Perceptual_Net(nn.Module):
    def __init__(self):
        self.inception = inception_resnet_v1.inception_resnet_v1()
        self.inception.train(False)
        #self.train(False)
    def forward(self, x):
        inputImage = torch.reshape(x, (-1, 3, 224, 224))
        inputImage = torch.clip(inputImage, 0, 255)
        inputImage = (inputImage - 127.5) / 128.0
        feature_128 = self.inception(inputImage)
        return feature_128



