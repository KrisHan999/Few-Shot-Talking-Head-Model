import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg19
from network.vgg import *
import config

class LossD(nn.Module):
    def __init__(self, gpu=None):
        super(LossD, self).__init__()

        self.gpu = gpu
        if self.gpu is not None:
            self.cuda(gpu)

    def forward(self, d_x, d_x_hat):
        '''
        return loss of discriminator given the score of real and fake image
        :param d_x: [B, 1]
        :param d_x_hat: [B, 1]
        :return:
        '''
        if self.gpu is not None:
            d_x = d_x.cuda(self.gpu)
            d_x_hat = d_x_hat.cuda(self.gpu)
        return F.relu(1-d_x).mean() + F.relu(1+d_x_hat).mean()


class LossEG(nn.Module):
    def __init__(self, gpu=None):
        super(LossEG, self).__init__()

        self.vgg19_layers = [1, 6, 11, 18, 25]
        self.vggface_layers = [1, 6, 11, 20, 29]
        self.VGG19_Activations = VGG_Activations(vgg19(pretrained=True), self.vgg19_layers)
        self.VGGface_Activations = VGG_Activations(vgg_face(pretrained=True), self.vggface_layers)

        self.gpu = gpu
        if self.gpu is not None:
            self.cuda(gpu)

    def loss_cnt(self, x, x_hat):

        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

        x = (x-IMG_NET_MEAN)/IMG_NET_STD
        x_hat = (x_hat-IMG_NET_MEAN)/IMG_NET_STD

        vgg19_x_activations = self.VGG19_Activations(x)
        vgg19_x_hat_activations = self.VGG19_Activations(x_hat)

        vgg19_loss = 0
        for i in range(len(self.vgg19_layers)):
            vgg19_loss += F.l1_loss(vgg19_x_activations[i], vgg19_x_hat_activations[i])

        vggface_x_activations = self.VGGface_Activations(x)
        vggface_x_hat_activations = self.VGGface_Activations(x_hat)

        vggface_loss = 0
        for i in range(len(self.vggface_layers)):
            vggface_loss += F.l1_loss(vggface_x_activations[i], vggface_x_hat_activations[i])

        return vgg19_loss*config.LOSS_VGG19_WEIGHT + vggface_loss*config.LOSS_VGG_FACE_WEIGHT



    def loss_adv(self, d_x_hat, d_x_fm, d_x_hat_fm):
        '''
        adversarial loss for training generator
        :param d_x_hat: [B, 1]
        :return: scalar
        '''
        return (-d_x_hat.mean() + self.loss_fm(d_x_fm, d_x_hat_fm) * config.LOSS_FM_WEIGHT)

    def loss_mch(self, mean_vector, wi):
        '''
        match loss of mean vector and wi in the discriminator
        :param vector: [B, 512, 1]
        :param wi: [B, 512, 1]
        :return:
        '''
        return F.l1_loss(mean_vector, wi)*config.LOSS_MCH_WEIGHT

    def loss_fm(self, d_x_fm, d_x_hat_fm):

        loss = 0
        for i in range(len(d_x_fm)):
            loss += F.l1_loss(d_x_fm[i], d_x_hat_fm[i])

        # if self.gpu is not None:
        #     loss = loss.cuda(self.gpu)
        return loss

    def forward(self, x, x_hat, d_x_hat, mean_vector, wi, d_x_fm, d_x_hat_fm):

        if self.gpu is not None:
            x = x.cuda(self.gpu)
            x_hat = x_hat.cuda(self.gpu)
            d_x_hat = d_x_hat.cuda(self.gpu)
            mean_vector = mean_vector.cuda(self.gpu)
            wi = wi.cuda(self.gpu)
            d_x_fm = [data.cuda(self.gpu) for data in d_x_fm]
            d_x_hat_fm = [data.cuda(self.gpu) for data in d_x_hat_fm]


        cnt_loss = self.loss_cnt(x, x_hat)
        mch_loss = self.loss_mch(mean_vector, wi)
        adv_loss = self.loss_adv(d_x_hat, d_x_fm, d_x_hat_fm)

        return cnt_loss+mch_loss+adv_loss


