import torch
import torch.nn as nn
from network.buildingBlock import *

class Embedder(nn.Module):
    '''
    input -> [B, 6, 256, 256]
    output -> [B, 512, 1, 1]
    '''
    def __init__(self):

        super(Embedder, self).__init__()

        self.resBlock = ResidualBlockED(in_dim=6, out_dim=64)  # [B, 64, 256, 256]

        self.resDown1 = ResidualBlockED(in_dim=64, out_dim=128, down_sample=True,  pre_activation=True)  # [B, 128, 128, 128]
        self.resDown2 = ResidualBlockED(in_dim=128, out_dim=256, down_sample=True, pre_activation=True)  # [B, 256, 64, 64]
        self.resDown3 = ResidualBlockED(in_dim=256, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 32, 32]

        self.selfAttendion = SelfAttention(in_dim=512)  # [B, 512, 32, 32]

        self.resDown4 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 16, 16]
        self.resDown5 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 8, 8]
        self.resDown6 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 4, 4]


        self.relu = nn.ReLU(inplace=False)
        # average pooling is propositional to the sum pooling
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))                         # [B, 512, 1, 1]


    def forward(self, x, y):

        B, C, H, W = x.shape

        out = torch.cat((x, y), dim=1)

        out = self.resBlock(out)

        out = self.resDown1(out)
        out = self.resDown2(out)
        out = self.resDown3(out)

        out = self.selfAttendion(out)

        out = self.resDown4(out)
        out = self.resDown5(out)
        out = self.resDown6(out)

        # apply sum pooling after ReLU activation based on BIG GAN
        out = self.relu(out)

        out = self.avgPool(out)

        out = out.view(B, 512, 1)

        return out


class Discriminator(nn.Module):

    def __init__(self, num_person, finetuning=False):

        super(Discriminator, self).__init__()

        self.resBlock1 = ResidualBlockED(in_dim=6, out_dim=64)  # [B, 64, 256, 256]


        self.resDown1 = ResidualBlockED(in_dim=64, out_dim=128, down_sample=True, pre_activation=True)   # [B, 128, 128, 128]
        self.resDown2 = ResidualBlockED(in_dim=128, out_dim=256, down_sample=True, pre_activation=True)  # [B, 256, 64, 64]
        self.resDown3 = ResidualBlockED(in_dim=256, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 32, 32]

        self.selfAttendion = SelfAttention(in_dim=512)  # [B, 512, 32, 32]

        self.resDown4 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 16, 16]
        self.resDown5 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 8, 8]
        self.resDown6 = ResidualBlockED(in_dim=512, out_dim=512, down_sample=True, pre_activation=True)  # [B, 512, 4, 4]

        self.resBlock2 = ResidualBlockED(in_dim=512, out_dim=512, pre_activation=True)                    # [B, 512, 4, 4]

        self.relu = nn.ReLU(inplace=False)

        # average pooling is propositional to the sum pooling
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))                     # [B, 512, 1, 1]


        self.W = nn.Parameter(torch.randn(512, num_person))
        self.w_0 = nn.Parameter(torch.randn(512, 1))
        self.b = nn.Parameter(torch.randn(1))

        self.finetuning = finetuning
        self.wPrime = nn.Parameter(torch.randn(512, 1))

    # fine tuning is for one single video or image sequence, input e is the mean of the vector from embedder.
    def initFinetuning(self, e):
        '''
        initialize w_prime = e + w_0
        :param e: [512, 1]
        '''
        self.finetuning = True
        self.wPrime.data = e.detach().data + self.w_0.data                                    # [512, 1]

    def forward(self, x, y, i=None):

        B, C, H, W = x.shape

        out = torch.cat((x, y), dim=1)

        out = self.resBlock1(out)

        out = self.resDown1(out)
        out = self.resDown2(out)
        out = self.resDown3(out)

        out = self.selfAttendion(out)

        out = self.resDown4(out)
        out = self.resDown5(out)
        out = self.resDown6(out)

        out = self.resBlock2(out)

        # apply sum pooling after ReLU activation based on BIG GAN
        out = self.relu(out)                                            # [B, 512, 4, 4]
        out = self.avgPool(out)
        out = out.view(B, 512, 1).transpose(1, 2)                       # [B, 1, 512]

        if self.finetuning:
            # out -> [1, 1, 512]
            # wPrime -> [512, 1]
            print('finetuning')
            out = torch.bmm(out, self.wPrime.unsqueeze(0)) + self.b
        else:
            assert i is not None, "input person id"
            out = torch.bmm(out, self.W[:, i].transpose(0, 1).unsqueeze(-1) + self.w_0) + self.b

        out = out.view(B, -1)

        return out


class Generator(nn.Module):
    P_Len = (512 * 2 + 512 * 2) * 4 + (512 * 2 + 256 * 2) + (256 * 2 + 128 * 2) + (128 * 2 + 64 * 2) + (
            64 * 2 + 3 * 2) + (3 * 2)
    P_Slice = [0,
               512 * 4,
               512 * 4,
               512 * 4,
               512 * 4,
               512 * 2 + 256 * 2,
               256 * 2 + 128 * 2,
               128 * 2 + 64 * 2,
               64 * 2 + 3 * 2,
               3 * 2]
    for i in range(1, len(P_Slice)):
        P_Slice[i] = P_Slice[i - 1] + P_Slice[i]

    def __init__(self, finetuning = False):
        super(Generator, self).__init__()

        self.resDown1 = ResidualBlockG_Down(in_dim=3, out_dim=64)                                                   # [B, 64, 128, 128]
        self.resDown2 = ResidualBlockG_Down(in_dim=64, out_dim=128)                                                 # [B, 128, 64, 64]
        self.resDown3 = ResidualBlockG_Down(in_dim=128, out_dim=256)                                                # [B, 256, 32, 32]
        self.resDown4 = ResidualBlockG_Down(in_dim=256, out_dim=512)                                                # [B, 512, 16, 16]

        self.resBlock1 = ResidualBlockG_Regular(in_dim=512, out_dim=512)                                            # [B, 512, 16, 16]
        self.resBlock2 = ResidualBlockG_Regular(in_dim=512, out_dim=512)                                            # [B, 512, 16, 16]
        self.resBlock3 = ResidualBlockG_Regular(in_dim=512, out_dim=512)                                            # [B, 512, 16, 16]
        self.resBlock4 = ResidualBlockG_Regular(in_dim=512, out_dim=512)                                            # [B, 512, 16, 16]

        self.resUp1 = ResidualBlockG_Up(in_dim=512, out_dim=256)                                                    # [B, 256, 32, 32]
        self.resUp2 = ResidualBlockG_Up(in_dim=256, out_dim=128)                                                    # [B, 128, 64, 64]
        self.resUp3 = ResidualBlockG_Up(in_dim=128, out_dim=64)                                                     # [B, 64, 128, 128]
        self.resUp4 = ResidualBlockG_Up(in_dim=64, out_dim=3)                                                       # [B, 2, 256, 256]

        self.attention1 = SelfAttention(256)
        self.attention2 = SelfAttention(128)

        self.adaIn = AdaIn()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()

        self.P = nn.Parameter(torch.randn(self.P_Len, 512))

        self.finetuning = finetuning

        self.psi = nn.Parameter(torch.randn(1, self.P_Len, 1))




        # fine tuning is for one single video or image sequence, input e is the mean of the vector from embedder.
    def initFinetuning(self, e):
        '''
        initialize w_prime = e + w_0
        :param e: [512, 1]
        '''
        self.finetuning = True
        self.psi.data = torch.bmm(self.P.detach().view(1, self.P_Len, 512), e.detach().view(1, 512, 1)).data              # [1, P_Len, 1]


    def forward(self, x, e=None):
        '''

        :param x: [B, C, H, W]
        :param e: [B, 512, 1]
        :return:
        '''
        B, C, H, W = x.shape

        if self.finetuning:
            psi = self.psi
        else:
            psi = torch.bmm(self.P.expand(B, self.P_Len, 512), e)                                       # [B, P_Len, 1]

        out = self.resDown1(x)
        out = self.resDown2(out)
        out = self.resDown3(out)
        out = self.attention1(out)
        out = self.resDown4(out)

        out = self.resBlock1(out, psi[:, self.P_Slice[0]:self.P_Slice[1], :])
        out = self.resBlock2(out, psi[:, self.P_Slice[1]:self.P_Slice[2], :])
        out = self.resBlock3(out, psi[:, self.P_Slice[2]:self.P_Slice[3], :])
        out = self.resBlock4(out, psi[:, self.P_Slice[3]:self.P_Slice[4], :])

        out = self.resUp1(out, psi[:, self.P_Slice[4]:self.P_Slice[5], :])
        out = self.resUp2(out, psi[:, self.P_Slice[5]:self.P_Slice[6], :])
        out = self.attention2(out)
        out = self.resUp3(out, psi[:, self.P_Slice[6]:self.P_Slice[7], :])
        out = self.resUp4(out, psi[:, self.P_Slice[7]:self.P_Slice[8], :])

        out = self.adaIn(out, psi[:, self.P_Slice[8]:self.P_Slice[8]+3, :], psi[:, self.P_Slice[8]+3:self.P_Slice[9], :])
        out = self.relu(out)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out