import torch
import torch.nn as nn

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, feature, mean_style, std_style):
        '''
        Adaptive instance normalization
        :param feature: [B, C, H, W]
        :param mean_style: [B, C, !]
        :param std_style: [B, C, 1]
        :return:[B, C, H, W]
        '''
        B, C, H, W = feature.shape

        # get the channel-wise feature
        feature = feature.view(B, C, H*W)
        # get the mean and std of feature channel-wisely
        mean_feature = torch.mean(feature, dim=2).view(B, C, 1)
        std_feature = (torch.std(feature, dim=2) + self.eps).view(B, C, 1)

        adaIn_feature = ((feature - mean_feature) / std_feature) * std_style + mean_style
        # change back to the input shape
        adaIn_feature = adaIn_feature.view(B, C, H, W)

        return adaIn_feature


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.in_dim = in_dim

        self.conv_f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv_o = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        '''
        Self attention
        :param x: [B, C, H, W]
        :return: [B, C, H, W]
        '''
        B, C, H, W = x.shape

        f = self.conv_f(x).view(B, -1, H*W)                                                    # [B, C//8, N]
        g = self.conv_g(x).view(B, -1, H*W).permute(0, 2, 1)                                   # [B, N, C//8]
        h = self.conv_h(x).view(B, -1, H*W)                                                    # [B, C, N]
        attention = self.softmax(torch.bmm(g, f))                                              # [B, N, N]
        # permute here is because of the paper
        o = self.conv_o(torch.bmm(h, attention.permute(0, 2, 1)).view(B, C, H, W))             # [B, C, H, W]
        y = self.gamma * o + x

        return y


'''
Based on BIG gan from [Large Scale GAN Training for High Fidelity Natural Image Synthesis]
https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py#L103
'''
class ResidualBlockED(nn.Module):

    def __init__(self, in_dim, out_dim, down_sample=False, pre_activation=False):
        super(ResidualBlockED, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.down_sample = down_sample
        self.pre_activation = pre_activation

        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1))

        self.relu = nn.ReLU(inplace=False)

        if(self.down_sample):
            self.avg_pooling = nn.AvgPool2d(kernel_size=2)

        self.skip_flag = ((in_dim != out_dim) or down_sample)
        if(self.skip_flag):
            self.conv_l = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))

    def shortcut(self, x):
        if(self.pre_activation):
            if (self.skip_flag):
                x = self.conv_l(x)
            if (self.down_sample):
                x = self.avg_pooling(x)
        else:
            if (self.down_sample):
                x = self.avg_pooling(x)
            if (self.skip_flag):
                x = self.conv_l(x)
        return x

    def forward(self, x):
        '''
        Residual block: regular or down sample residual block for embedder and discriminator without normalization
        :param x: [B, C_in, H, W]
        :return: [B, C_out, H, W]
        '''
        # shorcut
        out_res = self.shortcut(x)

        if(self.pre_activation):
            out = self.relu(x)
        else:
            out = x
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        if(self.down_sample):
            out = self.avg_pooling(out)

        out = out + out_res

        return out

'''
Based on BIG gan from [Large Scale GAN Training for High Fidelity Natural Image Synthesis]
https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py#L103
'''
class ResidualBlockG_Down(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlockG_Down, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv_l = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))

        self.ins_r1 = nn.InstanceNorm2d(num_features=in_dim, affine=True)
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))
        self.ins_r2 = nn.InstanceNorm2d(num_features=out_dim, affine=True)
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1))

        self.relu = nn.ReLU(inplace=False)

        self.avg_pooling = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        '''
        Down sample residual block for generator using instance normalization
        :param x: [B, C_in, H, W]
        :return: [B, C_out, H, W]
        '''
        out_res = self.conv_l(x)
        out_res = self.avg_pooling(out_res)

        out = self.ins_r1(x)
        out = self.relu(out)
        out = self.conv_r1(out)
        out = self.ins_r2(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        out = self.avg_pooling(out)

        out = out + out_res

        return out


class ResidualBlockG_Regular(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlockG_Regular, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.adaIn = AdaIn()

        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1))

        self.relu = nn.ReLU(inplace=False)

        self.skip_conv = (in_dim != out_dim)
        if (self.skip_conv):
            self.conv_l = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))


    def forward(self, x, slice):
        '''
        Regular residual block for generator using adaptive instance normalization
        :param x: [B, C_in, H, W]
        :param mean_style_1: [B, C_in, 1]
        :param std_style_1: [B, C_in, 1]
        :param mean_style_2: [B, C_out, 1]
        :param std_style_2: [B, C_out, 1]
        :return: [B, C_out, H, W]
        '''
        mean_style_1, std_style_1, mean_style_2, std_style_2 = getMeanAndStdFromSlice(slice, self.in_dim, self.out_dim)

        if(self.skip_conv):
            out_res = self.conv_l(x)
        else:
            out_res = x

        out = self.adaIn(x, mean_style_1, std_style_1)
        out = self.relu(out)
        out = self.conv_r1(out)
        out = self.adaIn(out, mean_style_2, std_style_2)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


class ResidualBlockG_Up(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlockG_Up, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.adaIn = AdaIn()

        self.conv_l = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))

        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1))

        self.relu = nn.ReLU(inplace=False)

        self.up_sampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, slice):
        '''
        Up sample residual block for generator using adaptive instance normalization
        :param x: [B, C_in, H, W]
        :param mean_style_1: [B, C_in, 1]
        :param std_style_1: [B, C_in, 1]
        :param mean_style_2: [B, C_out, 1]
        :param std_style_2: [B, C_out, 1]
        :return: [B, C_out, H, W]
        '''
        mean_style_1, std_style_1, mean_style_2, std_style_2 = getMeanAndStdFromSlice(slice, self.in_dim, self.out_dim)

        out_res = self.up_sampling(x)
        out_res = self.conv_l(out_res)

        out = self.adaIn(x, mean_style_1, std_style_1)
        out = self.relu(out)
        out = self.up_sampling(out)
        out = self.conv_r1(out)
        out = self.adaIn(out, mean_style_2, std_style_2)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out

def getMeanAndStdFromSlice(slice, in_dim, out_dim):
    '''
    get mean_1, std_1, mean_2, std_2
    :param slice: [B, C, 1]
    :param in_dim:
    :param out_dim:
    :return:
    '''
    mean_1= slice[:, :in_dim, :]
    std_1 = slice[:, in_dim:in_dim*2, :]
    mean_2 = slice[:, in_dim*2:in_dim*2+out_dim, :]
    std_2 = slice[:, in_dim*2+out_dim:, :]

    return mean_1, std_1, mean_2, std_2