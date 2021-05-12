
import torch
import torch.nn as nn
import torch.nn.functional as F

from reltextgan.unet import UnetGenerator, Discriminator, ImageToEncoding

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view([x.shape[0]] + list(self.shape))
    
# class EncodingToImage(nn.Module):
#     def __init__(self, encoding_size, ngf=64, output_nc=1):
#         super(EncodingToImage, self).__init__()
#         model = [nn.Linear(encoding_size, 4*16**2),
#                  nn.LeakyReLU(0.2, True),
#                  nn.Linear(4*16**2, 4*16**2),
#                  Reshape(4,16,16)]
#         i = 0
#         while 16*2**i < 128:
#             model += [nn.ConvTranspose2d(4 if i==0 else ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), 
#                       nn.Tanh()]
#             i += 1
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        
#         # #self.dense = nn.Linear(512, 3*16**2)
#         # self.reshaped = Reshape(2,16,16)
#         # self.upsample1 = upBlock(2, ngf)
#         # self.upsample2 = upBlock(ngf, ngf // 2)
#         # self.upsample3 = upBlock(ngf // 2, ngf // 4)
#         # self.img_out = nn.Sequential(
#         #     conv3x3(ngf // 4, output_nc),
#         #     nn.Tanh()
#         # )
#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)
#         # #out_code = self.dense(input)
#         # out_code = self.reshaped(input)
#         # out_code = self.upsample1(out_code)
#         # out_code = self.upsample2(out_code)
#         # out_code = self.upsample3(out_code)
#         # out_code = self.img_out(out_code)

#         # return out_code


# class RelTextGenerator(nn.Module):
#     def __init__(self, image_size, ngf=64):
#         super(RelTextGenerator, self).__init__()
        
#         self.text2img = EncodingToImage(image_size, ngf=ngf)
#         self.generator = UnetGenerator(4, 3, 6, ngf=ngf)

#     def forward(self, imgs, encodings_diff, text_encoding):
#         t = self.text2img(torch.cat((encodings_diff, text_encoding), 1))
#         return self.generator(torch.cat((imgs, t), 1))

class EncodingToImage(nn.Module):
    def __init__(self, emb_size, image_size, ngf=64, output_nc=1):
        super(EncodingToImage, self).__init__()
        model = [nn.Linear(emb_size, 3*32**2),
                 Reshape(3,32,32)]
        i = 0
        while 32*2**i < image_size:
            model += [nn.ConvTranspose2d(3 if i==0 else ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1)]
            i += 1
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class RelTextGenerator(nn.Module):
    def __init__(self, emb_size, image_size, ngf=64):
        super(RelTextGenerator, self).__init__()
        
        self.text2img = EncodingToImage(emb_size, image_size, ngf=ngf)
        self.generator = UnetGenerator(4, 3, 6, ngf=ngf)

    def forward(self, imgs, sent_emb):
        sent2img = self.text2img(sent_emb)
        return self.generator(torch.cat((imgs, sent2img), 1))

class RelTextDiscriminator(nn.Module):
    def __init__(self, emb_size, image_size, ngf=64):
        super(RelTextDiscriminator, self).__init__()
        
        self.text2img = EncodingToImage(emb_size, image_size, ngf=ngf)
        self.discrim = Discriminator(image_size, ndf=ngf)

    def forward(self, imgs, encodings):
        t = self.text2img(encodings)
        return self.discrim(torch.cat((imgs, t), 1))

class AlignmentDiscriminator(nn.Module):
    def __init__(self, emb_size, ngf=64):
        super(AlignmentDiscriminator, self).__init__()
        
        model = [
                nn.Linear(emb_size*3, emb_size*2),
                nn.LeakyReLU(0.2, True),
                nn.Linear(emb_size*2, emb_size),
                nn.LeakyReLU(0.2, True),
                nn.Linear(emb_size, 1),
                nn.Sigmoid()
        ]
        self.model =  nn.Sequential(*model)

    def forward(self, img0_encoding, img1_encoding, text_encoding_difference):
        # print(img_encoding.shape, text_encoding.shape, torch.cat((img_encoding, text_encoding), 1).shape)
        return self.model(torch.cat((img0_encoding, img1_encoding, text_encoding_difference), 1))


# CLIP encodings to image
# https://github.com/taoxugit/AttnGAN/blob/master/code/model.py

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

class ClipEncodings2Image(nn.Module):
    def __init__(self, ngf):
        super(ClipEncodings2Image, self).__init__()
        self.gf_dim = ngf
        self.in_dim = 512*2

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)
        self.upsample6 = upBlock(ngf // 16, ngf // 16)

        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3)
        )

    def forward(self, image_encoding, text_encoding):
        """
        :param image_encoding: batch x 512
        :param text_encoding: batch x 512
        :return: batch x 3 x 128 x 128
        """
        c_z_code = torch.cat((image_encoding, text_encoding), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)
        # state size ngf/16 x 128 x 128
        out_code128 = self.upsample5(out_code64)
        # state size ngf/16 x 128 x 128
        out_code256 = self.upsample6(out_code128)

        out_img = self.img(out_code256)

        return out_img