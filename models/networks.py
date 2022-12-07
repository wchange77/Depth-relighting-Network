import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .Modules_chutak import MultiDilationResnetBlock_attention
#from fusion import AFF, iAFF, DAF
from .base_model_DMSN import UpBlock, DownBlock
import torch.nn.functional as F


from lib.nn import SynchronizedBatchNorm2d as SynBN2d
import torch.nn.utils.spectral_norm as spectral_norm

###############################################################################
# Functions
###############################################################################
def pad_tensor(input):
    
    height_org = input.shape[2]
    width_org = input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height = input.data.shape[2]
    width = input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height = input.shape[2]
    width = input.shape[3]
    x = height - pad_bottom
    y = width - pad_right
    return input[:,:, pad_top: x, pad_left: y]

'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
'''

def weights_init(m):     # 初始化权重
     if isinstance(m, nn.Conv2d):
         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
         torch.nn.init.constant_(m.bias.data, 0.0)



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(SynBN2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        #netG = Res_HDR(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
        netG = Res_Light(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
        #netG = LightTrans(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
        #netG = Gategen(input_nc, output_nc, ngf)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if use_gpu:
        netG.cuda(device=gpu_ids[0])

    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    norm_layer = get_norm_layer(norm_type=norm)
    #netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    netD = Bag_Shad_Discriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) >= 0:
        netD.cuda(device=gpu_ids[0])
    
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=SynBN2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class LightTrans(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        super(LightTrans, self).__init__()

        self.toHDR = BP_HDR(input_nc, output_nc, ngf=32, n_downsampling=n_downsampling, n_blocks=n_blocks, norm_layer=norm_layer,
                            padding_type=padding_type)
        #self.estLight = BP_Light(input_nc, output_nc, ngf=32, n_downsampling=n_downsampling, n_blocks=n_blocks, norm_layer=norm_layer,
                            #padding_type=padding_type)

        #self.estLight_channel = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(32, 64, kernel_size=3, padding=0), nn.ReLU())
        #self.toHDR_channel = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(32, 64, kernel_size=3, padding=0), nn.ReLU())

        #self.dropout_1 = nn.Dropout(0.95)
        #self.dropout_2 = nn.Dropout(0.95)

        #self.comb =MultiDilationResnetBlock_attention(ngf)

        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, output_nc, kernel_size=7, padding=0), nn.Tanh())
        
        #for param in self.toHDR.parameters():
            #param.requires_grad = False
            
        #for param in self.estLight.parameters():
             #param.requires_grad = False
        
        # self.toHDR.load_state_dict(torch.load("checkpoints/Pretrained/hdr_v2_net_pretrainedG.pth", map_location=lambda storage, loc: storage),
        #                            strict=False)
        # self.estLight.load_state_dict(torch.load("checkpoints/Pretrained/relight_net_pretrinedG.pth", map_location=lambda storage, loc: storage),
        #                            strict=False)

    def forward(self, input):
        m_hdr = self.toHDR(input)
        #m_estLight = self.estLight(input)

        #m_hdr = self.toHDR_channel(m_hdr)
        #m_estLight = self.estLight_channel(m_estLight)

        m_hdr = self.dropout_1(m_hdr)
        #m_estLight = self.dropout_2(m_estLight)

        #m_fea = self.comb(x_hdr=m_hdr, x_relight=m_estLight)
        #out = self.out(m_fea)
        out = self.out(m_hdr)
        return out
    def load_newhdrmodel(self):
        return
        # self.toHDR.load_state_dict(torch.load("checkpoints/pretrained/BPHDR29_net_G.pth", map_location=lambda storage, loc: storage),
        #                            strict=True)
        # print("load model")
        # self.estLight.load_state_dict(torch.load("checkpoints/pretrained/BPLight22_net_G.pth", map_location=lambda storage, loc: storage),
        #                            strict=True)





class BP_HDR_gg(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(BP_HDR, self).__init__()
        activation = nn.ReLU(True)

        Inlayer = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                   activation]
        ### downsample
        '''
        Encoder = []
        for i in range(n_downsampling):
            mult = 2**i
            Encoder += [DownBlock(ngf * mult)]
        '''
        self.encoder1 = DownBlock(ngf*(2**0))
        self.encoder2 = DownBlock(ngf*(2**1))
        self.encoder3 = DownBlock(ngf*(2**2))
        self.encoder4 = DownBlock(ngf*(2**3))



        ###############################
        
        ###############################


        ### resnet blocks
        Manipulate = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            Manipulate += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        '''
        Decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
            #                              output_padding=1),
            #           norm_layer(int(ngf * mult / 2)), activation]
            Decoder += [UpBlock(ngf * mult)]
        '''

        self.decoder1 = UpBlock(ngf*(2**(4-0)))
        self.decoder2 = UpBlock(ngf*(2**(4-1)))
        self.decoder3 = UpBlock(ngf*(2**(4-2)))
        self.decoder4 = UpBlock(ngf*(2**(4-3)))

        self.multi_scale_1 = nn.ConvTranspose2d(ngf * (2**(4 - 1)), ngf * (2**(4 - 4)), kernel_size=8, stride=8, bias=True)
        self.multi_scale_2 = nn.ConvTranspose2d(ngf * (2**(4 - 2)), ngf * (2**(4 - 4)), kernel_size=4, stride=4, padding=0, bias=True)
        self.multi_scale_3 = nn.ConvTranspose2d(ngf * (2**(4 - 3)), ngf * (2**(4 - 4)), kernel_size=4, stride=2, padding=1, bias=True)
        #self.multi_scale_4 = nn.ConvTranspose2d(ngf * (2**(4 - 3)), ngf * (2**(4 - 3)), kernel_size=4, stride=, padding=1, bias=True)
        self.cat = nn.Conv2d(ngf * (2**(4 - 4))*4, ngf * (2**(4 - 4)), kernel_size=3, padding=1)


        self.detail = Detail_Net()

        self.inlayer = nn.Sequential(*Inlayer)
        #self.encoder = nn.Sequential(*Encoder)
        self.manipulate = nn.Sequential(*Manipulate)
        #self.decoder = nn.Sequential(*Decoder)

        #self.dropout = nn.Dropout(0.85)
        self.shortconect = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=0), nn.ReLU())
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())


    def forward(self, input):
        inlayer = self.inlayer(input)
        #m = self.encoder(m_in)
        feature_d1 = self.encoder1(inlayer)
        feature_d2 = self.encoder2(feature_d1)
        feature_d3 = self.encoder3(feature_d2)
        feature_d4 = self.encoder4(feature_d3)

        m = self.manipulate(feature_d4)
        #m = self.decoder(m)
        feature_u1 = self.decoder1(m)
        #print('feature_u1',feature_u1.shape)
        feature_u2 = self.decoder2(feature_u1)
        #print('feature_u2',feature_u2.shape)
        feature_u3 = self.decoder3(feature_u2)
        #print('feature_u3',feature_u3.shape)
        feature_u4 = self.decoder4(feature_u3)
        #print('feature_u4',feature_u4.shape)

        scale1 = self.multi_scale_1(feature_u1)
        #print('scale1',scale1.shape)
        scale2 = self.multi_scale_2(feature_u2)
        #print('scale2',scale2.shape)
        scale3 = self.multi_scale_3(feature_u3)
        #print('scale3',scale3.shape)
        ##########
        #y = detail(input)
        #########


        feature_u4 = torch.cat([scale1,scale2,scale3,feature_u4],dim=1)
        feature_u4 = self.cat(feature_u4)
        #m = self.dropout(m)
        m = torch.cat([inlayer,feature_u4],dim=1)
        m = self.shortconect(m)
        #out = self.out(m)
        return m

    # Define a resnet block



#############################单独训练重构网络####################################
class BP_HDR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(BP_HDR, self).__init__()
        activation = nn.ReLU(True)

        Inlayer = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                   activation]

        self.encoder1 = DownBlock(ngf*(2**0))
        self.encoder2 = DownBlock(ngf*(2**1))
        self.encoder3 = DownBlock(ngf*(2**2))
        self.encoder4 = DownBlock(ngf*(2**3))


        ### resnet blocks
        Manipulate = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            Manipulate += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]



        self.decoder1 = UpBlock(ngf*(2**(4-0)))
        self.decoder2 = UpBlock(ngf*(2**(4-1)))
        self.decoder3 = UpBlock(ngf*(2**(4-2)))
        self.decoder4 = UpBlock(ngf*(2**(4-3)))

        self.multi_scale_1 = nn.ConvTranspose2d(ngf * (2**(4 - 1)), ngf * (2**(4 - 4)), kernel_size=8, stride=8, bias=True)
        self.multi_scale_2 = nn.ConvTranspose2d(ngf * (2**(4 - 2)), ngf * (2**(4 - 4)), kernel_size=4, stride=4, padding=0, bias=True)
        self.multi_scale_3 = nn.ConvTranspose2d(ngf * (2**(4 - 3)), ngf * (2**(4 - 4)), kernel_size=4, stride=2, padding=1, bias=True)

        self.cat = nn.Conv2d(ngf * (2**(4 - 4))*4, ngf * (2**(4 - 4)), kernel_size=3, padding=1)

        self.inlayer = nn.Sequential(*Inlayer)

        self.manipulate = nn.Sequential(*Manipulate)


        #self.dropout = nn.Dropout(0.85)
        ################################
        self.dropout = nn.Dropout(0.85)
        ################################
        self.shortconect = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=0), nn.ReLU())
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())


    def forward(self, input):
        inlayer = self.inlayer(input)
        #m = self.encoder(m_in)
        feature_d1 = self.encoder1(inlayer)
        feature_d2 = self.encoder2(feature_d1)
        feature_d3 = self.encoder3(feature_d2)
        feature_d4 = self.encoder4(feature_d3)

        m = self.manipulate(feature_d4)
        #m = self.decoder(m)
        feature_u1 = self.decoder1(m)
        feature_u2 = self.decoder2(feature_u1)
        feature_u3 = self.decoder3(feature_u2)
        feature_u4 = self.decoder4(feature_u3)


        m = self.dropout(m)

        m = torch.cat([inlayer,feature_u4],dim=1)
        m = self.shortconect(m)
        ####################################
        out = self.out(m)
        ####################################
        #out = self.out(m)
        return out



class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        #model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

    # Define a resnet block
class BP_Light(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(BP_Light, self).__init__()
        activation = nn.ReLU(True)
        for m in self.modules():weights_init(m)


        Inlayer = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                   activation]
        ### downsample
        Encoder = []
        for i in range(n_downsampling):
            mult = 2**i
            # model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
            #           norm_layer(ngf * mult * 2), activation]
            Encoder += [DownBlock(ngf * mult)]

        
        ##########################################################################
        self.gate_1 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_2 = gate_block(input_nc, ngf,transformer = 0)
        self.gate_3 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_4 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_5 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_6 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_7 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_8 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_9 = gate_block(input_nc, ngf, transformer= 0)
        ##########################################################################
        

        ### upsample
        Decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
            #                              output_padding=1),
            #           norm_layer(int(ngf * mult / 2)), activation]
            Decoder += [UpBlock(ngf * mult)]
        out = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        #Decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.inlayer = nn.Sequential(*Inlayer)
        self.encoder = nn.Sequential(*Encoder)
        
        self.decoder = nn.Sequential(*Decoder)
        #self.manipulate = nn.Sequential(*Manipulate)
        self.out = nn.Sequential(*out)

    def forward(self, input):
        
        m = self.inlayer(input)
        m = self.encoder(m)
        
        ##################################################################
        m = self.gate_1(m)
        m = self.gate_2(m)
        m = self.gate_3(m)
        m = self.gate_4(m)
        m = self.gate_5(m)
        m = self.gate_6(m)
        m = self.gate_7(m)
        m = self.gate_8(m)
        m = self.gate_9(m)
        ##################################################################



        m = self.decoder(m)
        #print(m.shape)
        out = self.out(m)
        return out
        '''
        m = self.inlayer(input)
        m = self.encoder(m)
        m = self.manipulate(m)

        for idx, layer in enumerate(self.decoder):
            m = layer(m)
            #if idx == 3:
                #break
        #out = self.decoder(m)
        return m
        '''


'''
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        use_bias = False
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
'''

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=SynBN2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean

'''
## Shadow Discriminator, Liwen
class Bag_Shad_Discriminator(nn.Module):
    def __init__(self, nput_nc, ndf=64, n_layers=3, norm_layer=SynBN2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(Bag_Shad_Discriminator, self).__init__()
        for m in self.modules():weights_init(m)
        self.multiDis = MultiscaleDiscriminator(nput_nc, ndf, n_layers, norm_layer,
                                                use_sigmoid, num_D, getIntermFeat)
        self.shadowDis = MultiscaleDiscriminator(3 + 1, ndf, n_layers, norm_layer,
                                                 use_sigmoid, num_D, getIntermFeat)
        return

    def forward(self, input):
        res_1 = self.multiDis(input)
        # for shadow adversiral loss
        img_in = input[:, :3, :, :]
        img2eval = input[:, 3:, :, :]
        img2eval_grey = torch.mean(img2eval, dim=1, keepdim=True)
        img2eval_grey[img2eval_grey > (15 / 255.0)] = 0
        mask_in = torch.cat([img_in, img2eval_grey], dim=1)
        res_2 = self.shadowDis(mask_in)
        result = res_1 + res_2
        return result


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=SynBN2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=SynBN2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

'''


## Shadow Discriminator, Liwen
class Bag_Shad_Discriminator(nn.Module):
    def __init__(self, nput_nc, ndf=64, n_layers=3, norm_layer=SynBN2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(Bag_Shad_Discriminator, self).__init__()
        self.multiDis = MultiscaleDiscriminator(nput_nc, ndf, n_layers, norm_layer,
                                                use_sigmoid, num_D, getIntermFeat)
        self.shadowDis = MultiscaleDiscriminator(3 + 1, ndf, n_layers, norm_layer,
                                                 use_sigmoid, num_D, getIntermFeat)
        return

    def forward(self, input):
        res_1 = self.multiDis(input)
        # for shadow adversiral loss
        img_in = input[:, :3, :, :]
        img2eval = input[:, 3:, :, :]
        img2eval_grey = torch.mean(img2eval, dim=1, keepdim=True)
        img2eval_grey[img2eval_grey > (15 / 255.0)] = 0
        mask_in = torch.cat([img_in, img2eval_grey], dim=1)
        res_2 = self.shadowDis(mask_in)
        result = res_1 + res_2
        return result


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=SynBN2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=SynBN2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        
        
        

        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 28):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        #h_relu4 = self.slice4(h_relu3)
        #h_relu5 = self.slice5(h_relu4)
        #out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out = [h_relu1, h_relu2, h_relu3]#, h_relu4, h_relu5]
        return out


#####################################################################################################################################################################################


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    #expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes, norm_layer=SynBN2d):
        super(SCBottleneck, self).__init__()
        half_planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, half_planes, 1, 1, bias=False)
        self.bn1_a = norm_layer(half_planes)

        self.k1 = nn.Sequential(
                    nn.Conv2d(half_planes, half_planes, 3, 1, 1), 
                    nn.LeakyReLU(0.2),
                    )

        self.conv1_b = nn.Conv2d(in_planes, half_planes, 1, 1, bias=False)
        self.bn1_b = norm_layer(half_planes)
        
        self.scconv = SCConv(half_planes, self.pooling_r)

        self.conv3 = nn.Conv2d(half_planes * 2, half_planes * 2, 1, 1, bias=False)
        self.bn3 = norm_layer(half_planes * 2)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

        
class Res_HDR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Res_HDR, self).__init__()
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)

        #Inlayer = [nn.Conv2d(input_nc+1, ngf, 3, padding=1), norm_layer(ngf),activation,
                        #SCBottleneck(ngf, ngf)]

        

        self.reflect = nn.ReflectionPad2d(3)
        
        Inlayer = [nn.Conv2d(input_nc+1, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                   activation,SCBottleneck(ngf, ngf)]

        Encoder1 = []
        mult = 2**0
        Encoder1 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder1 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder2 = []
        mult = 2**1
        Encoder2 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder2 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder3 = []
        mult = 2**2
        Encoder3 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder3 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder4 = []
        mult = 2**3
        Encoder4 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder4 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]


        ##########################################################################
        self.gate_1 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_2 = gate_block(input_nc, ngf,transformer = 0)
        self.gate_3 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_4 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_5 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_6 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_7 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_8 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_9 = gate_block(input_nc, ngf, transformer= 0)
        ##########################################################################



        Decoder4 = []
        mult = 2**4
        Decoder4 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder4 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder3 = []
        mult = 2**3
        Decoder3 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder3 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder2 = []
        mult = 2**2
        Decoder2 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder2 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder1 = []
        mult = 2**1
        Decoder1 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder1 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3,  padding=1),
                        activation]


        
        self.dropout = nn.Dropout(0.85)
        ################################
        self.shortconect = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=0), activation)
        


        #self.conv10 = nn.Conv2d(ngf, output_nc, 1)
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())



        self.deconv4 = nn.Conv2d(ngf*(2**4), ngf*(2**3), 3, padding=1)
        self.deconv3 = nn.Conv2d(ngf*(2**3), ngf*(2**2), 3, padding=1)
        self.deconv2 = nn.Conv2d(ngf*(2**2), ngf*(2**1), 3, padding=1)
        self.deconv1 = nn.Conv2d(ngf*(2**1), ngf*(2**0), 3, padding=1)


        self.inlayer = nn.Sequential(*Inlayer)

        self.Decoder1 = nn.Sequential(*Decoder1)
        self.Decoder2 = nn.Sequential(*Decoder2)
        self.Decoder3 = nn.Sequential(*Decoder3)
        self.Decoder4 = nn.Sequential(*Decoder4)

        self.Encoder1 = nn.Sequential(*Encoder1)
        self.Encoder2 = nn.Sequential(*Encoder2)
        self.Encoder3 = nn.Sequential(*Encoder3)
        self.Encoder4 = nn.Sequential(*Encoder4)

    def forward(self, input, gray):

        #input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        #print('input', input.shape)
        #gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        input = self.reflect(input)

        gray1 = self.reflect(gray)
        print('gray',gray.shape)
        gray_2 = self.downsample_1(gray)
        print('gray2',gray_2.shape)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
  

        x = torch.cat((input, gray1), 1)

        inlayer = self.inlayer(x)
        print('inlayer', inlayer.shape)

        x1 = self.Encoder1(inlayer)
        print(x1.shape)
        x2 = self.Encoder2(x1)
        print(x2.shape)
        x3 = self.Encoder3(x2)
        print(x3.shape)
        x4 = self.Encoder4(x3)
        print(x4.shape)

        ##################################################################
        x4 = self.gate_1(x4)
        x4 = self.gate_2(x4)
        x4 = self.gate_3(x4)
        x4 = self.gate_4(x4)
        x4 = self.gate_5(x4)
        x4 = self.gate_6(x4)
        x4 = self.gate_7(x4)
        x4 = self.gate_8(x4)
        x4 = self.gate_9(x4)
        ##################################################################


        x5= F.upsample(x4, scale_factor=2, mode='bilinear')
        x3 = x3*gray_4                                                                     
        up4 = torch.cat([self.deconv4(x5), x3], 1)
        up44 = self.Decoder4(up4)

        x6 = F.upsample(up44, scale_factor=2, mode='bilinear')
        x2 = x2*gray_3                                                                     
        up5 = torch.cat([self.deconv3(x6), x2], 1)
        up55 = self.Decoder3(up5)

        x7 = F.upsample(up55, scale_factor=2, mode='bilinear')
        x1 = x1*gray_2                                                                     
        up6 = torch.cat([self.deconv2(x7), x1], 1)
        up66 = self.Decoder2(up6)

        x8 = F.upsample(up66, scale_factor=2, mode='bilinear')
        #print('x',x.shape)
        #print('gray', gray.shape)
        x = inlayer*gray
        #x88 = self.deconv1(x8) 
        #x88 = self.reflect(x88)
                                                                      
        up7 = torch.cat([self.deconv1(x8), x], 1)
        #up7 = torch.cat([x88, x], 1)
        #print('x88', x88.shape)
        #print('up7', up7.shape)
        up77 = self.Decoder1(up7)

        #print('up77', up77.shape)
        m = self.out(up77)
        #print('m',m.shape)

        #latent = self.conv10(up77)
        #latent = latent*gray
        #output = latent + input
        
        #output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        #latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        #gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)

        
        return m, gray


class Res_Light(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=SynBN2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Res_Light, self).__init__()
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)


        

        self.reflect = nn.ReflectionPad2d(3)
        
        Inlayer = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                   activation,SCBottleneck(ngf, ngf)]

  

        Encoder1 = []
        mult = 2**0
        Encoder1 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder1 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder2 = []
        mult = 2**1
        Encoder2 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder2 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder3 = []
        mult = 2**2
        Encoder3 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder3 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]

        Encoder4 = []
        mult = 2**3
        Encoder4 += [nn.Conv2d(ngf * mult, ngf * mult *2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult*2), activation]
        Encoder4 += [SCBottleneck(ngf * mult *2 , ngf * mult *2)]


        ##########################################################################
        self.gate_1 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_2 = gate_block(input_nc, ngf,transformer = 0)
        self.gate_3 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_4 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_5 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_6 = gate_block(input_nc, ngf,transformer = 1)
        self.gate_7 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_8 = gate_block(input_nc, ngf, transformer= 0)
        self.gate_9 = gate_block(input_nc, ngf, transformer= 0)
        ##########################################################################



        Decoder4 = []
        mult = 2**4
        Decoder4 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder4 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder3 = []
        mult = 2**3
        Decoder3 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder3 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder2 = []
        mult = 2**2
        Decoder2 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder2 += [SCBottleneck(int(ngf * mult / 2), int(ngf * mult / 2))]

        Decoder1 = []
        mult = 2**1
        Decoder1 += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,  padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        Decoder1 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3,  padding=1),
                        activation]


        
        self.dropout = nn.Dropout(0.85)
        ################################
        self.shortconect = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=0), activation)
        


        #self.conv10 = nn.Conv2d(ngf, output_nc, 1)
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())



        self.deconv4 = nn.Conv2d(ngf*(2**4), ngf*(2**3), 3, padding=1)
        self.deconv3 = nn.Conv2d(ngf*(2**3), ngf*(2**2), 3, padding=1)
        self.deconv2 = nn.Conv2d(ngf*(2**2), ngf*(2**1), 3, padding=1)
        self.deconv1 = nn.Conv2d(ngf*(2**1), ngf*(2**0), 3, padding=1)


        self.inlayer = nn.Sequential(*Inlayer)

        self.Decoder1 = nn.Sequential(*Decoder1)
        self.Decoder2 = nn.Sequential(*Decoder2)
        self.Decoder3 = nn.Sequential(*Decoder3)
        self.Decoder4 = nn.Sequential(*Decoder4)

        self.Encoder1 = nn.Sequential(*Encoder1)
        self.Encoder2 = nn.Sequential(*Encoder2)
        self.Encoder3 = nn.Sequential(*Encoder3)
        self.Encoder4 = nn.Sequential(*Encoder4)

    def forward(self, input, gray):


        input = self.reflect(input)

        gray1 = self.reflect(gray)
        gray_2 = self.downsample_1(gray)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
  



        inlayer = self.inlayer(input)

        x1 = self.Encoder1(inlayer)
        print(x1.shape)
        x2 = self.Encoder2(x1)
        print(x2.shape)
        x3 = self.Encoder3(x2)
        print(x3.shape)
        x4 = self.Encoder4(x3)
        print(x4.shape)

        ##################################################################
        x4 = self.gate_1(x4)
        x4 = self.gate_2(x4)
        x4 = self.gate_3(x4)
        x4 = self.gate_4(x4)
        x4 = self.gate_5(x4)
        x4 = self.gate_6(x4)
        x4 = self.gate_7(x4)
        x4 = self.gate_8(x4)
        x4 = self.gate_9(x4)
        ##################################################################


        x5= F.upsample(x4, scale_factor=2, mode='bilinear')
        up44 = self.Decoder4(x5)

        x6 = F.upsample(up44, scale_factor=2, mode='bilinear')
        up55 = self.Decoder3(x6)

        x7 = F.upsample(up55, scale_factor=2, mode='bilinear')
        up66 = self.Decoder2(x7)

        x8 = F.upsample(up66, scale_factor=2, mode='bilinear')
        up77 = self.Decoder1(x8)


        m = self.out(up77)
        return m, gray
















class gate_block(nn.Module):
    def __init__(self, input_nc , ngf , transformer = 0):
        super(gate_block, self).__init__()
        self.transformer = transformer
   
        
        use_bias = False
        norm_layer = SynBN2d
        padding_type = 'reflect'
        if self.transformer == 1:
            self.gate = NAFBlock(ngf*(2**4))
 
            #Channel compression
            self.cc = channel_compression(ngf * (2**5), ngf * (2**4))
        # Residual CNN
        model = [ResnetBlock(ngf * (2**4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False
                            )]
                             
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer == 1 :
            tr_out = self.gate(x)
           
            # concat transformer output and resnet output
            x = torch.cat([tr_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x





class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        #self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = x_se.sigmoid()
        return x * x_se


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate


        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)


        
        x1, x2 = self.conv2(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2


        x = x * self.sca(x)


        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x 


        x = self.norm2(y)
        x3, x4 = self.conv4(x).chunk(2, dim=1)
        x = F.gelu(x3) * x4
  


        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x 


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out



##############################################################D###############################################################


###########################################################D####################################################################