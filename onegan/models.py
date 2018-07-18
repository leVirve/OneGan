import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


log = logging.getLogger('onegan.model')


def init_weights(net, init_method='normal', gain=1):

    def init_module_weight(m):
        module_name = m.__class__.__name__
        module_name = module_name.replace('1d', '').replace('2d', '').replace('3d', '')

        if module_name == 'BatchNorm':
            nn.init.uniform_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif module_name in ('Conv', 'Linear'):
            init_func(m)

    init_func = {
        'normal': lambda x: nn.init.uniform_(x.weight.data, 0.0, 0.02),
        'kaiming': lambda x: nn.init.kaiming_normal_(x.weight.data, a=0, mode='fan_in'),
        'xavier': lambda x: nn.init.xavier_normal_(x.weight.data, gain=gain),
        'orthogonal': lambda x: nn.init.orthogonal_(x.weight.data, gain=gain),
    }.get(init_method)

    if init_func is None:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_method)

    log.info('initialization method [%s]' % init_method)
    net.apply(init_module_weight)


class GeneratorUNet(nn.Module):

    def __init__(self, input_channel, output_channel, ngf, norm='batch'):
        """ Generator model

        Args:
            input_channel: input image dimension
            output_channel: output image dimension
            ngf: the number of filters
        """
        super().__init__()
        bias = True if norm == 'instance' else False

        self.enc1 = nn.Conv2d(input_channel, ngf, kernel_size=4, stride=2, padding=1, bias=bias)
        self.enc2 = self._make_encode_layer(ngf, ngf * 2, bias)
        self.enc3 = self._make_encode_layer(ngf * 2, ngf * 4, bias)
        self.enc4 = self._make_encode_layer(ngf * 4, ngf * 8, bias)
        self.enc5 = self._make_encode_layer(ngf * 8, ngf * 8, bias)
        self.enc6 = self._make_encode_layer(ngf * 8, ngf * 8, bias)
        self.enc7 = self._make_encode_layer(ngf * 8, ngf * 8, bias)
        self.enc8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias),
        )

        self.dec1 = self._make_decode_layer(ngf * 8, ngf * 8, bias, dropout=True)
        self.dec2 = self._make_decode_layer(ngf * 8 * 2, ngf * 8, bias, dropout=True)
        self.dec3 = self._make_decode_layer(ngf * 8 * 2, ngf * 8, bias, dropout=True)
        self.dec4 = self._make_decode_layer(ngf * 8 * 2, ngf * 8, bias)
        self.dec5 = self._make_decode_layer(ngf * 8 * 2, ngf * 4, bias)
        self.dec6 = self._make_decode_layer(ngf * 4 * 2, ngf * 2, bias)
        self.dec7 = self._make_decode_layer(ngf * 2 * 2, ngf, bias)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channel, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.InstanceNorm2d(output_channel) if bias else nn.BatchNorm2d(output_channel),
        )

    @staticmethod
    def _make_encode_layer(in_dim, out_dim, bias):
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.InstanceNorm2d(out_dim) if bias else nn.BatchNorm2d(out_dim),
        )

    @staticmethod
    def _make_decode_layer(in_dim, out_dim, bias, dropout=False):
        layers = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.InstanceNorm2d(out_dim) if bias else nn.BatchNorm2d(out_dim),
        ]
        if dropout:
            layers += [nn.Dropout(0.5)]
        return nn.Sequential(*layers)

    def forward(self, x):
        ''' Encoder '''
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        ''' Decoder '''
        d1 = torch.cat((self.dec1(e8), e7), dim=1)
        d2 = torch.cat((self.dec2(d1), e6), dim=1)
        d3 = torch.cat((self.dec3(d2), e5), dim=1)
        d4 = torch.cat((self.dec4(d3), e4), dim=1)
        d5 = torch.cat((self.dec5(d4), e3), dim=1)
        d6 = torch.cat((self.dec6(d5), e2), dim=1)
        d7 = torch.cat((self.dec7(d6), e1), dim=1)
        d8 = self.dec8(d7)

        output = F.tanh(d8)
        return output


class Discriminator(nn.Module):
    '''
    -- if n=0, then use pixelGAN (rf=1)
    -- else rf is 16 if n=1
    --            34 if n=2
    --            70 if n=3
    --            142 if n=4
    --            286 if n=5
    --            574 if n=6
    '''

    def __init__(self, input_channel, output_channel, ndf, n_layers=3, norm='batch'):
        """ Discriminator model

        Args:
            input_channel: input image dimension
            output_channel: output image dimension
            ngf: the number of filters
        """
        super().__init__()
        bias = True if norm == 'instance' else False
        layers = [
            nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=bias),
                nn.InstanceNorm2d(ndf * nf_mult) if bias else nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(ndf * nf_mult) if bias else nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
