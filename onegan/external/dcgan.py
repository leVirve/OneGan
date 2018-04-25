import torch
import torch.nn as nn
import torch.nn.parallel


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size, z_dim, input_channel, ndf=64, ngpu=1, n_extra_layers=0):
        super().__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x image_size x image_size
        main.add_module('initial_conv_{0}-{1}'.format(input_channel, ndf),
                        nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = image_size / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGANGenerator(nn.Module):
    def __init__(self, image_size, z_dim, output_channel, ngf=64, ngpu=1, n_extra_layers=0):
        super().__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        cngf, timage_size = ngf//2, 4
        while timage_size != image_size:
            cngf = cngf * 2
            timage_size = timage_size * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(z_dim, cngf),
                        nn.ConvTranspose2d(z_dim, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < image_size//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, output_channel),
                        nn.ConvTranspose2d(cngf, output_channel, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(output_channel),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output


class DCGANDiscriminatorNobn(nn.Module):
    def __init__(self, image_size, dim_z, input_channel, ndf=64, ngpu=1, n_extra_layers=0):
        super().__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x image_size x image_size
        # input is nc x image_size x image_size
        main.add_module('initial_conv_{0}-{1}'.format(input_channel, ndf),
                        nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = image_size / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGANGeneratorNobn(nn.Module):
    def __init__(self, image_size, z_dim, output_channel, ngf=64, ngpu=1, n_extra_layers=0):
        super().__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        cngf, timage_size = ngf//2, 4
        while timage_size != image_size:
            cngf = cngf * 2
            timage_size = timage_size * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(z_dim, cngf),
                        nn.ConvTranspose2d(z_dim, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < image_size//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, output_channel),
                        nn.ConvTranspose2d(cngf, output_channel, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(output_channel),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else:
            output = self.main(input)
        return output
