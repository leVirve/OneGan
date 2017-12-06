import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as F

import onegan as ohgan
from onegan.io.loader import SourceToTargetDataset
from onegan.external import pix2pix

torch.backends.cudnn.benchmark = True


def get_dataloader(args, input_size, source_folder, target_folder):
    transform = F.Compose([
        F.Resize(input_size, interpolation=Image.BICUBIC),
        F.ToTensor(),
        F.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    dataset_params = {'source_folder': source_folder,
                      'target_folder': target_folder,
                      'transform': transform}
    loader_params = {'batch_size': args.batch_size, 'num_workers': args.worker}
    train_loader = SourceToTargetDataset(phase='train', **dataset_params).to_loader(**loader_params)
    val_loader = SourceToTargetDataset(phase='val', **dataset_params).to_loader(**loader_params)
    return train_loader, val_loader


def make_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))


if __name__ == '__main__':
    parser = ohgan.option.Parser(description='Inpainting cGAN', config='./example/config.yml')
    parser.add_argument('--name')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--source_folder')
    parser.add_argument('--target_folder')
    args = parser.parse()
    print(args)

    train_loader, val_loader = get_dataloader(
        args, (args.image_size, args.image_size), args.source_folder, args.target_folder)

    conditional = True
    # g = pix2pix.define_G(3, 3, 64, 'unet_256', norm='instance', init_type='xavier').cuda()
    # d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', norm='instance', init_type='xavier').cuda()
    g = ohgan.models.GeneratorUNet(3, 3, 64, norm='instance').cuda()
    d = ohgan.models.Discriminator(6 if conditional else 3, 3, 64, norm='instance').cuda()
    ohgan.models.init_weights(g, 'xavier')
    ohgan.models.init_weights(d, 'xavier')

    gan_criterion = (ohgan.losses.GANLoss(conditional)
                     .add_term('smooth', nn.L1Loss(), weight=100)
                     .add_term('adv', ohgan.losses.AdversarialLoss(d), weight=1)
                     .add_term('gp', ohgan.losses.GradientPaneltyLoss(d), weight=0.25))
    metric = ohgan.metrics.Psnr()

    estimator = ohgan.estimator.OneWGANEstimator(
        model=(g, d),
        optimizer=(make_optimizer(g, lr=args.lr), make_optimizer(d, lr=args.lr)),
        metric=metric,
        saver=ohgan.utils.GANCheckpoint(save_epochs=5),
        name=args.name
    )
    estimator.run(train_loader, val_loader, epochs=args.epoch)
