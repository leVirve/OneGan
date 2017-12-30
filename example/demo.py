import torch
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
    g = pix2pix.define_G(3, 3, 64, 'unet_256', norm='instance', init_type='xavier').cuda()
    d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', norm='instance', init_type='xavier').cuda()

    estimator = ohgan.estimator.OneGANEstimator(
        model=(g, d),
        optimizer=(make_optimizer(g, lr=args.lr), make_optimizer(d, lr=args.lr)),
        metric=ohgan.metrics.psnr,
        save_epochs=5,
        name=args.name
    )
    estimator.conditional = conditional
    estimator.run(train_loader, val_loader, epochs=args.epoch)
