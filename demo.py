import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as F

import onegan as ohgan
from onegan.io.loader import SourceToTargetDataset
from onegan.external import pix2pix

torch.backends.cudnn.benchmark = True


def get_dataloader():
    transform = F.Compose([
        F.Resize((128, 128), interpolation=Image.BICUBIC),
        F.ToTensor(),
        F.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    dataset_params = {'source_folder': '../human-recovery/datasets/bmvc/bmvc_flowers/train_128/inputs/scribble',
                      'target_folder': '../human-recovery/datasets/bmvc/bmvc_flowers/train_128/gt/',
                      'transform': transform}
    loader_params = {'batch_size': 32, 'num_workers': 4}
    train_loader = SourceToTargetDataset(phase='train', **dataset_params).to_loader(**loader_params)
    val_loader = SourceToTargetDataset(phase='val', **dataset_params).to_loader(**loader_params)
    return train_loader, val_loader


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))


if __name__ == '__main__':
    train_loader, val_loader = get_dataloader()

    conditional = True
    g = pix2pix.define_G(3, 3, 64, 'unet_128', init_type='xavier').cuda()
    d = pix2pix.define_D(6 if conditional else 3, 64, 'basic', init_type='xavier').cuda()

    gan_criterion = (ohgan.losses.GANLoss(conditional)
                     .add_term('smooth', nn.L1Loss(), weight=100)
                     .add_term('adv', ohgan.losses.AdversarialLoss(d), weight=1)
                     .add_term('gp', ohgan.losses.GradientPaneltyLoss(d), weight=0.25))
    metric = ohgan.metrics.Psnr()

    estimator = ohgan.estimator.OneGANEstimator(
        model=(g, d),
        optimizer=(make_optimizer(g), make_optimizer(d)),
        criterion=gan_criterion,
        metric=metric,
        saver=ohgan.utils.GANCheckpoint(save_epochs=5),
        name='flowers'
    )
    estimator.run(train_loader, val_loader, epochs=50)
