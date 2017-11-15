import torch
import torch.nn as nn

import onegan as ohgan

import nets
from data import train_loader, val_loader


if __name__ == '__main__':
    conditional = True

    def make_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    g = nets.define_G(3, 3, 64, 'unet_128', init_type='xavier').cuda()
    d = nets.define_D(6 if conditional else 3, 64, 'basic', init_type='xavier').cuda()

    gan_criterion = (ohgan.losses.GANLoss(conditional)
                     .add_term('smooth', nn.L1Loss(), weight=100)
                     .add_term('adv', ohgan.losses.AdversarialLoss(d), weight=1)
                     .add_term('gp', ohgan.losses.GradientPaneltyLoss(d), weight=0.25))
    metric = ohgan.metrics.PsnrMixin()

    estimator = ohgan.estimator.OneGANEstimator(
        model=(g, d),
        optimizer=(make_optimizer(g), make_optimizer(d)),
        criterion=gan_criterion,
        metric=metric,
        saver=ohgan.utils.GANCheckpoint(save_epochs=5),
        name='flowers'
    )
    estimator.run(train_loader, val_loader, epochs=50)
