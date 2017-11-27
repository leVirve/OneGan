import onegan.models as models


def test_pix2pix():
    g = models.GeneratorUNet(3, 3, 64)
    d = models.Discriminator(6, 3, 64)
    assert g, d


def test_weight_init():
    g = models.GeneratorUNet(3, 3, 64)
    models.init_weights(g, 'xavier')
    assert g
