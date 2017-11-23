import onegan.external as ext


def test_dcgan():
    g = ext.dcgan.DCGANGenerator(64, 100, 3, 64)
    d = ext.dcgan.DCGANDiscriminator(64, 100, 3, 64)
    assert g, d


def test_dcgan_nobn():
    g = ext.dcgan.DCGANGeneratorNobn(64, 100, 3, 64)
    d = ext.dcgan.DCGANDiscriminatorNobn(64, 100, 3, 64)
    assert g, d
