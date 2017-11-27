import onegan.option as option


def test_paser():
    empty_parser = option.Parser(description='cGAN')
    assert empty_parser.parse() == {}

    args = option.Parser(description='cGAN', config='./example/config.yml').parse()
    assert args
