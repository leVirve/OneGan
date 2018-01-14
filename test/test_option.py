import onegan.option as option


def test_paser():
    empty_parser = option.Parser(description='cGAN')
    dummy_args = []
    args = empty_parser.parse(dummy_args)
    for k, v in vars(args).items():
        assert not v

    yml_parser = option.Parser(description='cGAN', config='./example/config.yml')
    yml_cfg = yml_parser._load_option_config('./example/config.yml')
    args = yml_parser.parse(dummy_args)

    for a, b in zip(sorted(vars(args).items()), sorted(yml_cfg.items())):
        ka, va = a
        kb, vb = b
        assert ka == kb and va == vb
