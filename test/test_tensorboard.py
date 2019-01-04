import onegan.extension.tensorboard as tensorboard


def test_new_writer():
    logger = tensorboard.TensorBoardLogger()
    assert logger
