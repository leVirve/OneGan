import torch

from onegan import io


class DummyDataset(io.BaseDataset):

    def __init__(self, phase, **kwargs):
        super().__init__(phase)

    def __len__(self):
        return len(self.filenames)


def test_base_loader():
    dataset_params = {'root': None}
    loader_params = dict(batch_size=4, num_workers=1, pin_memory=True)

    dataset = DummyDataset(phase='train', **dataset_params)
    dataloader = dataset.to_loader(**loader_params)

    assert isinstance(dataset, torch.utils.data.Dataset)
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_load_image():
    image = io.load_image('test/test_data/scene.jpg')
    assert image


def test_image_resize():
    image = io.load_image('test/test_data/scene.jpg')

    height, width = 100, 200
    small_image = io.image_resize(image, (height, width))
    assert small_image.size == (width, height)


def test_transform_pipeline():
    tf = io.TransformPipeline()
    assert tf
