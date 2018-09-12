import torch
from torchvision import transforms as tf

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
    image = io.pil_open('test/test_data/scene.jpg')
    assert image


def test_image_resize():
    image = io.pil_open('test/test_data/scene.jpg')

    height, width = 100, 200
    small_image = io.resize(image, (height, width))
    assert small_image.size == (width, height)


def test_compose_pipeline():
    data_pipeline = tf.Compose([
        io.transform.LoadPILImage(mode='RGB'),
        io.transform.Resize((320, 240), interpolation='bilinear'),
        tf.RandomCrop((240, 120)),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,)),
    ])

    iamge_path = 'test/test_data/scene.jpg'
    random_image = data_pipeline(iamge_path)

    assert random_image.size() == torch.Size((3, 240, 120))


def test_state_compose_on_pairs():
    image_pipeline = io.transform.StateCompose([
        io.transform.LoadPILImage(mode='RGB'),
        io.transform.Resize((320, 240), interpolation='bilinear'),
        tf.RandomCrop((240, 120)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,)),
    ])
    image_dup_pipeline = io.transform.StateCompose([
        io.transform.LoadPILImage(mode='RGB'),
        io.transform.Resize((320, 240), interpolation='bilinear'),
        tf.RandomCrop((240, 120)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,)),
    ])
    iamge_path = 'test/test_data/scene.jpg'

    old_img1, old_img2 = None, None
    for i in range(10):
        img1 = image_pipeline(iamge_path)
        img2 = image_dup_pipeline(iamge_path, random_state=image_pipeline.random_state)
        assert (img1 - img2).sum() == 0

        if old_img1 is None and old_img2 is None:
            old_img1, old_img2 = img1, img2
            continue
        assert (img1 - old_img1).sum() != 0
        assert (img2 - old_img2).sum() != 0


def test_composer_pipeline_on_pairs():
    image_pipeline = io.transform.StateCompose([
        io.transform.LoadPILImage(mode='RGB'),
        io.transform.Resize((320, 240), interpolation='bilinear'),
        tf.RandomCrop((240, 120)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,)),
    ])
    label_pipeline = io.transform.StateCompose([
        io.transform.LoadPILImage(mode='L'),
        io.transform.Resize((320, 240), interpolation='nearest'),
        tf.RandomCrop((240, 120)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,)),
    ])

    iamge_path = 'test/test_data/scene.jpg'

    # test for pairs data
    processed_image = image_pipeline(iamge_path)
    processed_label = label_pipeline(iamge_path, random_state=image_pipeline.random_state)

    assert processed_image.size() == torch.Size((3, 240, 120))
    assert processed_label.size() == torch.Size((1, 240, 120))
