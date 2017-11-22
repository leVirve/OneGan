import onegan
from onegan.data.loader import load_image


def dummy_collect_images(root):
    return [None] * 100


class DummyDastaset(onegan.data.BaseDastaset):

    def __init__(self, root, target_size, **kwargs):
        self.root = root
        self.target_size = target_size
        self.filenames = dummy_collect_images(root)
        self.debug = kwargs.get('debug')

    def initialize(self, phase):
        self.filenames = self._initialize(self.filenames, phase)
        return self

    def __getitem__(self, index):
        return load_image(self.filenames[index]).convert('RGB')

    def __len__(self):
        return len(self.filenames)


dataset_params = {'root': None, 'target_size': (128, 128)}
loader_params = dict(batch_size=32, num_workers=4, pin_memory=True)
train_loader = DummyDastaset(**dataset_params).to_loader(phase='train', **loader_params)
val_loader = DummyDastaset(**dataset_params).to_loader(phase='val', **loader_params)


def test_loader():
    assert len(train_loader) != len(val_loader)
