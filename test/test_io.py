import onegan
from onegan.io.loader import load_image


def dummy_collect_images():
    return [None] * 100


class DummyDataset(onegan.io.BaseDataset):

    def __init__(self, phase, debug=False, **kwargs):
        super().__init__(phase)
        self.phase = phase
        self.filenames = self._split_data(dummy_collect_images(), phase, debug=debug)

    def __getitem__(self, index):
        return load_image(self.filenames[index]).convert('RGB')

    def __len__(self):
        return len(self.filenames)


def test_base_loader():
    dataset_params = {'root': None}
    loader_params = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DummyDataset(phase='train', **dataset_params).to_loader(**loader_params)
    val_loader = DummyDataset(phase='val', **dataset_params).to_loader(**loader_params)
    assert len(train_loader) != len(val_loader)


def test_source_to_target_loader():
    dataset_params = {'source_folder': '.', 'target_folder': '.'}
    loader_params = dict(batch_size=32, num_workers=4, pin_memory=True)
    onegan.io.SourceToTargetDataset(phase='train', **dataset_params).to_loader(**loader_params)
    onegan.io.SourceToTargetDataset(phase='val', **dataset_params).to_loader(**loader_params)
