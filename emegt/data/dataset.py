import pickle

from torch_geometric.data import Data, Dataset
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

class BatchDatapoint:
    def __init__(self,
                 block_file,
                 n_samples,
                 ):
        self.block_file = block_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        
        self.datapoints = []

        with open(self.block_file, 'rb') as f:
            dp = pickle.load(f)
            self.datapoints = dp

        assert len(self.datapoints) == self.n_samples

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchDatapointProperty:
    def __init__(self,
                 block,
                 ):
        self.block = block
        # deal with the last batch graph numbers.
        self.n_samples = len(block)
        self.datapoints = None

    def load_datapoints(self):
        self.datapoints = []
        self.datapoints = self.block

        assert len(self.datapoints) == self.n_samples

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class GEOMDataset(Dataset):

    def __init__(self, data, graph_per_file=None, transforms=None):

        self.data = data

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
        self.transforms = transforms

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx):
        # print(idx)
        dp_idx = idx // self.sample_per_file
        real_idx = idx % self.sample_per_file
        tar = self.data[dp_idx][real_idx].clone()
        if self.transforms:
            tar = self.transforms(tar)
        return tar


    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res
