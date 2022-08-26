import os
import pathlib
import torch
import cs3600.download
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
torch.manual_seed(42)

class MyDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle=True):
        DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')

        _, dataset_dir = cs3600.download.download_data(out_path=DATA_DIR, url=self.data_url, extract=True, force=False)

        im_size = 64
        tf = T.Compose([
            # Resize to constant spatial dimensions
            T.Resize((im_size, im_size)),
            # PIL.Image -> torch.Tensor
            T.ToTensor(),
            # Dynamic range [0,1] -> [-1, 1]
            T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        ])

        self.ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)

        super().__init__(self.ds_gwb, batch_size, shuffle)

    @property
    def im_size(self):
        return self.ds_gwb[0][0].shape
    
    @property
    def data_url(self):
        return 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    

class BushDataLoader(MyDataLoader):    
    @property
    def data_url(self):
        return 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip' 