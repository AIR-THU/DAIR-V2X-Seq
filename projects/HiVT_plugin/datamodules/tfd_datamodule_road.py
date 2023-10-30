#---------------------------------------------------------------------------------#
# V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting (https://arxiv.org/abs/2305.05938)  #
# Source code: https://github.com/AIR-THU/DAIR-V2X-Seq                              #
# Copyright (c) DAIR-V2X. All rights reserved.                                #
#---------------------------------------------------------------------------------#
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets import TFDInfraDataset


class TFDInfraDataModule(LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(TFDInfraDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        TFDInfraDataset(self.root, 'train', self.train_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TFDInfraDataset(self.root, 'train', self.train_transform, self.local_radius)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
