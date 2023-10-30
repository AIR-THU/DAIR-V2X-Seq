from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import TFDDataModule
from models.hivt import HiVT

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='../../dataset/v2x-seq-tfd/V2X-Seq-TFD-Example/cooperative-vehicle-infrastructure/fusion_for_prediction/')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    datamodule = TFDDataModule.from_argparse_args(args)
    datamodule.setup()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, accelerator='gpu', callbacks=[model_checkpoint])
    model = HiVT(**vars(args))
    if args.ckpt_path:
        model = model.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    model.T_max = args.T_max
    model.max_epochs = args.max_epochs
    trainer.fit(model, datamodule)
