import os
import sys
from os.path import join as pjoin
from datetime import datetime

import argparse
import torch

from torch_geometric.data import DataLoader

from core.dataloader.dataset import GraphDataset
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem
from core.trainer.tnt_trainer import TNTTrainer

sys.path.append("core/dataloader")


def test(args):
    """
    script to test the tnt model
    "param args:
    :return:
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    # config
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    test_set_dir = pjoin(args.data_root, "{}_intermediate".format(args.split))
    output_dir = pjoin(args.save_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

    # data loading
    try:
        test_set = ArgoverseInMem(test_set_dir)
    except:
        raise Exception("Failed to load the data, please check the dataset!")

    # init trainer
    trainer = TNTTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        horizon=args.horizon,
        top_k=args.top_k,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    trainer.test(miss_threshold=2.0, save_pred=False, plot=True, convert_coordinate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default="../../dataset/v2x-seq-tfd/V2X-Seq-TFD-Example/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data",
                        help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="val")

    parser.add_argument("-b", "--batch_size", type=int, default=480,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=80,
                        help="dataloader worker size")
    parser.add_argument("--horizon", type=int, default=50,
                        help="prediting the number of future points")
    parser.add_argument("--top_k", type=int, default=6,
                        help="prediting the number of trajectories")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=0,
                        help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        default="",
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="",
                        help="resume a model state for fine-tune")

    parser.add_argument("-d", "--save_dir", type=str, default="test_result")
    args = parser.parse_args()
    test(args)
