import os
import argparse

import torch
import torchvision

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(".")
from models.pipeline.texture_pipeline import TexturePipeline

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/optimize_texture.yaml")
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_step", type=int, default=0)
    parser.add_argument("--texture_size", type=int, default=4096)

    # only with template
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--scene_id", type=str, default="", help="<house_id>/<room_id>")

    args = parser.parse_args()

    if args.stamp is None:
        setattr(args, "stamp", "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "debug"))

    return args

def init_config(args):
    config = OmegaConf.load(args.config)

    # template
    if len(args.log_dir) != 0 and len(args.prompt) != 0 and len(args.scene_id) != 0:
        print("=> filling template with following arguments:")
        print("   log_dir:", args.log_dir)
        print("   prompt:", args.prompt)
        print("   scene_id:", args.scene_id)

        config.log_dir = args.log_dir
        config.prompt = args.prompt
        config.scene_id = args.scene_id

    return config


def init_pipeline(
        config,
        stamp,
        device=DEVICE,
        inference_mode=False
    ):
    pipeline = TexturePipeline(
        config=config,
        stamp=stamp,
        device=device
    ).to(device)

    pipeline.configure(inference_mode=inference_mode)

    return pipeline

if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["WANDB_START_METHOD"] = "thread"

    torch.backends.cudnn.benchmark = True

    args = init_args()

    inference_mode = len(args.checkpoint_dir) > 0

    print("=> loading config file...")
    config = init_config(args)

    print("=> initializing pipeline...")
    pipeline = init_pipeline(config=config, stamp=args.stamp, inference_mode=inference_mode)

    if not inference_mode:
        print("=> start training...")
        with torch.autograd.set_detect_anomaly(True):
            pipeline.fit()
    else:
        print("=> loading checkpoint...")
        pipeline.load_checkpoint(args.checkpoint_dir, args.checkpoint_step)
        pipeline.inference(args.checkpoint_dir, args.checkpoint_step, args.texture_size)