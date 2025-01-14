"""Sampling scripts for generator models for One-D-Piece and TiTok on ImageNet.

Original code Copyright (2024) Bytedance Ltd. and/or its affiliates
Modified code Copyright (2024) Turing Inc. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
    https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.path.pardir))
sys.path.append(parent_dir)

import demo_util
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch
from omegaconf import OmegaConf

from modeling.one_d_piece import OneDPiece
from modeling.titok import TiTok
from modeling.maskgit import ImageBert

import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    config = OmegaConf.load(args.config)
    tokenizer_checkpoint = args.tokenizer_checkpoint if args.tokenizer_checkpoint else config.experiment.get("tokenizer_checkpoint", None)
    if tokenizer_checkpoint is None:
        raise ValueError("tokenizer_checkpoint is not found in the config file.")

    if args.checkpoint is None:
        # search for the latest checkpoint
        # format: Path(config.experiment.output_dir) / "checkpoint-%d/ema_model/pytorch_model.bin"
        checkpoints = list(Path(config.experiment.output_dir).glob("checkpoint-*"))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint = [x / "ema_model" / "pytorch_model.bin" for x in checkpoints][-1]
        print(f"Using the latest checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    elif args.checkpoint.isdigit():
        # search for the checkpoint with the given number
        checkpoint = Path(config.experiment.output_dir) / f"checkpoint-{args.checkpoint}" / "ema_model" / "pytorch_model.bin"
        print(f"Using the checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()

    if config.model.type == "one_d_piece":
        Model = OneDPiece
    elif config.model.type == "titok":
        Model = TiTok
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    tokenizer_weight = torch.load(tokenizer_checkpoint, map_location="cpu")
    tokenizer = Model(config)
    tokenizer.load_state_dict(tokenizer_weight)

    tokenizer.eval()
    tokenizer.requires_grad_(False)

    weight = torch.load(args.checkpoint, map_location="cpu")
    model = ImageBert(config)
    model.load_state_dict(weight)

    model.eval()
    model.requires_grad_(False)

    device = "cuda"
    print(f"Working on device: {device}")
    tokenizer = tokenizer.to(device)
    model = model.to(device)

    num_fid_samples = 50000
    batch_size = 125
    sample_folder_dir = args.sample_dir
    seed = 42

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)
    # Set up single GPU
    torch.manual_seed(seed)
    print(f"Using device: {device}, seed={seed}")

    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Number of iterations to run for generating the required samples
    iterations = num_fid_samples // batch_size
    pbar = tqdm(range(iterations), desc="Generating samples")
    total = 0

    all_classes = list(range(config.model.generator.condition_num_classes)) * (num_fid_samples // config.model.generator.condition_num_classes)
    cur_idx = 0

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * batch_size: (cur_idx + 1) * batch_size]).to(device)
        cur_idx += 1

        sampling_args = dict(
            randomize_temperature=args.randomize_temperature if args.randomize_temperature is not None else config.model.generator.randomize_temperature,
            softmax_temperature_annealing=args.softmax_temperature_annealing if args.softmax_temperature_annealing is not None else config.model.generator.softmax_temperature_annealing,
            num_sample_steps=args.num_sample_steps if args.num_sample_steps is not None else config.model.generator.num_sample_steps,
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else config.model.generator.guidance_scale,
            guidance_decay=args.guidance_decay if args.guidance_decay is not None else config.model.generator.guidance_decay,
        )
        # autocast to bfloat16
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            samples = demo_util.sample_fn(
                generator=model,
                tokenizer=tokenizer,
                labels=y.long(),
                device=device,
                **sampling_args,
            )
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += batch_size

    # Convert saved samples to a .npz file
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample ImageNet images using TiTok.")
    # config file of llamagen
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--sample_dir", type=str, default="generated/imagenet_samples")
    parser.add_argument("--checkpoint", type=str, default=None)
    # overrides
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None)
    parser.add_argument("--randomize_temperature", type=bool, default=None)
    parser.add_argument("--softmax_temperature_annealing", type=bool, default=None)
    parser.add_argument("--num_sample_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--guidance_decay", type=str, default=None)
    args = parser.parse_args()
    main(args)
