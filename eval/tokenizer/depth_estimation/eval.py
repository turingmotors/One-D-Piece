"""Evaluation script for One-D-Piece.

Code Copyright (2024) Turing Inc. and/or its affiliates

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
    https://github.com/bytedance/1d-tokenizer/blob/main/scripts/train_titok.py
"""
import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.path.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.path.pardir))
sys.path.append(parent_dir)

import torch
from evaluator import DepthEstimationQualityEvaluator

from PIL import Image
from tqdm import tqdm
import numpy as np

from accelerate.utils import set_seed

@torch.no_grad()
def process_buffer(buffer, device, evaluators):
    original_images = torch.cat([x[0] for x in buffer], dim=0).permute(0, 3, 1, 2).to(torch.float) / 255.0
    eval_images = torch.cat([x[1] for x in buffer], dim=0).permute(0, 3, 1, 2).to(torch.float) / 255.0

    for evaluator in evaluators:
        evaluator.update(original_images, eval_images)

    return evaluators

@torch.no_grad()
def eval_reconstruction(
    original_images: str,
    target_images: str,
    device,
    evaluators,
):
    for evaluator in evaluators:
        evaluator.reset_metrics()

    assert original_images.shape == target_images.shape, f"Original images shape {original_images.shape} != Eval images shape {target_images.shape}"

    buffer = []
    buffer_capacity = 256

    for original_image, eval_image in tqdm(zip(original_images, target_images)):
        original_image = torch.tensor(original_image).unsqueeze(0)
        eval_image = torch.tensor(eval_image).unsqueeze(0)
        original_images = original_image.to(
            device, memory_format=torch.contiguous_format, non_blocking=True
        )
        eval_images = eval_image.to(device, memory_format=torch.contiguous_format, non_blocking=True)
        buffer.append((original_images, eval_images))

        if len(buffer) == buffer_capacity:
            process_buffer(buffer, device, evaluators)
            buffer = []
    if buffer:
        process_buffer(buffer, device, evaluators)
        buffer = []

    return tuple(evaluator.result() for evaluator in evaluators)


def main(args):
    # Enable TF32 on Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    set_seed(42)

    if args.target_npy is not None:
        target_images = np.load(args.target_npy)
    elif args.target_format is not None:
        target_images = []
        # %05d is the frame number
        for i in tqdm(range(50000)):
            path = args.target_format.format(i)
            # open as uint8 
            image = Image.open(path)
            image = np.array(image)
            # assert uint8
            assert image.dtype == np.uint8
            target_images.append(image)

        target_images = np.array(target_images)


    if args.original_npy is not None:
        original_images = np.load(args.original_npy)
    elif args.original_format is not None:
        original_images = []
        # %05d is the frame number
        for i in tqdm(range(50000)):
            path = args.original_format.format(i)
            # open as uint8 
            image = Image.open(path)
            image = np.array(image)
            # assert uint8
            assert image.dtype == np.uint8
            original_images.append(image)

        original_images = np.array(original_images)

    # Start training.
    eval_scores = eval_reconstruction(
        original_images=original_images,
        target_images=target_images,
        device="cuda",
        evaluators=[
            DepthEstimationQualityEvaluator(
                device="cuda",
            ),
        ]
    )
    print("Eval scores:", eval_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_npy",
        type=str,
        default=None,
        help="Original images with npy format.",
    )
    parser.add_argument(
        "--original_format",
        type=str,
        default=None,
        help="Original images glob path with format string. There must be 0-49999 images.",
    )
    parser.add_argument(
        "--target_npy",
        type=str,
        default=None,
        help="Eval images with npy format.",
    )
    parser.add_argument(
        "--target_format",
        type=str,
        default=None,
        help="Original images glob path. There must be 0-49999 images.",
    )
    args = parser.parse_args()
    if not args.original_npy:
        assert args.original_format, "Original format is required."
    elif not args.original_format:
        assert args.original_npy, "Original npy is required."

    if not args.target_npy:
        assert args.target_format, "Target format is required."
    elif not args.target_format:
        assert args.target_npy, "Target npy is required."

    main(args)
