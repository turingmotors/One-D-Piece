import random
import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from PIL import Image
from modeling.titok import TiTok
from modeling.one_d_piece import OneDPiece
from modeling.maskgit import ImageBert
from omegaconf import OmegaConf
import numpy as np
from demo_util import sample_fn


def main(args):
    print("Loading model model...")
    config = OmegaConf.load(args.config)
    tokenizer_checkpoint = config.experiment.get("tokenizer_checkpoint", None)
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

    if config.model.type == "titok":
        Tokenizer = TiTok
    elif config.model.type == "one_d_piece":
        Tokenizer = OneDPiece
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    tokenizer_weight = torch.load(tokenizer_checkpoint, map_location="cpu")
    tokenizer = Tokenizer(config)
    tokenizer.load_state_dict(tokenizer_weight)

    tokenizer.eval()
    tokenizer.requires_grad_(False)

    generator_weight = torch.load(args.checkpoint, map_location="cpu")
    generator = ImageBert(config)
    generator.load_state_dict(generator_weight)

    generator.eval()
    generator.requires_grad_(False)

    device = "cuda"
    print(f"Working on device: {device}")
    tokenizer = tokenizer.to(device)
    generator = generator.to(device)
    random_classes = random.choices(range(1000), k=100)

    # mkdir
    Path(args.output_dir).mkdir(exist_ok=True)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = sample_fn(
            generator,
            tokenizer,
            labels=random_classes,
            guidance_scale=args.guidance_scale,
            guidance_decay=args.guidance_decay,
            guidance_scale_pow=args.guidance_scale_pow,
            randomize_temperature=args.randomize_temperature,
            softmax_temperature_annealing=args.softmax_temperature_annealing,
            num_sample_steps=args.num_sample_steps,
            device=device,
            return_tensor=False
        )
        # save the generated images
        for i, img in enumerate(result):
            path = Path(args.output_dir) / f"{i:04d}.png"
            Image.fromarray(img).save(path.as_posix())
        # create 10x10 grid of images
        grid_img = np.concatenate([np.concatenate(result[i:i+10], axis=1) for i in range(0, 100, 10)], axis=0)
        Image.fromarray(grid_img).save(Path(args.output_dir) / "grid.png")
        print(f"Images saved to {args.output_dir}/grid.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generator/maskgit_one-d-piece_s256.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="generated/maskgit")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--guidance_decay", type=str, default="constant")
    parser.add_argument("--guidance_scale_pow", type=float, default=3.0)
    parser.add_argument("--randomize_temperature", type=float, default=2.0)
    parser.add_argument("--softmax_temperature_annealing", action="store_true")
    parser.add_argument("--num_sample_steps", type=int, default=8)
    args = parser.parse_args()
    main(args)
