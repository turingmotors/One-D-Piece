import random
import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from PIL import Image
from modeling.one_d_piece import OneDPiece
from modeling.titok import TiTok
from modeling.llamagen_diffustyle import GPT_models
from tqdm import tqdm
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
import numpy as np

def greedy_decode(model, classes):
    model.eval()
    model.requires_grad_(False)

    device = "cuda"
    print(f"Working on device: {device}")
    model = model.to(device)

    cond = torch.tensor(classes, device=device).view(-1)
    input_pos = torch.zeros(1, device=device, dtype=torch.long)
    print(cond.shape, input_pos.shape)

    device = cond.device
    max_batch_size = cond.shape[0]
    cfg_scale = 1.0
    max_seq_length = 128
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        # FIXME: specifying bf16 here is a hack to avoid dtype mismatch in the model
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=torch.bfloat16)

    # autocast
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        logits = model(None, cond, input_pos)
        # argmax
        next_token = logits.argmax(dim=-1)
    
        tokens = next_token
        for i in range(127):
            input_pos = torch.cat([input_pos, torch.tensor([i+1], device=device, dtype=torch.long)], dim=0)
            logits = model(tokens, cond_idx=cond, input_pos=input_pos)
            next_token = logits.argmax(dim=-1)[:,-1:]
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens



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

    if config.model.type == "onedpiece":
        Model = OneDPiece
    elif config.model.type == "titok":
        Model = TiTok
    elif config.model.type == "progressive_titok":
        Model = ProgressiveTiTok
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    tokenizer_weight = torch.load(tokenizer_checkpoint, map_location="cpu")
    tokenizer = Model(config)
    tokenizer.load_state_dict(tokenizer_weight)

    tokenizer.eval()
    tokenizer.requires_grad_(False)

    
    # モデルの読み込みは書き換える
    model = GPT_models[config.model.generator.model_type](config)
    
    # llamagen_weight = torch.load(args.checkpoint, map_location="cpu")
    # model.load_state_dict(llamagen_weight)
    model.load_pretrained_ar(args.checkpoint)

    model.eval()
    model.requires_grad_(False)

    device = "cuda"
    print(f"Working on device: {device}")
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    random_classes = labels = [7]

    # tokens = greedy_decode(model, random_classes)
    # generated_image = tokenizer.decode_tokens(tokens)
    # print(generated_image.shape) # B, 3, H, W
    # # save it
    # generated_image = torch.clamp(generated_image, 0.0, 1.0)
    # generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    # for i, img in enumerate(generated_image):
    #     path = Path(args.output_dir) / f"{i:04d}.png"
    #     Image.fromarray(img).save(path.as_posix())
    
    # # create 10x10 grid of images
    # grid_img = np.concatenate([np.concatenate(generated_image[i:i+10], axis=1) for i in range(0, 100, 10)], axis=0)
    # Image.fromarray(grid_img).save(Path(args.output_dir) / "grid.png")
    # exit()


    # mkdir
    Path(args.output_dir).mkdir(exist_ok=True)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = sample_fn(
            model,
            tokenizer,
            labels=random_classes,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
            return_tensor=False
        )

        # save the generated images
        for i, img in enumerate(result):
            path = Path(args.output_dir) / f"{i:04d}.png"
            Image.fromarray(img).save(path.as_posix())
        # create 10x10 grid of images
        # grid_img = np.concatenate([np.concatenate(result[i:i+10], axis=1) for i in range(0, 100, 10)], axis=0)
        # Image.fromarray(grid_img).save(Path(args.output_dir) / "grid.png")


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              temperature=1.0,
              top_k=0,
              top_p=1.0,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        steps=8,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp-miwa/generator/exp002-progressive_titok_s256_llamagen.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="generated/llamagen_diff")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()
    main(args)