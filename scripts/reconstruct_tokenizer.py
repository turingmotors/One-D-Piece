"""Generate images using One-D-Piece.

# For Tokenizer Models
# Use `--length` to specify the number of tokens to generate.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --config configs/one-d-piece_s256.yaml \
        --length=128 \
        --output_dir generated/one_d_piece_s256_len128

# For TiTok Models
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --config configs/titok_b64.yaml \
        --output_dir generated/titok_b64

# For Image Formats
# Use `--png`, `--jp2`, `--jpg`, `--webp` to specify the image format.
# Also use `--save_raw` to save the raw images instead of numpy arrays.
# Use `--original` to save the original images.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --output_dir generated/jpeg_comp2 \
        --save_raw \
        --original \
        --jpg \
        --jpg_compression_rate=2 \

# For Saving Original Images
# Use `--original` to save the original images.
# Use `--labels` to save the labels of the images.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --output_dir generated/original \
        --original \
        --labels
"""

import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from omegaconf import OmegaConf

import numpy as np
import torch
from types import SimpleNamespace

from utils.logger import setup_logger
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator

from utils.train_utils import create_model, create_dataloader, auto_resume

def image_generator(config, logger, accelerator):
    config.training.per_gpu_batch_size = 1
    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    count = 0
    for batch in eval_dataloader:
        for img_tensor, key, class_id in zip(batch['image'], batch['__key__'], batch['class_id']):
            img_tensor = img_tensor.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True, dtype=torch.float)
            count += 1
            assert tuple(img_tensor.shape) == (3, 256, 256), img_tensor.shape
            # yield img_tensor.unsqueeze(0).to(device), Path(key + ".png")
            yield img_tensor.unsqueeze(0), Path(f"image_{count-1:05d}.png"), key, class_id.unsqueeze(0).to(accelerator.device)

class Namespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

@torch.no_grad()
def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if not args.original:
        config = OmegaConf.load(args.config)
        config.model.vq_model.finetune_decoder = True
        config.model.vq_model.strict_length_assertion = False
        # Avoiding unexpectedly loading pretrained tokenizer weights.
        config.model.vq_model.pretrained_tokenizer_weight = None

        config = OmegaConf.load(args.config)
        if args.length is None:
            args.length = config.model.vq_model.num_latent_tokens
    else:
        config = Namespace(
            dataset=Namespace(
                params=Namespace(
                    train_shards_path_or_url="/data/dataset/imagenet/imagenet-val-{000000..000049}.tar",
                    eval_shards_path_or_url="/data/dataset/imagenet/imagenet-val-{000000..000049}.tar",
                    num_workers_per_gpu=12
                ),
                preprocessing=Namespace(
                    resize_shorter_edge=256,
                    crop_size=256,
                    random_crop=False,
                    random_flip=False
                ),
            ),
            experiment=Namespace(
                max_train_examples=1_281_167,
                output_dir=args.output_dir + "_log",
            ),
            training=Namespace(
                enable_tf32=True,
                mixed_precision="bf16",
                gradient_accumulation_steps=1,
            )
        )

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = config.experiment.output_dir + "_generation"
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="Generation", log_level="INFO", output_file=f"{output_dir}/log{accelerator.process_index}.txt")
    if not args.original:
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers(config.experiment.name)
            config_path = Path(output_dir) / "config.yaml"
            logger.info(f"Saving config to {config_path}")
            OmegaConf.save(config, config_path)
            logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

        # If passed along, set the training seed now.
        if config.training.seed is not None:
            set_seed(config.training.seed, device_specific=True)

        if config.model.type in ["titok"]:
            config.model.vq_model.strict_length_assertion = False

        model, ema_model = create_model(
            config, logger, accelerator, model_type=config.model.type)
        if args.checkpoint is not None:
            logger.info(f"Loading model from checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

        # Prepare everything with accelerator.
        logger.info("Preparing model, optimizer and dataloaders")
        model = accelerator.prepare(model)
        if config.training.use_ema:
            ema_model.to(accelerator.device)

        # Start training.
        logger.info(f""" Start evaluation of the model. """)
        _, _ = auto_resume(config, logger, accelerator, ema_model, -1, strict=True)

        accelerator.print(f"Evaluation of the checkpoint started.")
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        model.eval()

    # mkdir
    Path(args.output_dir).mkdir(exist_ok=True)

    count = 0
    generator = image_generator(config, logger, accelerator)
    numpy_images = {}
    numpy_tokens = {}
    labels = {}
    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1
        if not args.original:
            if config.model.type == "titok":
                reconstructed_image, result_dict = accelerator.unwrap_model(model)(image)
                reconstructed_image = reconstructed_image[0]
            elif args.length != -1:
                length = args.length
                reconstructed_image, result_dict = accelerator.unwrap_model(model)(image, length=length)
                reconstructed_image = reconstructed_image[0]
            numpy_tokens[image_key] = result_dict["min_encoding_indices"][0][0].cpu().numpy()
        else:
            reconstructed_image = image[0]

        reconstructed_image = torch.round(reconstructed_image.clamp(0, 1) * 255.0).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
        numpy_images[image_key] = reconstructed_image
        if args.labels:
            labels[image_key] = class_id.item()
    
    # save numpy images (fix order)
    numpy_images = [numpy_images[key] for key in sorted(numpy_images.keys())]
    numpy_images = np.stack(numpy_images)
    if not args.dry_run:
        if args.labels:
            labels = [labels[key] for key in sorted(labels.keys())]
            labels = np.array(labels)
            np.save(os.path.join(args.output_dir, "labels.npy"), labels)
        if args.tokens:
            np_keys = [key for key in sorted(numpy_tokens.keys())]
            np_keys = np.array(np_keys)
            np_tokens = [numpy_tokens[key] for key in sorted(numpy_tokens.keys())]
            np_tokens = np.stack(np_tokens)
            np.savez_compressed(os.path.join(args.output_dir, "tokens.npz"), tokens=np_tokens, keys=np_keys)
        if args.save_raw:
            import cv2
            for i, image in enumerate(numpy_images):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if args.png:
                    cv2.imwrite(os.path.join(args.output_dir, f"image_{i:05d}.png"), image)
                if args.jp2:
                    cv2.imwrite(os.path.join(args.output_dir, f"image_{i:05d}.jp2"), image, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), args.jp2_compression_rate])
                if args.jpg:
                    cv2.imwrite(os.path.join(args.output_dir, f"image_{i:05d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_compression_rate])
                if args.webp:
                    cv2.imwrite(os.path.join(args.output_dir, f"image_{i:05d}.webp"), image, [int(cv2.IMWRITE_WEBP_QUALITY), args.webp_compression_rate])
        elif not args.without_image:
            np.save(os.path.join(args.output_dir, "images.npy"), numpy_images)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint if you want to use a specific checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--length", type=int, default=-1, help="Number of tokens to generate")
    # For saving images in different formats
    parser.add_argument("--save_raw", action="store_true", help="Save raw images instead of numpy arrays")
    parser.add_argument("--png", action="store_true", help="Save images in PNG format")
    parser.add_argument("--jp2", action="store_true", help="Save images in JPEG2000 format")
    parser.add_argument("--jp2_compression_rate", type=int, default=2, help="0-100")
    parser.add_argument("--jpg", action="store_true", help="Save images in JPEG format")
    parser.add_argument("--jpg_compression_rate", type=int, default=2, help="0-100")
    parser.add_argument("--webp", action="store_true", help="Save images in WebP format")
    parser.add_argument("--webp_compression_rate", type=int, default=2, help="0-100")
    # For saving original images
    parser.add_argument("--original", action="store_true", help="Save original images, without any processing")
    parser.add_argument("--labels", action="store_true", help="Save labels")
    # For saving tokens
    parser.add_argument("--tokens", action="store_true", help="Save image tokens")
    # For faster IO
    parser.add_argument("--without_image", action="store_true", help="Do not save images. This will make the io faster if you only need tokens or labels.")
    # For debugging
    parser.add_argument("--dry_run", action="store_true", help="Do not save images")


    args = parser.parse_args()
    if args.dry_run:
        print("Dry run mode is enabled. No images will be saved.")
    if (args.save_raw or args.png or args.jp2 or args.jpg or args.webp) and args.without_image:
        raise Warning("You cannot save raw images and not save images at the same time. For this run, `--without_image` is automatically set to False.")
        args.without_image = False
    if (args.png or args.jp2 or args.jpg or args.webp) and not args.save_raw:
        raise Warning("You need to specify --save_raw to save images in different formats. For this run, `--save_raw` is automatically set to True.")
        args.save_raw = True
    if args.original and args.tokens:
        raise Warning("You cannot save both original images and tokens at the same time. For this run, `--tokens` is automatically set to False.")
        args.tokens = False

    main(args)
