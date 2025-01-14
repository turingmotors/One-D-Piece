import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from omegaconf import OmegaConf

import numpy as np
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

@torch.no_grad()
def main(args):
    config = OmegaConf.load(args.config)
    config.model.vq_model.finetune_decoder = True
    config.model.vq_model.strict_length_assertion = False
    # Avoiding unexpectedly loading pretrained tokenizer weights.
    config.model.vq_model.pretrained_tokenizer_weight = None

    config = OmegaConf.load(args.config)

    length = config.model.vq_model.num_latent_tokens

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir + "_eval"
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="Tokenizer", log_level="INFO", output_file=f"{output_dir}/log{accelerator.process_index}.txt")

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

    length_to_diff_l1 = [list() for _ in range(length)]
    length_to_diff_l2 = [list() for _ in range(length)]
    buffer = []
    max_count = 50000
    buffer_capacity = 100

    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1

        buffer.append((class_id, image, image))
        if len(buffer) < buffer_capacity:
            continue
        
        images = torch.cat([x[1] for x in buffer], dim=0).to(accelerator.device)

        unwrapped_model = accelerator.unwrap_model(model)
        z_quantized, result_dict = unwrapped_model.encode(images)
        original_decoded = unwrapped_model.decode_tokens(result_dict["min_encoding_indices"])

        for i in range(length):
            if i % 16 == 0:
                print(f"Processing {i}/{length}")
            tokens = result_dict["min_encoding_indices"].to(images.device).clone()
            # randomly replace tokens (index)
            assert tokens.shape == (images.shape[0], 1, length), f"tokens.shape: {tokens.shape}"
            tokens[:, :, i] = torch.randint(0, 4096, (tokens.shape[0], 1), device=tokens.device)
            reconstructed_images = unwrapped_model.decode_tokens(tokens)
            # calculate the L1 norm and L2 norm between the last_image and each pixel
            diff_l1 = torch.abs(reconstructed_images - original_decoded).sum(dim=1).sum(dim=0) # (256, 256)
            assert diff_l1.shape == (256, 256), f"diff_l1.shape: {diff_l1.shape}"
            diff_l2 = torch.pow(reconstructed_images - original_decoded, 2).sum(dim=1).sum(dim=0) # (256, 256)
            assert diff_l2.shape == (256, 256), f"diff_l2.shape: {diff_l2.shape}"
            length_to_diff_l1[i].append(diff_l1.cpu().numpy())
            length_to_diff_l2[i].append(diff_l2.cpu().numpy())
            buffer = []

        if count % args.save_every == 0:
            if not args.dry_run:
                # for each length, calculate the average
                np_l1 = [np.array(x).sum(axis=0) / count for x in length_to_diff_l1]
                np_l1 = np.array(np_l1) # (length, 256, 256)
                assert np_l1.shape == (length, 256, 256), f"np_l1.shape: {np_l1.shape}"
                np_l2 = [np.array(x).sum(axis=0) / count for x in length_to_diff_l2]
                np_l2 = np.array(np_l2) # (length, 256, 256)
                assert np_l2.shape == (length, 256, 256), f"np_l2.shape: {np_l2.shape}"
                np.savez_compressed(args.output_dir + "/diffs.npz", length_to_diff_l1=np_l1, length_to_diff_l2=np_l2)
        
    assert not buffer, f"buffer: {buffer}"
    # assert count == max_count, f"count: {count}, max_count: {max_count}"

    # save
    # for each length, calculate the average
    np_l1 = [np.array(x).sum(axis=0) / max_count for x in length_to_diff_l1]
    np_l1 = np.array(np_l1) # (length, 256, 256)
    assert np_l1.shape == (length, 256, 256), f"np_l1.shape: {np_l1.shape}"
    np_l2 = [np.array(x).sum(axis=0) / max_count for x in length_to_diff_l2]
    np_l2 = np.array(np_l2) # (length, 256, 256)
    assert np_l2.shape == (length, 256, 256), f"np_l2.shape: {np_l2.shape}"
    
    if not args.dry_run:
        np.savez_compressed(args.output_dir + "/diffs.npz", length_to_diff_l1=np_l1, length_to_diff_l2=np_l2)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/one-d-piece_s256.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="generated/one_d_piece/")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_every", type=int, default=50000)

    args = parser.parse_args()
    main(args)
