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
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.path.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.path.pardir))
sys.path.append(parent_dir)

from pathlib import Path
import pprint

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    create_model, create_dataloader,
    create_evaluator, auto_resume, eval_reconstruction
)

import argparse

def main(args):
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = OmegaConf.load(args.config)

    if args.length is None:
        args.length = config.model.vq_model.num_latent_tokens

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

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)
    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model = accelerator.prepare(model)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # Start training.
    logger.info(f""" Start evaluation of the model. """)
    global_step = 0

    global_step, _ = auto_resume(config, logger, accelerator, ema_model, -1, strict=True)

    accelerator.print(f"Evaluation of the checkpoint started.")
    # Evaluate reconstruction.
    additional_args = {}
    if config.model.type in ["titok"]:
        pass
    elif config.model.type in ["one_d_piece"]:
        additional_args = dict(length=args.length)
    else:
        raise ValueError(f"Unsupported model type {config.model.type}")

    logger.info(f"Computing metrics on the validation set.")
    if config.training.get("use_ema", False):
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        # Eval for EMA.
        eval_scores = eval_reconstruction(
            model,
            eval_dataloader,
            accelerator,
            evaluator,
            pretrained_tokenizer=None,
            **additional_args
        )
        logger.info(
            f"EMA EVALUATION "
            f"Step: {global_step + 1} "
        )
        logger.info(pprint.pformat(eval_scores))
        if accelerator.is_main_process:
            eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
            accelerator.log(eval_log, step=global_step + 1)
        ema_model.restore(model.parameters())
    else:
        # Eval for non-EMA.
        eval_scores = eval_reconstruction(
            model,
            eval_dataloader,
            accelerator,
            evaluator,
            pretrained_tokenizer=None,
            **additional_args
        )

        logger.info(
            f"Non-EMA EVALUATION "
            f"Step: {global_step + 1} "
        )
        logger.info(pprint.pformat(eval_scores))
        if accelerator.is_main_process:
            eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
            accelerator.log(eval_log, step=global_step + 1)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--length", type=int, default=None)
    args = parser.parse_args()
    main(args)
