import math
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_model_and_loss_module, get_titok_tokenizer, create_dataloader,
    auto_resume, save_checkpoint,
    train_one_epoch_classifier)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
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

    logger = setup_logger(name="TiTok-Classifier", log_level="INFO",
     output_file=f"{output_dir}/log{accelerator.process_index}.txt")

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

    tokenizer = get_titok_tokenizer(config)
    tokenizer.to(accelerator.device)

    model, ema_model, loss_module = create_model_and_loss_module(config, logger, accelerator, model_type=config.model.type)

    import torch.optim as optim
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    from torchlars import LARS

    optimizer = LARS(optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0))
    optimizer.defaults = optimizer.optim.defaults

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=int(1280000/16384*90*8),
        anneal_strategy="cos",
        pct_start=10/90,
        div_factor=100.0,
        final_div_factor=10000.0,
    )
    # optimizer, _ = create_optimizer(config, logger, model, loss_module, need_discrminator=False)

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    assert not config.training.use_ema, "EMA is not supported for linear probing in MAE Protocol."

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size_with_accum = total_batch_size_without_accum * config.training.gradient_accumulation_steps
    assert total_batch_size_with_accum == 16384, "Total batch size should be 16384 for linear probing in MAE Protocol."

    num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = 90 # This is a fixed value for linear probing in MAE Protocol.

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=False)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch_classifier(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer,
                            lr_scheduler,
                            train_dataloader,
                            eval_dataloader,
                            tokenizer,
                            global_step)
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
