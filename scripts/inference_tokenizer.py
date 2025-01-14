import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from PIL import Image
from modeling.one_d_piece import OneDPiece
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

def encode(model: OneDPiece, image: Image.Image):
    z_quantized, result_dict = model.encode(image)
    encoded_tokens = result_dict["min_encoding_indices"]
    return encoded_tokens

def decode(model: OneDPiece, encoded_tokens):
    return model.decode_tokens(encoded_tokens)

def main(args):
    print("Loading model model...")
    config = OmegaConf.load(args.config)
    was_finetune_decoder = config.model.vq_model.finetune_decoder
    config.model.vq_model.finetune_decoder = True
    config.model.vq_model.strict_length_assertion = False

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

    model_weight = torch.load(args.checkpoint, map_location="cpu")
    if not was_finetune_decoder:
        # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
        pretrained_tokenizer_weight = torch.load(
            config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
        )
        # Only keep the quantize and decoder part
        pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
        model_weight.update(pretrained_tokenizer_weight)

    model: OneDPiece = OneDPiece(config)
    model.load_state_dict(model_weight)
    model.eval()
    model.requires_grad_(False)

    device = "cuda"
    print(f"Working on device: {device}")
    model = model.to(device)

    transform = transforms.Compose([
        # remove alpha channel
        transforms.Lambda(lambda x: x.convert("RGB")),
        # resize height to 256, keep aspect ratio
        transforms.Resize(256),
        # clip center rectangle
        transforms.CenterCrop(256),
        # convert to tensor
        transforms.ToTensor()
    ])

    # mkdir
    Path(args.output_dir).mkdir(exist_ok=True)
    image = Image.open(args.image)
    image = transform(image).unsqueeze(0).to(device)
    encoded_tokens = encode(model, image)
    print("Encoded tokens:", encoded_tokens)

    for l in tqdm(range(0, config.model.vq_model.num_latent_tokens)):
        tokens = encoded_tokens[:, :, :l+1]
        reconstructed_image = decode(model, tokens)[0]
        reconstructed_image = (reconstructed_image * 255.0).clamp(0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
        path = Path(args.output_dir) / f"{l+1:04d}.png"
        Image.fromarray(reconstructed_image).save(path.as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/one-d-piece_s256.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default="assets/ILSVRC2012_val_00010240.png")
    parser.add_argument("--output_dir", type=str, default="generated/one_d_piece/")

    args = parser.parse_args()
    main(args)
