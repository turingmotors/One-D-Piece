#!/usr/bin/env python3

import random
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from modeling.titok import TiTok
from modeling.llamagen_diffustyle import GPT_models
from omegaconf import OmegaConf

def create_token_visualization(tokens, mask_id, step_idx, save_dir, grid_size=(4, 8), token_size=20):
    """
    トークンを可視化する関数
    mask_idは灰色、その他のトークンは色付きで表示
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # グリッドサイズを計算
    if tokens.shape[1] != grid_size[0] * grid_size[1]:
        # 32トークンの場合、4x8グリッド
        grid_size = (4, 8)
    
    # カラーマップを作成（mask_id以外のトークン用）
    num_tokens = tokens.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_tokens))
    
    # 画像を作成
    img_size = (grid_size[1] * token_size, grid_size[0] * token_size)
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # フォントを設定（小さめのサイズ）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    for i in range(tokens.shape[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        
        x = col * token_size
        y = row * token_size
        
        token_id = tokens[0, i].item()
        
        if token_id == mask_id:
            # マスクトークンは灰色
            color = (128, 128, 128)
            bg_color = (200, 200, 200)
        else:
            # その他のトークンは色付き
            color_idx = token_id % len(colors)
            color = tuple(int(c * 255) for c in colors[color_idx][:3])
            bg_color = tuple(int(c * 255 * 0.3) for c in colors[color_idx][:3])
        
        # 背景を描画
        draw.rectangle([x, y, x + token_size - 1, y + token_size - 1], fill=bg_color, outline=color, width=1)
        
        # トークンIDを描画
        text = str(token_id)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x + (token_size - text_width) // 2
        text_y = y + (token_size - text_height) // 2
        
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    # 保存
    filename = os.path.join(save_dir, f"tokens_step_{step_idx:02d}.png")
    img.save(filename)
    return filename

def create_attention_visualization(attention_masks, step_idx, save_dir, num_heads=8):
    """
    アテンションマスクを可視化する関数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 最初のバッチ、最初のヘッドのみを可視化
    attn_mask = attention_masks[0, 0].cpu().numpy()  # (seq_len, seq_len)
    
    # マスクを正規化（0-1の範囲に）
    attn_mask = (attn_mask - attn_mask.min()) / (attn_mask.max() - attn_mask.min() + 1e-8)
    
    # 可視化
    plt.figure(figsize=(8, 8))
    plt.imshow(attn_mask, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Attention Mask - Step {step_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # 保存
    filename = os.path.join(save_dir, f"attention_step_{step_idx:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_progressive_visualization(output_dir, mask_id, grid_size=(4, 8)):
    """
    生成過程の可視化を作成（既存のトークン可視化ファイルを使用）
    """
    import glob
    from PIL import Image
    
    output_dir = Path(output_dir)
    tokens_dir = output_dir / "tokens"
    
    if not tokens_dir.exists():
        print("No token visualization files found")
        return
    
    # トークンファイルを取得
    token_files = sorted(glob.glob(str(tokens_dir / "tokens_step_*_b0.png")))
    
    if not token_files:
        print("No token visualization files found")
        return
    
    # プログレッシブ可視化を作成
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, token_file in enumerate(token_files):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 画像を読み込み
        img = Image.open(token_file)
        ax.imshow(img)
        
        # ステップ番号を抽出
        step_num = token_file.split("step_")[1].split("_")[0]
        ax.set_title(f'Step {step_num}')
        ax.axis('off')
    
    plt.tight_layout()
    filename = output_dir / "progressive_generation.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Progressive visualization saved to {filename}")

@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              temperature=1.0,
              top_k=0,
              top_p=1.0,
              device="cuda",
              return_tensor=False,
              output_dir=None,
              steps=8):
    """
    visualize_llamagen_diffullama.pyのsample_fnを参考にした可視化付き生成関数
    """
    generator.eval()
    tokenizer.eval()

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    # 出力ディレクトリを設定（generateメソッド内で使用される）
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 可視化ディレクトリを設定
        import os
        os.environ['ATTN_VIZ_DIR'] = str(output_dir / "attention")
        os.environ['TOKEN_VIZ_DIR'] = str(output_dir / "tokens")

    # diffullamaのgenerateメソッドを呼び出し
    generated_tokens = generator.generate(
        cond=labels,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # 生成されたトークンの内容を確認
    print(f"Generated tokens shape: {generated_tokens.shape}")
    print(f"Token range: {generated_tokens.min().item()} - {generated_tokens.max().item()}")
    print(f"Mask tokens remaining: {(generated_tokens == generator.mask_id).sum().item()}")
    
    # マスクトークンが残っている場合は、ランダムなトークンで置き換える
    if (generated_tokens == generator.mask_id).any():
        print("Warning: Mask tokens found in generated output, replacing with random tokens")
        mask_positions = (generated_tokens == generator.mask_id)
        # マスクトークンをランダムな有効なトークンで置き換え
        random_tokens = torch.randint(0, generator.vocab_size, generated_tokens.shape, device=generated_tokens.device)
        generated_tokens = torch.where(mask_positions, random_tokens, generated_tokens)
    
    # 画像を生成
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image

def main(args):
    print("Loading models...")
    
    # 設定を読み込み
    config = OmegaConf.load(args.config)
    
    # トークナイザーを読み込み（titok-l32）
    tokenizer_checkpoint = config.experiment.get("tokenizer_checkpoint", None)
    if tokenizer_checkpoint is None:
        raise ValueError("tokenizer_checkpoint is not found in the config file.")

    if args.checkpoint is None:
        # 最新のチェックポイントを検索
        checkpoints = list(Path(config.experiment.output_dir).glob("checkpoint-*"))
        if not checkpoints:
            print(f"Error: No checkpoint directories found in {config.experiment.output_dir}")
            print("Available directories:")
            if Path(config.experiment.output_dir).exists():
                for item in Path(config.experiment.output_dir).iterdir():
                    print(f"  - {item.name}")
            else:
                print(f"  Directory {config.experiment.output_dir} does not exist")
            raise ValueError("No checkpoints found")
        
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        print(f"Found {len(checkpoints)} checkpoints: {[x.name for x in checkpoints]}")
        
        checkpoint_paths = [x / "ema_model" / "pytorch_model.bin" for x in checkpoints]
        # 存在するチェックポイントファイルをフィルタリング
        existing_checkpoints = [cp for cp in checkpoint_paths if cp.exists()]
        
        if not existing_checkpoints:
            print(f"Error: No checkpoint files found in any checkpoint directory")
            print("Checked paths:")
            for cp in checkpoint_paths:
                print(f"  - {cp} (exists: {cp.exists()})")
            raise ValueError("No checkpoint files found")
        
        checkpoint = existing_checkpoints[-1]
        print(f"Using the latest checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    elif args.checkpoint.isdigit():
        # 指定された番号のチェックポイントを検索
        checkpoint = Path(config.experiment.output_dir) / f"checkpoint-{args.checkpoint}" / "ema_model" / "pytorch_model.bin"
        if not checkpoint.exists():
            print(f"Error: Checkpoint file not found: {checkpoint}")
            print(f"Directory exists: {checkpoint.parent.exists()}")
            if checkpoint.parent.exists():
                print(f"Files in directory: {list(checkpoint.parent.iterdir())}")
            raise ValueError(f"Checkpoint file not found: {checkpoint}")
        print(f"Using the checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    else:
        # 直接パスが指定された場合
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"Error: Checkpoint file not found: {checkpoint}")
            raise ValueError(f"Checkpoint file not found: {checkpoint}")
        print(f"Using the specified checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    
    tokenizer_weight = torch.load(tokenizer_checkpoint, map_location="cpu")
    tokenizer = TiTok(config)
    tokenizer.load_state_dict(tokenizer_weight)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    
    # diffullamaモデルを読み込み
    model = GPT_models[config.model.generator.model_type](config)
    model.load_pretrained_ar(args.checkpoint)
    model.eval()
    model.requires_grad_(False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on device: {device}")
    
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    
    # ラベルを設定
    if args.labels:
        labels = args.labels
    else:
        # デフォルトのラベル（例：猫、犬、鳥など）
        labels = [7, 282, 14]  # 猫、犬、鳥
    
    print(f"Generating for labels: {labels}")
    
    # 出力ディレクトリを作成
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 可視化付き生成実行
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = sample_fn(
            generator=model,
            tokenizer=tokenizer,
            labels=labels,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
            return_tensor=False,
            output_dir=args.output_dir,
            steps=args.steps
        )

        # 生成された画像を保存
        for i, img in enumerate(result):
            path = Path(args.output_dir) / f"final_image_{i:04d}.png"
            Image.fromarray(img).save(path)
    
    # プログレッシブ可視化を作成
    create_progressive_visualization(args.output_dir, model.mask_id)
    
    print("Generation completed!")
    print(f"Generated {len(result)} images")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DiffuLlama generation process")
    parser.add_argument("--config", type=str, 
                       default="configs/training/generator/diffullama_titok_l32.yaml",
                       help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to the checkpoint file, checkpoint number, or full path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="generated/diffullama_visualization",
                       help="Output directory for visualization")
    parser.add_argument("--steps", type=int, default=8,
                       help="Number of generation steps")
    parser.add_argument("--labels", type=int, nargs='+', default=None,
                       help="Class labels to generate (e.g., 7 282 14 for cat, dog, bird)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling")
    
    args = parser.parse_args()
    main(args) 