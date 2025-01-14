import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_data(npz_file):
    """Loads tokens and keys from a .npz file."""
    data = np.load(npz_file)
    return data['tokens'], data['keys']

def find_matching_images(tokens, keys, query):
    """Finds keys of images matching the query prefix in tokens."""
    query_len = len(query)
    matches = [key for token, key in zip(tokens, keys) if np.array_equal(token[:query_len], query)]
    # matches = [key for token, key in zip(tokens, keys)][:5]
    return matches

def plot_images_grid(keys, output_path, img_dir="images", grid_size=(5, 5)):
    """Plots images corresponding to given keys in a grid."""
    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    # Display images for each key if found, else leave blank
    for ax, key in zip(axes, keys):
        img_path = os.path.join(img_dir, f"image_{key:05d}.png")  # Assuming image format is PNG
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"Image {key}")
            ax.axis("off")
        else:
            ax.axis("off")
    
    # Save and close the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Image grid saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize images matching a token prefix in a grid.")
    parser.add_argument("--data", type=str, default="generated/tokens_one_d_piece_l256_len256/tokens.npz")
    parser.add_argument("--prefix", type=str, required=True, help="Token prefix for query")
    parser.add_argument("--out", type=str, default="generated/prefix_tree.png", help="Output path for the PNG file")
    parser.add_argument("--grid_size", type=str, default="5,5", help="Grid size, e.g., '5,5' for 5x5")
    parser.add_argument("--img_dir", type=str, default="generated/eval_png", help="Directory containing images")
    args = parser.parse_args()
    
    # Convert prefix input (comma-separated string) to integer list
    prefix = [int(x) for x in args.prefix.split(",")]
    grid_size = tuple(map(int, args.grid_size.split(',')))

    # Load data and find matching keys
    tokens, keys = load_data(args.data)
    # print available prefixes
    from collections import Counter
    first_tokens = Counter([token[0] for token in tokens])
    # Display only those with a count of 25 or more
    first_tokens = {k: v for k, v in first_tokens.items() if v >= 25}
    # sorted
    first_tokens = dict(sorted(first_tokens.items(), key=lambda x: x[1], reverse=True))
    print(f"Available prefixes: {first_tokens}")

    first_and_second_tokens = Counter([tuple(token[:2]) for token in tokens])
    # Display only those with a count of 3 or more
    first_and_second_tokens = {k: v for k, v in first_and_second_tokens.items() if v >= 3}
    # sorted
    first_and_second_tokens = dict(sorted(first_and_second_tokens.items(), key=lambda x: x[1], reverse=True))
    print(f"Available prefixes: {first_and_second_tokens}")

    keys = np.arange(tokens.shape[0])  # Use indices is enough
    matching_keys = find_matching_images(tokens, keys, prefix)
    # shuffle
    import random
    random.shuffle(matching_keys)
    print("Found", len(matching_keys), "matching images")
    
    # Plot images in a grid
    plot_images_grid(matching_keys, args.out, img_dir=args.img_dir, grid_size=grid_size)

if __name__ == "__main__":
    main()