import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(grid_size, length_to_diff, output_path):
    # set up the figure
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axs = axs.flatten()

    # log scale
    length_to_diff = np.log(length_to_diff)

    # set up the color range
    vmin, vmax = np.min(length_to_diff), np.max(length_to_diff)

    # plot the heatmaps
    for i in range(len(length_to_diff)):
        ax = axs[i]
        _ = ax.imshow(length_to_diff[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.axis('off')     # display no axis
        ax.set_xticks([])  # remove x-axis ticks
        ax.set_yticks([])  # remove y-axis ticks

    # don't display the rest of the axes
    for j in range(len(length_to_diff), len(axs)):
        axs[j].axis('off')

    # save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved heatmap plot to {output_path}")

def main(args):
    npz = np.load(args.input + "/diffs.npz")
    length_to_diff_l1 = npz["length_to_diff_l1"] # (length, 256, 256)
    length_to_diff_l2 = npz["length_to_diff_l2"] # (length, 256, 256)
    length = length_to_diff_l1.shape[0]
    print(length)

    # calculate the grid size
    grid_size = int(np.ceil(np.sqrt(length)))

    plot_heatmap(grid_size, length_to_diff_l1, args.input + "/diffs_l1.png")
    print(f"Saved heatmaps to {args.input}/diffs_l1.png")

    plot_heatmap(grid_size, length_to_diff_l2, args.input + "/diffs_l2.png")
    print(f"Saved heatmaps to {args.input}/diffs_l2.png")

    length_to_diff_l1_mean = np.mean(length_to_diff_l1, axis=1).mean(axis=1)
    length_to_diff_l2_mean = np.mean(length_to_diff_l2, axis=1).mean(axis=1)

    plt.plot(length_to_diff_l1_mean, label="L1")
    # make it log-scale
    plt.legend()
    plt.savefig(args.input + "/diffs_amp_l1.png")
    plt.close()
    print(f"Saved mean plot to {args.input}/diffs_amp_l1.png")

    plt.plot(length_to_diff_l2_mean, label="L2")
    # make it log-scale
    plt.legend()
    plt.savefig(args.input + "/diffs_amp_l2.png")
    plt.close()
    print(f"Saved mean plot to {args.input}/diffs_amp_l2.png")

    # display the heatmaps
    length_to_diff_l1_mean = np.log(length_to_diff_l1_mean.reshape(1, -1).repeat(10 * length // 32, axis=0))
    length_to_diff_l2_mean = np.log(length_to_diff_l2_mean.reshape(1, -1).repeat(10 * length // 32, axis=0))

    # heatmaps for L1
    plt.imshow(length_to_diff_l1_mean, cmap='viridis')
    plt.axis('off')
    plt.savefig(args.input + "/diff_amp_heatmap_l1.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # heatmaps for L2
    plt.imshow(length_to_diff_l2_mean, cmap='viridis')
    plt.axis('off')
    plt.savefig(args.input + "/diff_amp_heatmap_l2.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved L1 heatmap to {args.input}/diff_amp_heatmap_l1.png")
    print(f"Saved L2 heatmap to {args.input}/diff_amp_heatmap_l2.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()
    main(args)