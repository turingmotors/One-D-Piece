import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import models

class SegmentationFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.png', '.jpg', '.jpeg', '.jp2', 'webp'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path).convert('RGB')
        height, width = sample.size[1], sample.size[0]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path, height, width

def custom_collate_fn(batch):
    images, paths, heights, widths = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, paths, heights, widths

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--nb_classes', type=int, default=50)
    parser.add_argument('--output_dir', type=str, required=True, help='The path to save segmentation masks')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--finetune', type=str, required=True, help='The model checkpoint file')
    parser.add_argument('--pretrained_rfnext', default='', help='Pretrained weights for RF-Next')
    parser.add_argument('--model', default='vit_small_patch16', help='Model architecture')
    parser.add_argument('--patch_size', type=int, default=4, help='For convnext/rfconvnext, the number of output channels is nb_classes * patch_size * patch_size.')
    parser.add_argument('--max_res', default=1000, type=int, help='Maximum resolution for evaluation. 0 for disable.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # build model
    model = models.__dict__[args.model](args)
    model = model.cuda()
    model.eval()
    
    # load checkpoints
    checkpoint = torch.load(args.finetune)['model']
    model.load_state_dict(checkpoint, strict=True)
    
    # data preparation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = SegmentationFolder(
        root=args.data_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for images, paths, heights, widths in tqdm(dataloader, desc='Processing'):
        images = images.cuda()

        with torch.no_grad():
            outputs = model(images)

            for idx in range(images.size(0)):
                output = outputs[idx]
                path = paths[idx]
                height = heights[idx]
                width = widths[idx]

                original_name = os.path.basename(path)
                name = original_name.split('.')[0]
            
                save_path = os.path.join(output_dir, f"{name}.png")

                if os.path.exists(save_path):
                    continue

                H, W = height, width
                if H * W > args.max_res * args.max_res and args.max_res > 0:
                    if H > W:
                        output_resized = F.interpolate(
                            output.unsqueeze(0), (args.max_res, int(args.max_res * W / H)),
                            mode='bilinear', align_corners=False)
                    else:
                        output_resized = F.interpolate(
                            output.unsqueeze(0), (int(args.max_res * H / W), args.max_res),
                            mode='bilinear', align_corners=False)
                else:
                    output_resized = F.interpolate(output.unsqueeze(0), (H, W), mode='bilinear', align_corners=False)

                output_resized = torch.argmax(output_resized, dim=1).squeeze(0)
                
                res = torch.zeros(size=(output_resized.shape[0], output_resized.shape[1], 3), dtype=torch.uint8)
                res[:, :, 0] = output_resized % 256
                res[:, :, 1] = output_resized // 256
                res_img = Image.fromarray(res.cpu().numpy(), mode='RGB')
                res_img.save(save_path)

if __name__ == '__main__':
    main()
