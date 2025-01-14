import argparse
from ultralytics import YOLO
import json
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and validation on COCO dataset with batch processing.")
    parser.add_argument('--model', type=str, required=True, help="Path to YOLO model file (e.g., yolov8x.pt)")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing COCO images (e.g., val2017)")
    parser.add_argument('--annotation_file', type=str, required=True, help="Path to COCO annotation file (e.g., instances_val2017.json)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference")
    parser.add_argument('--output_file', type=str, default="results.json", help="Output file for COCO format results")
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to the dataset YAML file (e.g., coco.yaml)")
    return parser.parse_args()

def run_inference(model, image_dir, batch_size):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    results = []
    batch_images = []

    for image_path in tqdm(image_paths):
        batch_images.append(image_path)

        if len(batch_images) == batch_size or image_path == image_paths[-1]:
            predictions_batch = model(batch_images)

            for img_path, predictions in zip(batch_images, predictions_batch):
                image_id = int(os.path.splitext(os.path.basename(img_path))[0])
                for pred in predictions.boxes:
                    bbox = pred.xywh[0].tolist()
                    score = pred.conf[0].item()
                    category_id = int(pred.cls[0].item()) + 1

                    result = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                        "score": score
                    }
                    results.append(result)

            batch_images = []

    return results

def save_results(results, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Inference results saved to {output_file}")

def main(args):
    model = YOLO(args.model)
    results = run_inference(model, args.image_dir, args.batch_size)
    save_results(results, args.output_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
