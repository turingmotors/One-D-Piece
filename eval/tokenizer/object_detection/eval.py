from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth using COCO format.")
    parser.add_argument('--gt_file', type=str, required=True, help="Path to the 256x256 GT JSON file.")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to the prediction JSON file.")
    return parser.parse_args()

def prepare_coco_gt_format(gt_file_path):
    # Load the GT predictions as annotations
    with open(gt_file_path, 'r') as f:
        predictions = json.load(f)

    # Organize predictions by image_id
    gt_by_image = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in gt_by_image:
            gt_by_image[image_id] = []
        gt_by_image[image_id].append(pred)

    # Create COCO formatted dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define unique categories from predictions
    category_ids = {pred["category_id"] for pred in predictions}
    coco_format["categories"] = [{"id": cid, "name": str(cid)} for cid in category_ids]

    annotation_id = 1

    # Prepare images and annotations
    for image_id, preds in gt_by_image.items():
        coco_format["images"].append({"id": image_id, "file_name": f"{image_id}.jpg"})  # Assuming file_name is based on image_id
        for pred in preds:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "area": pred["bbox"][2] * pred["bbox"][3],
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

    return coco_format

def evaluate_reconstructed_image(gt_file, prediction_file):
    # Prepare COCO GT dictionary
    coco_gt_dict = prepare_coco_gt_format(gt_file)

    # Load the GT COCO object from dictionary
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # Get GT image IDs
    gt_image_ids = coco_gt.getImgIds()

    # Load predictions and filter by GT image IDs
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    filtered_predictions = [pred for pred in predictions if pred["image_id"] in gt_image_ids]

    # Create a COCO formatted predictions dictionary
    coco_dt = coco_gt.loadRes(filtered_predictions)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main():
    args = parse_args()
    evaluate_reconstructed_image(args.gt_file, args.pred_file)

if __name__ == "__main__":
    main()
