# Evaluation

## Reconstruction Quality
For evaluating rFID and PSNR, use the following steps.

For tokenizer models, use the following command to evaluate.

```bash
WANDB_MODE=offline accelerate launch \
  --mixed_precision=bf16 \
  --num_machines=1 \
  --num_processes=1 \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=12389 \
  --same_network \
  eval/tokenizer/reconstruction_quality/eval_tokenizer.py \
    --config configs/one_d_piece_s256.yaml
```

For image formats, use the following command to evaluate for generated images using `scripts/reconstruct_tokenizer.py`.

```bash
python3 eval/tokenizer/reconstruction_quality/eval_image_format.py \
    --original_npy evaluation/original.npy
    --target_format evaluation/eval_jpeg_comp10/image_{:05d}.jpg
```

## Image Classification
Use the following command to evaluate for generated images using `scripts/reconstruct_tokenizer.py`.

```bash
python3 eval/tokenizer/image_classification/eval.py \
    --original_npy evaluation/original.npy
    --target_npy evaluation/one-d-piece-s-256.npy
```


## CLIP Embedding Reconstruction
Use the following command to evaluate for generated images using `scripts/reconstruct_tokenizer.py`.

```bash
python3 eval/tokenizer/clip_embedding_reconstruction/eval.py \
    --original_npy evaluation/original.npy
    --target_npy evaluation/one-d-piece-s-256.npy
```

## Depth Estimation
Use the following command to evaluate for generated images using `scripts/reconstruct_tokenizer.py`.

```bash
python3 eval/tokenizer/depth_estimation/eval.py \
    --original_npy evaluation/original.npy
    --target_npy evaluation/one-d-piece-s-256.npy
```

## Semantic Segmentation

1. Download the pretrained segmentation model

```
wget https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl-sup_sere_vit_small.pth
```

2. Inference 

```bash
python inference.py \
    --nb_classes 920 \
    --data_path "./imagenet_s_val/imagenet_s_webp" \
    --output_dir "./imagenet_s_val/mask/imagenet_s_webp" \
    --finetune "./imagenets_ssl-sup_sere_vit_small.pth" \
    --model "vit_small_patch16"
```

3. Evaluation

```bash
python evaluator.py \
    --gt-dir "./imagenet_s_val/mask/original_image" \
    --predict-dir "./imagenet_s_val/mask/imagenet_s_webp" \
    --mode "919" \
    --workers 32 \
    --name-list "./imagenets_im919_validation.txt"
```

For more details, please visit the official repository [https://github.com/LUSSeg/ImageNet-S](https://github.com/LUSSeg/ImageNet-S).


```bibtex
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```

## Object Detection

1. Download YoLov 11x model

```bash
cd ./eval/object_detection
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

2. Downlod CoCo val2017 images and labels

```
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# example
./eval/object_detection/coco
├── images
│   └── coco_val2017
└── labels
    └── val2017
        ├── 000000000139.txt
        ├── 000000000285.txt
        └── ...
```

3. Inference & Evaluation

```
# inference
python inference.py \
    --model ./yolo11x.pt \
    --image_dir "./coco/images/coco_val2017" \
    --annotation_file "./annotations/instances_val2017.json" \
    --batch_size 64 \
    --output_file "./output/val2017_webp.json" 
    --data_yaml "./annotations/coco.yaml"
    
# evaluation
python ./eval.py \
    --gt_file ./annotations/coco_original_image.json \
    --pred_file ./output/val_2017_webp.json
```

⚠️LICENSE:

For Object Detection, the ultralytics library and YoLo v11x are used. They are licensed under the [AGPL v3](./tokenizer/object_detection/LICENSE). In this repository, only the `eval/tokenizer/object_detection` directory is subject to AGPL v3 license.
For more information about ultralytics, please visit the official repository https://github.com/ultralytics/ultralytics.

```bibtex
@misc{lin2015microsoft,
      title={Microsoft COCO: Common Objects in Context},
      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
      year={2015},
      eprint={1405.0312},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Generation Quality
Before evaluating generation quality, download the following data.
```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

Then, run the following command.
```bash
python3 eval/generator/sample_imagenet.py \
    --config configs/training/generator/maskgit_one-d-piece_s256.yaml \
    --sample_dir evaluation/one-d-piece_s256 \
    --checkpoint maskgit_one-d-piece_s256.bin \
    --tokenizer_checkpoint one-d-piece_s256.bin
python3 eval/generator/guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz evaluation/one-d-piece_s256/one_d_piece_s256.npz
```
