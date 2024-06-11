# Zero-GRIC: Graph Retriever for Zero-shot Image Captioning

This repository contains the code for the paper "Zero-GRIC: Graph Retriever for Zero-shot Image Captioning" for ECE 285 UCSD course project.

## Authors
- Anand Kumar
- Apeksha Gaonkar

## Abstract
 We developed Zero-GRIC, a robust end-to-end framework for zero-shot image captioning that leverages CLIP embeddings and graph convolutional networks (GCN). Our approach is structured in three main steps:Knowledge Database: We construct a database of CLIP embeddings for image-text pairs.Retrieval Augmentation and Fusion: We retrieve the top-$k$ nearest image-text embedding pairs from this database and use GCN to fuse the top-k textual information.Caption Generation: The fused information is then used to generate the final image captions.Our model framework, trained and evaluated on the MS-COCO dataset, achieves competitive results as measured by the CIDEr score. Notably, our approach requires fewer training parameters compared to state-of-the-art (SOTA) models.

## Downloading Dataset 

You can download the COCO dataset from the following link: [COCO Dataset](https://cocodataset.org/#download)

## How to run the code

### Requirements
  For running the code, you need to create conda environment using the `environment.yml` file. You can create the environment using the following command:
  `conda env create -f environment.yml`

### Training

You can use the following commands to fine-tune the model for COCO dataset.

`python train.py 48 base/ 0 1`

`python train.py 48 fusion/ 0 1`

where base/ and fusion/ are the directories where the model checkpoints will be saved. The first argument is batch size and last argument is the number of neighbors to consider for the graph based retriever.

### Evaluation

You can use the following commands to evaluate the model on COCO dataset.

#### For base pretrained model
`python validate_base_mode.py 48 save_prediction_loc`

#### For fusion models
`python validate_fusion_model.py 48 save_prediction_loc base/ 1`


where save_prediction_loc is the directory where the predictions will be saved, second last argument is the model saved location and final argument is the number of neighbors.

## Evaluation Results

Full table comparison

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE |
|-------|--------|--------|--------|--------|--------|---------|-------|-------|
| Baseline | 0.754 | 0.620 | 0.487 | 0.374 | 0.281 | 0.582 | 1.223 | 0.220 |
| Prompt Fusion K=1 | 0.843 | 0.695 | 0.512 | 0.382 | 0.301 | 0.838 | 1.602 | 0.504 |
| Prompt Fusion K=2 | 0.853 | 0.703 | 0.518 | 0.384 | 0.299 | 0.914 | 1.598 | 0.530 |
| Graph Fusion K=1 | 1.443 | 1.156 | 0.896 | 0.698 | 0.531 | 1.328 | 2.978 | 0.905 |
| **Graph Fusion K=2** | **1.573** | **1.282** | **0.993** | **0.748** | **0.529** | **1.394** | **3.073** | **1.013** |

### Baseline

- Bleu_1: 0.754
- Bleu_2: 0.620
- Bleu_3: 0.487
- Bleu_4: 0.374
- METEOR: 0.281
- ROUGE_L: 0.582
- CIDEr: 1.223
- SPICE: 0.220

### Prompt Fusion
  #### K=1
- Bleu_1: 0.843
- Bleu_2: 0.695
- Bleu_3: 0.512
- Bleu_4: 0.382
- METEOR: 0.301
- ROUGE_L: 0.838
- CIDEr: 1.602
- SPICE: 0.504

  #### K=2

- Bleu_1: 0.853
- Bleu_2: 0.703
- Bleu_3: 0.518
- Bleu_4: 0.384
- METEOR: 0.299
- ROUGE_L: 0.914
- CIDEr: 1.598
- SPICE: 0.530

### Graph based fusion
  #### K=1

- Bleu_1: 1.443
- Bleu_2: 1.156
- Bleu_3: 0.896
- Bleu_4: 0.698
- METEOR: 0.531
- ROUGE_L: 1.328
- CIDEr: 2.978
- SPICE: 0.905

  #### K=2

- Bleu_1: 1.573
- Bleu_2: 1.282
- Bleu_3: 0.993
- Bleu_4: 0.748
- METEOR: 0.529
- ROUGE_L: 1.394
- CIDEr: 3.073
- SPICE: 1.013


## References

- [1] [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)