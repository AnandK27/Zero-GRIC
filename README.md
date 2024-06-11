`python train.py 48 /3d_data/retreiver/base/ 0 1`

`python train.py 48 /3d_data/retreiver/fusion/ 0 1`

# Evaluation Results

## Baseline

- Bleu_1: 0.754
- Bleu_2: 0.620
- Bleu_3: 0.487
- Bleu_4: 0.374
- METEOR: 0.281
- ROUGE_L: 0.582
- CIDEr: 1.223
- SPICE: 0.220

## Simple Fusion
  ### K=1
- Bleu_1: 0.843
- Bleu_2: 0.695
- Bleu_3: 0.512
- Bleu_4: 0.382
- METEOR: 0.301
- ROUGE_L: 0.838
- CIDEr: 1.602
- SPICE: 0.504

  ### K=2

- Bleu_1: 0.853
- Bleu_2: 0.703
- Bleu_3: 0.518
- Bleu_4: 0.384
- METEOR: 0.299
- ROUGE_L: 0.914
- CIDEr: 1.598
- SPICE: 0.530

## Graph based fusion
  ### K=1

- Bleu_1: 1.443
- Bleu_2: 1.156
- Bleu_3: 0.896
- Bleu_4: 0.698
- METEOR: 0.531
- ROUGE_L: 1.328
- CIDEr: 2.978
- SPICE: 0.905

  ### K=2

- Bleu_1: 1.573
- Bleu_2: 1.282
- Bleu_3: 0.993
- Bleu_4: 0.748
- METEOR: 0.529
- ROUGE_L: 1.394
- CIDEr: 3.073
- SPICE: 1.013