# RESOLVE: Graph Contrastive Self-Supervised Learning Framework

This repository contains the source code for **RESOLVE**, a graph contrastive self-supervised learning framework for functional connectivity analysis.

## 🔹 Setup Instructions

1. **Download the required datasets** before running the code.  
2. **Run the following commands** to pre-train, fine-tune, and test on the MDD dataset.

### 🔹 Pre-training on MDD
```bash
python main.py --training_mode pre_train --dataset MDD

```
### 🔹 Fine-tuning and testing on MDD
```bash
python main.py --training_mode fine_tune_test --dataset MDD

```

### 🔹 Hyperparameters for MDD
$\lambda_1 = 0.5, \lambda_2 = 0.8, \lambda_3 = 1
$
