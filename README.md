# HyperMamba

## Overview
This repository provides a skeleton implementation of a model that integrates a local convolution branch and a global sequence branch, combined through hierarchical feature fusion. Its goal is to offer reproducible code, examples, and usage instructions to help readers reproduce the experiments.

## Abstract
Medical image classification is a cornerstone of computer-aided diagnosis, tasked with accurately identifying pathologies from complex and variable visual data. While recent advances in state space models (SSMs) like Mamba offer a compelling alternative to Transformers by enabling global context modelling with linear computational complexity, they often fall short in capturing the fine-grained local details that are crucial for discerning subtle lesion features. To address this, we propose HyperMamba, a novel hybrid architecture that synergistically integrates the strengths of convolutional neural networks (CNNs) and SSMs. Our approach is built upon two core innovations. First, we introduce a Dual-Branch Feature-Aware (DB-FA) block, which processes input through parallel pathways: a CNN branch to extract local spatial features and an SSM branch, equipped with a 2D selective scan (SS2D) mechanism, to capture long-range global dependencies. Second, we propose a Global-Local Enhancement (GLE) block that fuses these multi-scale features using complementary attention mechanisms—channel attention for global context and spatial attention for local details—thereby generating a more complete and discriminative feature representation. We rigorously evaluate HyperMamba on three public benchmarks (ISIC2018, Kvasir, COVID19-CT) and a self-constructed lung lesion dataset. Our model consistently achieves superior performance, with, for instance, HyperMamba-S attaining an accuracy of 86.40\% on Kvasir and 78.03\% on COVID19-CT, outperforming state-of-the-art methods. These results demonstrate that effectively fusing local and global features is critical for robust medical image classification. By combining the efficiency of SSMs with the fine-grained sensitivity of CNNs within a cohesive fusion framework, HyperMamba offers a promising new direction for developing accurate and efficient deep learning models for clinical applications.

## Main Contents
- `src/hypermamba/model.py`: Main model implementation (`Local`, `Global`, `GLE`)
- `examples/`: Training and inference examples
- `docs/`: Usage instructions, experiment reproduction steps, and citation recommendations
- `LICENSE`: Open-source license (MIT)

## Quick Start (Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/<doublefou>/HyperMamba.git
   ```
2. Create a Python virtual environment and install dependencies:
   ```bash
   conda create -n hypermamba python=3.10 -y
   conda activate hypermamba
   pip install -r requirements.txt
   ```
3. Run the examples (training/inference):
   ```bash
   python examples/train_example.py
   python examples/infer_example.py
   ```

## Recommendations for Reproducing Experiments
Please refer to `docs/USAGE.md`, which includes dependency versions, training hyperparameters, data preparation, and how to cite this repository and its Zenodo DOI.

## License
MIT License (see the `LICENSE` file)
