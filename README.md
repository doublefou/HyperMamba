# HyperMamba

## Overview
This repository provides a skeleton implementation of a model that integrates a local convolution branch and a global sequence branch, combined through hierarchical feature fusion. Its goal is to offer reproducible code, examples, and usage instructions to help readers reproduce the experiments.

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
