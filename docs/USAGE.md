**Usage Instructions (USAGE)**

1) **Environment and Dependencies**
   - It is recommended to create an isolated environment using conda or venv.
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

   **Note:** SS2D's `selective_scan` depends on external C++/CUDA extensions (`mamba_ssm` or `selective_scan`).
   If you do not need to run SS2D, the examples can still be used for a smoke test (small-batch random forward pass).

2) **Running the Examples**
   ```bash
   python examples/train_example.py
   python examples/infer_example.py
   ```

3) **Training Recommendations (Example)**
   - Optimizer: AdamW; adjust the learning rate according to the batch size.
   - Dataset: ImageNet or a custom dataset; ensure that the image size matches `patch_size` (or that padding is allowed).
   - Random seed: set `torch.manual_seed(...)` and `numpy.random.seed(...)`.

4) **Experiment Reproducibility Checklist**
   - Fix the random seed.
   - Record the hardware information (GPU model, driver version).
   - Record dependency versions.
   - Use the repository release version, and provide the release DOI in the paper abstract.

5) **Release and DOI**
   Please refer to `release.md` in the repository root directory.
