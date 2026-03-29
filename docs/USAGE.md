使用说明 (USAGE)

1) 环境与依赖
   - 建议使用 conda / venv 创建隔离环境。
   - 安装依赖：
       pip install -r requirements.txt

   注意：SS2D 的 selective_scan 依赖于外部 C++/CUDA 扩展（mamba_ssm 或 selective_scan）。
   若不需要运行 SS2D，示例仍可用于 smoke test（小批量随机前向）。

2) 运行示例
   python examples/train_example.py
   python examples/infer_example.py

3) 训练建议（示例）
   - 优化器：AdamW，学习率根据 batch-size 调整。
   - 数据集：ImageNet 或自定义数据集，确保图像尺寸与 patch_size 一致（或允许 padding）。
   - 随机种子：设置 torch.manual_seed(...) 与 numpy.random.seed(...)

4) 复现实验 checklist
   - 固定随机种子
   - 记录硬件（GPU 型号、驱动）
   - 记录依赖版本
   - 使用仓库 release 版本，并在论文摘要提供 release DOI

5) 发布与 DOI
   请参考 release.md（仓库根目录）
