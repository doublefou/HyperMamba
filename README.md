# HyperMamba

概述
本仓库实现了一个融合局部卷积分支与全局序列分支并通过层次化特征融合进行结合的骨架代码。目标是提供可复现的代码、示例以及使用说明，便于读者复现实验。

主要内容
- src/hypermamba/model.py: 主模型实现（Local、Global、GLE）
- examples/: 训练与推理示例
- docs/: 使用说明、复现实验步骤、引用建议
- LICENSE: 开源许可（MIT）

快速开始（本地）
1. 克隆仓库：
   git clone https://github.com/<doublefou>/HyperMamba.git
2. 创建 python 虚拟环境并安装依赖：
   conda create -n hypermamba python=3.10 -y
   conda activate hypermamba
   pip install -r requirements.txt
3. 运行示例（训练/推理）：
   python examples/train_example.py
   python examples/infer_example.py

复现实验建议
请参阅 docs/USAGE.md，包含依赖版本、训练超参、数据准备、如何引用本仓库与 Zenodo DOI。

License
MIT License（见 LICENSE 文件）
EOF
