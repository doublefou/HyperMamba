发布与 Zenodo DOI 说明

1) 在 GitHub 创建 release（tag）：
   git tag -a v0.1.0 -m "Initial release"
   git push origin v0.1.0

2) 在 Zenodo 登录并连接你的 GitHub，启用本仓库的自动归档。之后在 GitHub 每次发布 release，Zenodo 都会生成 DOI。

3) 在论文摘要与 README 中写入 release 的 GitHub 链接与 Zenodo DOI（永久链接）。

建议：在发布 release 时附带 docs/USAGE.md、CITATION.md 和一个包含关键超参的文件（如 configs/）以便可复现。
