# 图像数据集划分与可视化工具 (Image Dataset Splitter and Visualizer)

这是一个用于图像分类数据集的数据划分与数据分布可视化小项目。它旨在将前面学过的所有 Python 基础、文件路径、NumPy 随机数控制、Pandas 数据表操作、Matplotlib 绘图以及项目规范化工程组织（Git、.gitignore、README、代码与数据分离）融会贯通。

## 1. 项目目录结构

我们将严格遵循“代码、数据、结果分开放”的规范进行工程管理：

```text
MUSIC实验室基础学习/
├── data/                     # 存放数据集原始文件 (被 .gitignore 忽略)
│   └── mini_imagenet/        # 自动生成的模拟分类图像集 (cat, dog, panda)
├── results/                  # 存放程序输出的划分结果与可视化图表
│   ├── train_split.csv       # 训练集划分 CSV 表格
│   ├── val_split.csv         # 验证集划分 CSV 表格
│   └── class_distribution.png # 训练集和验证集的类别数量分布柱状图
├── src/                      # 核心代码文件夹
│   ├── utils.py              # 工具函数（如设置随机种子以确保可复现性）
│   └── split_dataset.py      # 主程序入口（命令行解析、数据读取、划分与绘图保存）
├── .gitignore                # 告诉 Git 哪些大文件或缓存不需要上传
├── README.md                 # 项目文档说明
└── requirements.txt          # 依赖的 Python 第三方库声明
```

---

## 2. 核心功能要求

1. **设置种子 (Reproducibility)**：利用 `utils.py` 中的 `set_seed` 函数锁定 Python 和 NumPy 的随机生成器状态，保证无论运行多少次，切分出来的训练集/验证集数据都是完全一致的。
2. **命令行参数支持 (argparse)**：支持在终端通过命令参数运行，如：
   ```bash
   python src/split_dataset.py --data_dir data/mini_imagenet --val_ratio 0.2 --seed 42
   ```
3. **面向对象实现 (OOP)**：主逻辑由 `DatasetSplitter` 类实现，包含数据扫描、打乱、切分和生成可视化图表。
4. **表格保存 (Pandas)**：保存 `train_split.csv` 和 `val_split.csv`，记录对应的图片绝对路径和标签。
5. **图表绘制 (Matplotlib)**：在 `results/` 下生成柱状图，直观展示训练集和验证集中猫、狗、熊猫的样本占比，确保切分无误。
