# 图像数据集划分与可视化工具

这是一个用于图像分类数据集的数据划分与数据分布可视化小项目。它旨在将前面学过的所有 Python 基础、文件路径、NumPy 随机数控制、Pandas 数据表操作、Matplotlib 绘图以及项目规范化工程组织（Git、.gitignore、README、代码与数据分离）融会贯通。

## 1. 项目目录结构

严格遵循“代码、数据、结果分开放”的规范进行工程管理：

```text
MUSIC实验室基础学习/
├── data/                     # 存放数据集原始文件 (被 .gitignore 忽略)
│   └── mini_imagenet/        # 自动生成的模拟分类图像集 (cat, dog, panda)
├── results/                  # 存放程序输出的划分结果与可视化图表
│   ├── train.csv             # 单次划分的训练集 CSV 表格
│   ├── val.csv               # 单次划分的验证集 CSV 表格
│   ├── train_fold_x.csv      # K折交叉划分的各折训练集表格
│   ├── val_fold_x.csv        # K折交叉划分的各折验证集表格
│   └── class_distribution.png # 类别数量分布柱状图 (单次划分或K折首折的可视化)
├── src/                      # 核心代码文件夹
│   ├── utils.py              # 工具函数
│   └── split_and_visualize.py # 主程序入口
├── .gitignore                # 告诉 Git 哪些大文件或缓存不需要上传
├── README.md                 # 项目文档说明
└── requirements.txt          # 依赖的 Python 第三方库声明
```

---

## 2. 核心功能

1. **设置种子**：利用 `utils.py` 中的 `set_seed` 函数锁定 Python 和 NumPy 的随机生成器状态，保证无论运行多少次，切分出来的训练集/验证集数据都是完全一致的。
2. **面向对象实现**：主逻辑由 `DatasetSplitter` 类实现，包含数据扫描、分层单次划分、分层 K 折划分和生成可视化图表。
3. **分层划分**：手动在各类别内部按比例进行随机划分并进行拼装，保持训练集和验证集的类别比例与原始数据集一致，防止类别不平衡影响模型评估。
4. **分层K折交叉划分**：手动实现分层 K 折交叉划分，对各类别分别按 K 折均匀分并组合，循环保存每一折的训练集和验证集。
5. **表格保存**：自动将划分结果保存为 CSV 表格，记录对应的图片绝对路径和标签。
6. **图表绘制**：在 `results/` 下生成柱状图，直观展示训练集和验证集中猫、狗、熊猫的样本占比，确保切分无误。

---

## 3. 使用方法与运行命令

支持在终端通过命令参数运行，参数说明如下：
* `--data_dir`：原始数据集文件夹路径（默认 `'data/mini_imagenet'`）
* `--val_ratio`：验证集划分比例（默认 `0.2`）
* `--split_method`：划分方法，可选值包括 `stratified`（单次分层）和 `kfold`（分层K折）
* `--k_folds`：K折交叉划分折数（默认 `5`）
* `--seed`：随机数种子（默认 `42`）
* `--output_dir`：结果输出目录（默认 `'results'`）

### 运行示例

* **单次分层划分**：
  ```bash
  python src/split_and_visualize.py --data_dir data/mini_imagenet --split_method stratified --val_ratio 0.2
  ```

* **3 折分层交叉验证划分**：
  ```bash
  python src/split_and_visualize.py --data_dir data/mini_imagenet --split_method kfold --k_folds 3
  ```
