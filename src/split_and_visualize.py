import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from utils import set_seed

class DatasetSplitter:
    def __init__(self, data_dir, val_ratio, k_folds, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.k_folds = k_folds
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_and_split(self):
        image_path = []
        image_path = list(self.data_dir.rglob('*.jpg'))
        if len(image_path) == 0:
            raise FileNotFoundError(f'未在{self.data_dir}中找到jpg图像文件')
        print(f'扫描完毕,一共发现{len(image_path)}张图片')
        data_list = []
        for path in image_path:
            data_list.append(
                {
                    'image_path': str(path),
                    'label': path.parent.name
                }
            )
        return pd.DataFrame(data_list)
    
    def stratified_split(self, df):
        train_list = []
        val_list = []
        for label, group in df.groupby('label'):
            shuffle_group = group.sample(frac=1).reset_index(drop=True)
            val_size = int(len(shuffle_group) * (1 - self.val_ratio))
            val_part = shuffle_group.iloc[val_size:]
            train_part = shuffle_group.iloc[0:val_size]
            train_list.append(train_part)
            val_list.append(val_part)
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        return train_df, val_df

    def k_fold_split(self, df):
        folds_data = [[] for _ in range(self.k_folds)]
        for label, group in df.groupby('label'):
            shuffled_group = group.sample(frac=1.0).reset_index(drop=True)
            split_group = np.array_split(shuffled_group, self.k_folds)
            for j in range(self.k_folds):
                folds_data[j].append(split_group[j])
        fold_dfs = []
        for j in range(self.k_folds):
            fold_df = pd.concat(folds_data[j], ignore_index=True)
            fold_dfs.append(fold_df)
        
        for i in range(self.k_folds):
            val_df = fold_dfs[i].sample(frac=1.0).reset_index(drop=True)
            train_parts = [fold_dfs[j] for j in range(self.k_folds) if j != i]
            train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0).reset_index(drop=True)
            train_path = self.output_dir / f'train_fold_{i+1}.csv'
            val_path = self.output_dir / f'val_fold_{i+1}.csv'
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            print(f"第 {i+1}/{self.k_folds} 折划分完成：训练集 {len(train_df)} 张，验证集 {len(val_df)} 张")
            if i == 0:
                self.visualize_distribution(train_df, val_df)


    def visualize_distribution(self, train_df, val_df):
        train_counts = train_df['label'].value_counts()
        val_counts = val_df['label'].value_counts()

        classes = sorted(list(train_counts.keys()))

        train_vals = [train_counts.get(cls, 0) for cls in classes]
        val_vals = [val_counts.get(cls, 0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7,5))     
        ax.bar(x - width/2, train_vals, width, label='Train', color='royalblue')
        ax.bar(x + width/2, val_vals, width, label='Validation', color='tomato') 
        ax.set_ylabel('Number of Images')
        ax.set_title('Dataset Split Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)  

        fig_path = self.output_dir / 'class_distribution.png'
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"分布图已成功保存至: {fig_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Image Dataset Splitter')

    parser.add_argument('--data_dir', type=str, default='data/mini_imagenet', help='Raw dataset folder')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--split_method', type=str, default='stratified', choices=['stratified', 'kfold'], 
                        help='Split method: stratified split or k-fold cross validation')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for K-fold split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for CSVs and plots')
    args = parser.parse_args()

    try:
        set_seed(args.seed)
        splitter = DatasetSplitter(
            data_dir=args.data_dir,
            val_ratio=args.val_ratio,
            k_folds=args.k_folds, 
            output_dir=args.output_dir
        )
        full_df = splitter.scan_and_split()
        if args.split_method == 'stratified':
            print("正在进行分层单次划分...")
            train_df, val_df = splitter.stratified_split(full_df)
            train_df.to_csv(splitter.output_dir / 'train.csv', index=False)
            val_df.to_csv(splitter.output_dir / 'val.csv', index=False)
            splitter.visualize_distribution(train_df, val_df)
        elif args.split_method == 'kfold':
            print(f"正在进行 {args.k_folds} 折分层交叉划分...")
            splitter.k_fold_split(full_df)
        print("====== 数据集划分与可视化任务全部顺利完成！ ======")
        
    except Exception as e:
        print(f'发生错误:{e}')

if __name__ == '__main__':
    main()

    