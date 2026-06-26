from scipy.stats import triang
import argparse
from IPython.core.pylabtools import figsize
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from utils import set_seed

class DatasetSplitter:
    def __init__(self, data_dir, val_ratio, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_and_split(self):

        image_paths = []
        image_paths = list(self.data_dir.rglob('*.jpg'))
        if len(image_paths) == 0:
            raise FileNotFoundError(f'在{self.data_dir}下未找到jpg图像文件')
        print(f'扫描完毕，一共发现{len(image_paths)}张图片')

        data_list = []
        for path in image_paths:
            data_list.append({
                'image_path': str(path.resolve()),
                'label': path.parent.name
            }) 

        df = pd.DataFrame(data_list) 
        df_shuffle = df.sample(frac=1.0).reset_index(drop=True)
        
        train_df = None
        val_df = None
        split_idx = int(len(df_shuffle) * (1 - self.val_ratio))
        train_df = df_shuffle.iloc[:split_idx]
        val_df = df_shuffle.iloc[split_idx:]
        print(f'划分结果：训练集{len(train_df)}张,验证集{len(val_df)}张')

        train_df.to_csv(self.output_dir / 'train_split.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_split.csv', index=False)
        return train_df, val_df
    
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for CSVs and plots')
    args = parser.parse_args()

    try:
        set_seed(args.seed)
        splitter = DatasetSplitter(
            data_dir=args.data_dir,
            val_ratio=args.val_ratio,
            output_dir=args.output_dir
        )
        train_df, val_df = splitter.scan_and_split()
        splitter.visualize_distribution(train_df, val_df)
        print("====== 数据集划分与可视化任务全部顺利完成！ ======")

    except Exception as e:
        print(f'发生错误:{e}')

if __name__ == '__main__':
    main()

    