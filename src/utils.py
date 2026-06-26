# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import random
import numpy as np

def set_seed(seed=42):
    """
    固定所有随机源的种子以实现实验可复现性。
    """
    random.seed(seed)
    
    np.random.seed(seed)
    print(f"-> 已固定随机种子为: {seed}")
