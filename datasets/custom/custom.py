import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, root_path, split='train', inference=False):
        """
        root_path: 存放 pkl 文件的根目录 (例如 /usr/data/yeqi3/biot_processed_fixed)
        split: 'train', 'val', 'test'
        """
        super().__init__()
        self.root_path = root_path
        self.split = split
        
        # 1. 获取所有文件
        # 这里假设你把所有pkl都放在 root_path 下，或者你可以根据 split 读不同的子文件夹
        # 例如: if split == 'train': search_path = os.path.join(root_path, 'train')
        all_files = sorted(list(Path(root_path).rglob("*.pkl")))
        
        # 简单划分逻辑：80% 训练, 10% 验证, 10% 测试
        # 建议你在数据预处理阶段就分好文件夹，这里直接读文件夹更安全
        total = len(all_files)
        indices = list(range(total))
        # 这里为了演示简单，用了硬切分，实际建议根据文件名或预设列表切分
        split1 = int(0.8 * total)
        split2 = int(0.9 * total)
        
        if split == 'train':
            self.files = [all_files[i] for i in indices[:split1]]
        elif split == 'val':
            self.files = [all_files[i] for i in indices[split1:split2]]
        else: # test
            self.files = [all_files[i] for i in indices[split2:]]

        print(f"[{split.upper()}] Loaded {len(self.files)} samples from {root_path}")

        # 2. 读取一个样本以获取通道名称 (假设所有样本通道一致)
        if len(self.files) > 0:
            with open(self.files[0], 'rb') as f:
                tmp = pickle.load(f)
                self.channel_names = tmp['ch_names']
                self.fs = tmp['fs']
        else:
            self.channel_names = []
            self.fs = 200

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # pkl 内容: {'X': (C, T), 'y': int, 'ch_names': list, 'fs': 200}
        x = data['X']
        y = data['y']
        
        # 转 Tensor
        # BIOT 期望输入 (Channel, Time)
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(y).long()
        
        # 注意：这里我们不需要返回 channel_names，因为通常同一个 Dataset 内通道是固定的
        # 通道名将在初始化 Model 时用到
        return x_tensor, y_tensor