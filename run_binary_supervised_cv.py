import os
import argparse
import pickle
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 引入 sklearn 指标
from sklearn.metrics import cohen_kappa_score, f1_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pyhealth.metrics import binary_metrics_fn

# 引入你的模型定义 (确保 model.py 在同级目录)
from model import BIOTClassifier
# 引入损失函数 (确保 utils.py 在同级目录)
# 这里的 BCE 实际上在下面的代码中被 CrossEntropy 替代了，但为了兼容 import 保留
from utils import BCE

# ==========================================
# 0. 全局配置：BIOT 18 通道映射
# ==========================================
# 格式: (通道1, 通道2) -> 表示计算 通道1 - 通道2
BIOT_18_PAIRS = [
    # --- 左颞区 (Left Temporal) ---
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    # --- 右颞区 (Right Temporal) ---
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    # --- 左旁矢状区 (Left Parasagittal) ---
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    # --- 右旁矢状区 (Right Parasagittal) ---
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    # --- 中心区 (Central) ---
    ("C3", "A2"), ("C4", "A1")
]

# ==========================================
# 1. 自定义 Dataset (含 64->18 通道映射)
# ==========================================
class CustomPKLLoader(Dataset):
    def __init__(self, file_paths, sampling_rate=200):
        self.files = file_paths
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with open(path, "rb") as f:
            sample = pickle.load(f)
        
        # 原始数据: (64, T)
        raw_X = sample["X"]
        y = int(sample["y"])
        
        # 获取通道名列表
        raw_ch_names = sample.get("ch_names", [])
        # 建立 名字 -> 索引 的映射字典
        ch_map = {name.upper().replace('EEG', '').replace(' ', '').replace('-REF', ''): i 
                  for i, name in enumerate(raw_ch_names)}

        new_channels = []
        
        # 遍历 BIOT 要求的 18 对通道
        for ch1_name, ch2_name in BIOT_18_PAIRS:
            idx1 = ch_map.get(ch1_name)
            idx2 = ch_map.get(ch2_name)

            # 情况 1: 两个通道都找到了 (完美匹配) -> 做差分
            if idx1 is not None and idx2 is not None:
                diff_data = raw_X[idx1] - raw_X[idx2]
                new_channels.append(diff_data)
            
            # 情况 2: 只找到了第一个通道 -> 保留单通道
            elif idx1 is not None:
                new_channels.append(raw_X[idx1])
                
            # 情况 3: 没找到 -> 补全 0
            else:
                new_channels.append(np.zeros_like(raw_X[0]))

        # 堆叠成 (18, T)
        X = np.stack(new_channels)

        # 归一化 (Robust Scaling)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )

        return torch.FloatTensor(X), y

# ==========================================
# 2. Lightning Module (修复了 API 和维度问题)
# ==========================================
class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        
        # 存储列表 (Lightning 2.0 必需)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        # 模型输出 (B, 2)
        logits = self.model(X)
        
        # 使用 CrossEntropy 处理 (B,2) 输出和 (B,) 标签
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self.model(X) # Shape: (B, 2)
            
            # --- 维度修复: 只取正类概率 ---
            probs = torch.softmax(logits, dim=-1)
            prob_class_1 = probs[:, 1] # 变成 (B,)
            
            step_result = prob_class_1.cpu().numpy()
            step_gt = y.cpu().numpy()
        
        self.validation_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        # 1. 聚合结果
        result_list = []
        gt_list = []
        for out in self.validation_step_outputs:
            result_list.append(out[0])
            gt_list.append(out[1])
        
        if len(result_list) == 0:
            self.validation_step_outputs.clear()
            return

        result = np.concatenate(result_list)
        gt = np.concatenate(gt_list)

        # 2. 计算指标
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            # 动态寻找最佳阈值
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            metrics = binary_metrics_fn(
                gt, result, metrics=["roc_auc", "accuracy"], 
                threshold=self.threshold
            )
            
            # === 新增: 计算 Kappa 和 F1 Weighted ===
            y_pred_binary = (result >= self.threshold).astype(int)
            kappa = cohen_kappa_score(gt, y_pred_binary)
            f1_w = f1_score(gt, y_pred_binary, average='weighted')
            
            metrics["cohen_kappa"] = kappa
            metrics["f1_weighted"] = f1_w
        else:
            metrics = {"accuracy": 0.0, "roc_auc": 0.0, "cohen_kappa": 0.0, "f1_weighted": 0.0}
        
        self.log("val_acc", metrics["accuracy"], sync_dist=True)
        self.log("val_auroc", metrics["roc_auc"], sync_dist=True, prog_bar=True)
        self.log("val_kappa", metrics["cohen_kappa"], sync_dist=True)
        
        # 3. 关键: 清空列表
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self.model(X)
            
            # --- 维度修复 ---
            probs = torch.softmax(logits, dim=-1)
            prob_class_1 = probs[:, 1]
            
            step_result = prob_class_1.cpu().numpy()
            step_gt = y.cpu().numpy()
        
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result_list = []
        gt_list = []
        for out in self.test_step_outputs:
            result_list.append(out[0])
            gt_list.append(out[1])

        if len(result_list) == 0:
            self.test_step_outputs.clear()
            return

        result = np.concatenate(result_list)
        gt = np.concatenate(gt_list)
            
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            # 使用验证集确定的阈值
            metrics = binary_metrics_fn(
                gt, result, metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                threshold=self.threshold 
            )
            
            # === 新增: 计算 Kappa 和 F1 Weighted ===
            y_pred_binary = (result >= self.threshold).astype(int)
            kappa = cohen_kappa_score(gt, y_pred_binary)
            f1_w = f1_score(gt, y_pred_binary, average='weighted')
            
            metrics["cohen_kappa"] = kappa
            metrics["f1_weighted"] = f1_w
        else:
            metrics = {
                "accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0, "roc_auc": 0.0,
                "cohen_kappa": 0.0, "f1_weighted": 0.0
            }
            
        self.log_dict({
            "test_acc": metrics["accuracy"],
            "test_auroc": metrics["roc_auc"],
            "test_pr_auc": metrics["pr_auc"],
            "test_bacc": metrics["balanced_accuracy"],
            "test_kappa": metrics["cohen_kappa"],
            "test_f1_w": metrics["f1_weighted"]
        }, sync_dist=True)
        
        # 关键: 清空列表
        self.test_step_outputs.clear()
        
        # 返回结果给主循环
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

# ==========================================
# 3. 主运行逻辑：严谨的 5折切分
# ==========================================
def run_strict_cv(args):
    # --- A. 获取并打乱文件 ---
    search_pattern = os.path.join(args.root_path, "**/*.pkl")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if len(all_files) == 0:
        raise ValueError(f"未找到数据文件: {args.root_path}")
    
    # 随机打乱 (设定种子以复现)
    np.random.seed(42)
    all_files = np.array(all_files)
    np.random.shuffle(all_files)
    
    print(f"检测到总样本数: {len(all_files)}")

    # --- B. 切分为 5 份 (Chunks) ---
    k_folds = 5
    folds = np.array_split(all_files, k_folds)
    
    # 初始化结果记录器
    fold_metrics = {
        "accuracy": [], "roc_auc": [], "pr_auc": [], "balanced_accuracy": [],
        "cohen_kappa": [], "f1_weighted": []
    }

    # --- C. 循环 5 次 ---
    for i in range(k_folds):
        print(f"\n{'#'*60}")
        print(f"Running Fold {i+1} / {k_folds}")
        print(f"{'#'*60}")

        # === 核心切分逻辑: 1 Test / 1 Val / 3 Train ===
        test_files = folds[i]                 # 当前份做测试
        val_idx = (i + 1) % k_folds           # 下一份做验证
        val_files = folds[val_idx]
        
        # 剩下的做训练
        train_chunks = []
        for j in range(k_folds):
            if j != i and j != val_idx:
                train_chunks.append(folds[j])
        train_files = np.concatenate(train_chunks)
        
        print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

        # 构建 DataLoader
        train_loader = DataLoader(
            CustomPKLLoader(train_files, args.sampling_rate),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, persistent_workers=True
        )
        val_loader = DataLoader(
            CustomPKLLoader(val_files, args.sampling_rate),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, persistent_workers=True
        )
        test_loader = DataLoader(
            CustomPKLLoader(test_files, args.sampling_rate),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, persistent_workers=True
        )

        # 初始化模型
        model = BIOTClassifier(
            n_classes=args.n_classes,
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        
        # 加载预训练权重 (每次重置)
        if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
            print(f"Loading pretrained weights: {args.pretrain_model_path}")
            state_dict = torch.load(args.pretrain_model_path, map_location='cpu')
            model.biot.load_state_dict(state_dict, strict=True)

        lightning_model = LitModel_finetune(args, model)

        # Logger 和 Callbacks
        logger = TensorBoardLogger(
            save_dir="logs_strict_cv", 
            name=args.dataset, 
            version=f"fold_{i+1}"
        )
        
        early_stop = EarlyStopping(monitor="val_auroc", patience=5, mode="max", verbose=True)
        
        checkpoint = ModelCheckpoint(
            monitor="val_auroc", mode="max",
            dirpath=os.path.join("logs_strict_cv", args.dataset, f"fold_{i+1}", "ckpt"),
            filename="best",
            save_top_k=1
        )

        # Trainer
        trainer = pl.Trainer(
            devices=[0] if torch.cuda.is_available() else "auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=args.epochs,
            logger=logger,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        
        # 训练
        trainer.fit(lightning_model, train_loader, val_loader)

        # 测试
        print(f"Testing Fold {i+1} with best validation model...")
        # test 返回的是一个 list of dict，取出第一个
        res = trainer.test(ckpt_path="best", dataloaders=test_loader)[0]
        
        # 收集结果
        # 注意: Lightning 的 test 返回的 keys 通常带有 'test_' 前缀
        # 或者我们直接看 LitModel 中 on_test_epoch_end 返回的 dict 结构
        # 这里统一用 .get 获取，防止 key 报错
        fold_metrics["accuracy"].append(res.get("test_acc", 0))
        fold_metrics["roc_auc"].append(res.get("test_auroc", 0))
        fold_metrics["pr_auc"].append(res.get("test_pr_auc", 0))
        fold_metrics["balanced_accuracy"].append(res.get("test_bacc", 0))
        fold_metrics["cohen_kappa"].append(res.get("test_kappa", 0))
        fold_metrics["f1_weighted"].append(res.get("test_f1_w", 0))

        # 清理
        del model, lightning_model, trainer
        torch.cuda.empty_cache()

    # --- D. 输出最终统计 ---
    print("\n" + "="*80)
    print("Strict 5-Fold Cross-Validation Results")
    print("Strategy: 3 Train / 1 Val / 1 Test")
    print("="*80)
    
    for key, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}  | Raw: {[round(x,4) for x in values]}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--root_path", type=str, required=True, help="PKL文件根目录")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="预训练权重路径")
    
    # 实验参数
    parser.add_argument("--dataset", type=str, default="Experiment1", help="实验名称")
    parser.add_argument("--in_channels", type=int, default=18, help="BIOT 需要 18 通道")
    parser.add_argument("--n_classes", type=int, default=2, help="分类数")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # BIOT 参数
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--token_size", type=int, default=200)
    parser.add_argument("--hop_length", type=int, default=100)
    parser.add_argument("--sample_length", type=float, default=10)

    args = parser.parse_args()
    
    run_strict_cv(args)