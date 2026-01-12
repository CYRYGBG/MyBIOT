import os
import argparse
import pickle
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold  # 引入 KFold

# 引入 sklearn 指标
from sklearn.metrics import cohen_kappa_score, f1_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pyhealth.metrics import binary_metrics_fn

# 引入你的模型定义
from model import BIOTClassifier
# 引入损失函数
from utils import BCE

# ==========================================
# 0. 全局配置：BIOT 18 通道映射
# ==========================================
BIOT_18_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("C3", "A2"), ("C4", "A1")
]

# ==========================================
# 1. 自定义 Dataset (保持不变)
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
        
        raw_X = sample["X"]
        y = int(sample["y"])
        
        raw_ch_names = sample.get("ch_names", [])
        ch_map = {name.upper().replace('EEG', '').replace(' ', '').replace('-REF', ''): i 
                  for i, name in enumerate(raw_ch_names)}

        new_channels = []
        for ch1_name, ch2_name in BIOT_18_PAIRS:
            idx1 = ch_map.get(ch1_name)
            idx2 = ch_map.get(ch2_name)
            if idx1 is not None and idx2 is not None:
                new_channels.append(raw_X[idx1] - raw_X[idx2])
            elif idx1 is not None:
                new_channels.append(raw_X[idx1])
            else:
                new_channels.append(np.zeros_like(raw_X[0]))

        X = np.stack(new_channels)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        return torch.FloatTensor(X), y

# ==========================================
# 2. Lightning Module (保持不变)
# ==========================================
class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1)
            prob_class_1 = probs[:, 1]
            step_result = prob_class_1.cpu().numpy()
            step_gt = y.cpu().numpy()
        self.validation_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result_list = [out[0] for out in self.validation_step_outputs]
        gt_list = [out[1] for out in self.validation_step_outputs]
        
        if len(result_list) == 0:
            self.validation_step_outputs.clear()
            return

        result = np.concatenate(result_list)
        gt = np.concatenate(gt_list)

        if sum(gt) * (len(gt) - sum(gt)) != 0:
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            metrics = binary_metrics_fn(gt, result, metrics=["roc_auc", "accuracy"], threshold=self.threshold)
            y_pred_binary = (result >= self.threshold).astype(int)
            metrics["cohen_kappa"] = cohen_kappa_score(gt, y_pred_binary)
            metrics["f1_weighted"] = f1_score(gt, y_pred_binary, average='weighted')
        else:
            metrics = {"accuracy": 0.0, "roc_auc": 0.0, "cohen_kappa": 0.0, "f1_weighted": 0.0}
        
        self.log("val_acc", metrics["accuracy"], sync_dist=True)
        self.log("val_auroc", metrics["roc_auc"], sync_dist=True, prog_bar=True)
        self.log("val_kappa", metrics["cohen_kappa"], sync_dist=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1)
            prob_class_1 = probs[:, 1]
            step_result = prob_class_1.cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result_list = [out[0] for out in self.test_step_outputs]
        gt_list = [out[1] for out in self.test_step_outputs]

        if len(result_list) == 0:
            self.test_step_outputs.clear()
            return

        result = np.concatenate(result_list)
        gt = np.concatenate(gt_list)
            
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            metrics = binary_metrics_fn(gt, result, metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], threshold=self.threshold)
            y_pred_binary = (result >= self.threshold).astype(int)
            metrics["cohen_kappa"] = cohen_kappa_score(gt, y_pred_binary)
            metrics["f1_weighted"] = f1_score(gt, y_pred_binary, average='weighted')
        else:
            metrics = {"accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0, "roc_auc": 0.0, "cohen_kappa": 0.0, "f1_weighted": 0.0}
            
        self.log_dict({
            "test_acc": metrics["accuracy"],
            "test_auroc": metrics["roc_auc"],
            "test_pr_auc": metrics["pr_auc"],
            "test_bacc": metrics["balanced_accuracy"],
            "test_kappa": metrics["cohen_kappa"],
            "test_f1_w": metrics["f1_weighted"]
        }, sync_dist=True)
        
        self.test_step_outputs.clear()
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

# ==========================================
# 3. 主运行逻辑：被试级别（Subject-Level）五折交叉验证
# ==========================================
def run_subject_level_cv(args):
    # --- A. 获取文件并解析被试ID ---
    search_pattern = os.path.join(args.root_path, "**/*.pkl")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if len(all_files) == 0:
        raise ValueError(f"未找到数据文件: {args.root_path}")
    
    subject_map = {}
    for f in all_files:
        filename = os.path.basename(f)
        # 解析 ID: sub_xx_...
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'sub':
            subject_id = f"{parts[0]}_{parts[1]}"
        else:
            subject_id = "unknown"
            
        if subject_id not in subject_map:
            subject_map[subject_id] = []
        subject_map[subject_id].append(f)
    
    unique_subjects = np.array(sorted(list(subject_map.keys())))
    print(f"检测到被试数量: {len(unique_subjects)} | 总样本数: {len(all_files)}")

    # --- B. 被试级别划分 ---
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics = {
        "accuracy": [], "roc_auc": [], "pr_auc": [], "balanced_accuracy": [],
        "cohen_kappa": [], "f1_weighted": []
    }

    # 这里的 split 是对“人名列表”进行划分，而不是对文件进行划分
    # split 返回的是索引
    subject_splits = list(kf.split(unique_subjects))

    # --- C. 循环 5 次 ---
    for i in range(k_folds):
        print(f"\n{'#'*60}")
        print(f"Running Fold {i+1} / {k_folds} (Subject-Level / Cross-Subject)")
        print(f"{'#'*60}")

        # === 核心切分逻辑: 按照人名分配 ===
        # Test: 第 i 折选中的人
        # Val:  第 (i+1) 折选中的人
        # Train: 剩下的人
        
        test_sub_indices = subject_splits[i][1] # KFold[1] 是 test index
        val_sub_indices  = subject_splits[(i + 1) % k_folds][1]
        
        # 找出训练集索引 (全集 - Test - Val)
        all_indices = np.arange(len(unique_subjects))
        exclude_indices = np.concatenate([test_sub_indices, val_sub_indices])
        train_sub_indices = np.setdiff1d(all_indices, exclude_indices)
        
        # 获取人名
        train_subs = unique_subjects[train_sub_indices]
        val_subs   = unique_subjects[val_sub_indices]
        test_subs  = unique_subjects[test_sub_indices]
        
        print(f"Subjects -> Train: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)}")
        print(f"Test Subjects: {test_subs}")

        # 辅助函数：通过人名拿文件
        def get_files_from_subs(sub_list):
            files = []
            for s in sub_list:
                files.extend(subject_map[s])
            return np.array(files)

        train_files = get_files_from_subs(train_subs)
        val_files   = get_files_from_subs(val_subs)
        test_files  = get_files_from_subs(test_subs)
        
        print(f"Samples  -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

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
        
        if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
            print(f"Loading pretrained weights: {args.pretrain_model_path}")
            state_dict = torch.load(args.pretrain_model_path, map_location='cpu')
            model.biot.load_state_dict(state_dict, strict=True)

        lightning_model = LitModel_finetune(args, model)

        logger = TensorBoardLogger(
            save_dir="logs_subject_level_cv", 
            name=args.dataset, 
            version=f"fold_{i+1}"
        )
        
        early_stop = EarlyStopping(monitor="val_auroc", patience=5, mode="max", verbose=True)
        checkpoint = ModelCheckpoint(
            monitor="val_auroc", mode="max",
            dirpath=os.path.join("logs_subject_level_cv", args.dataset, f"fold_{i+1}", "ckpt"),
            filename="best",
            save_top_k=1
        )

        trainer = pl.Trainer(
            devices=[0] if torch.cuda.is_available() else "auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=args.epochs,
            logger=logger,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        
        trainer.fit(lightning_model, train_loader, val_loader)

        print(f"Testing Fold {i+1} with best validation model...")
        res = trainer.test(ckpt_path="best", dataloaders=test_loader)[0]
        
        fold_metrics["accuracy"].append(res.get("test_acc", 0))
        fold_metrics["roc_auc"].append(res.get("test_auroc", 0))
        fold_metrics["pr_auc"].append(res.get("test_pr_auc", 0))
        fold_metrics["balanced_accuracy"].append(res.get("test_bacc", 0))
        fold_metrics["cohen_kappa"].append(res.get("test_kappa", 0))
        fold_metrics["f1_weighted"].append(res.get("test_f1_w", 0))

        del model, lightning_model, trainer
        torch.cuda.empty_cache()

    # --- D. 输出最终统计 ---
    print("\n" + "="*80)
    print(args.root_path)
    print("Subject-Level (Cross-Subject) 5-Fold Cross-Validation Results")
    print("Strategy: Split by Unique Subjects (Unseen subjects in Test Set)")
    print("="*80)
    
    for key, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}  | Raw: {[round(x,4) for x in values]}")
    print("="*80)


if __name__ == "__main__":
    pl.seed_everything(3407, workers=True) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="PKL文件根目录")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="预训练权重路径")
    parser.add_argument("--dataset", type=str, default="SubLevelExp", help="实验名称")
    parser.add_argument("--in_channels", type=int, default=18, help="BIOT 需要 18 通道")
    parser.add_argument("--n_classes", type=int, default=2, help="分类数")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--token_size", type=int, default=200)
    parser.add_argument("--hop_length", type=int, default=100)
    parser.add_argument("--sample_length", type=float, default=10)

    args = parser.parse_args()
    
    run_subject_level_cv(args)