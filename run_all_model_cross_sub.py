import os
import argparse
import pickle
import glob
import numpy as np
import torch
import torch.nn as nn
import csv  # 新增
import datetime  # 新增
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, f1_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pyhealth.metrics import binary_metrics_fn

# ==========================================
# 引入所有模型定义
# ==========================================
from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)

# ==========================================
# 0. 全局配置
# ==========================================
BIOT_18_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("C3", "A2"), ("C4", "A1")
]

# ==========================================
# 1. Dataset
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
# 2. Lightning Module
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
# 3. 主运行逻辑
# ==========================================
def run_subject_level_cv(args):
    # --- A. 获取文件 ---
    search_pattern = os.path.join(args.root_path, "**/*.pkl")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if len(all_files) == 0:
        raise ValueError(f"未找到数据文件: {args.root_path}")
    
    subject_map = {}
    for f in all_files:
        filename = os.path.basename(f)
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
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = {
        "accuracy": [], "roc_auc": [], "pr_auc": [], "balanced_accuracy": [],
        "cohen_kappa": [], "f1_weighted": []
    }

    subject_splits = list(kf.split(unique_subjects))

    # --- C. 循环 5 次 ---
    for i in range(k_folds):
        print(f"\n{'#'*60}")
        print(f"Running Fold {i+1} / {k_folds} | Model: {args.model} | Seed: {args.seed}")
        print(f"{'#'*60}")

        test_sub_indices = subject_splits[i][1]
        val_sub_indices  = subject_splits[(i + 1) % k_folds][1]
        
        all_indices = np.arange(len(unique_subjects))
        exclude_indices = np.concatenate([test_sub_indices, val_sub_indices])
        train_sub_indices = np.setdiff1d(all_indices, exclude_indices)
        
        train_subs = unique_subjects[train_sub_indices]
        val_subs   = unique_subjects[val_sub_indices]
        test_subs  = unique_subjects[test_sub_indices]
        
        def get_files_from_subs(sub_list):
            files = []
            for s in sub_list:
                files.extend(subject_map[s])
            return np.array(files)

        train_files = get_files_from_subs(train_subs)
        val_files   = get_files_from_subs(val_subs)
        test_files  = get_files_from_subs(test_subs)
        
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

        # 模型初始化
        if args.model == "SPaRCNet":
            model = SPaRCNet(
                in_channels=args.in_channels,
                sample_length=int(args.sample_length * args.sampling_rate),
                n_classes=args.n_classes,
                block_layers=4, growth_rate=16, bn_size=16, drop_rate=0.5, conv_bias=True, batch_norm=True,
            )
        elif args.model == "ContraWR":
            model = ContraWR(in_channels=args.in_channels, n_classes=args.n_classes, fft=args.token_size, steps=args.hop_length // 5)
        elif args.model == "CNNTransformer":
            model = CNNTransformer(in_channels=args.in_channels, n_classes=args.n_classes, fft=args.sampling_rate, steps=args.hop_length // 5, dropout=0.2, nhead=4, emb_size=256, n_segments=5)
        elif args.model == "FFCL":
            model = FFCL(in_channels=args.in_channels, n_classes=args.n_classes, fft=args.token_size, steps=args.hop_length // 5, sample_length=int(args.sample_length * args.sampling_rate), shrink_steps=20)
        elif args.model == "STTransformer":
            model = STTransformer(emb_size=256, depth=4, n_classes=args.n_classes, channel_legnth=int(args.sampling_rate * args.sample_length), n_channels=args.in_channels)
        elif args.model == "BIOT":
            model = BIOTClassifier(n_classes=args.n_classes, n_channels=args.in_channels, n_fft=args.token_size, hop_length=args.hop_length)
            if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
                try:
                    model.biot.load_state_dict(torch.load(args.pretrain_model_path, map_location='cpu'), strict=True)
                except:
                    model.biot.load_state_dict(torch.load(args.pretrain_model_path, map_location='cpu'), strict=False)
        else:
            raise NotImplementedError

        lightning_model = LitModel_finetune(args, model)

        logger = TensorBoardLogger(
            # save_dir="logs_subject_level_cv", 
            save_dir=args.save_dir,
            name=f"{args.dataset}/{args.model}", 
            version=f"fold_{i+1}_seed_{args.seed}"
        )
        
        early_stop = EarlyStopping(monitor="val_auroc", patience=5, mode="max", verbose=True)
        checkpoint = ModelCheckpoint(
            monitor="val_auroc", mode="max",
            dirpath=os.path.join(args.save_dir, args.dataset, args.model, f"seed_{args.seed}", f"fold_{i+1}", "ckpt"),
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
        res = trainer.test(ckpt_path="best", dataloaders=test_loader)[0]
        
        fold_metrics["accuracy"].append(res.get("test_acc", 0))
        fold_metrics["roc_auc"].append(res.get("test_auroc", 0))
        fold_metrics["pr_auc"].append(res.get("test_pr_auc", 0))
        fold_metrics["balanced_accuracy"].append(res.get("test_bacc", 0))
        fold_metrics["cohen_kappa"].append(res.get("test_kappa", 0))
        fold_metrics["f1_weighted"].append(res.get("test_f1_w", 0))

        del model, lightning_model, trainer
        torch.cuda.empty_cache()

    # --- D. 控制台输出 ---
    print("\n" + "="*80)
    print(f"Experiment: {args.dataset} | Model: {args.model} | Seed: {args.seed}")
    print("="*80)
    
    summary_results = {}
    for key, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")
        summary_results[f"{key}_mean"] = mean_val
        summary_results[f"{key}_std"] = std_val

    # ==========================================
    # E. 保存汇总结果到文件 (重点修改部分)
    # ==========================================
    
    # 1. 确定保存目录
    results_dir = "summary_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. 方法一：保存到 CSV (适合汇总对比)
    csv_file_path = os.path.join(results_dir, "all_experiments_summary.csv")
    file_exists = os.path.isfile(csv_file_path)
    
    # 准备要写入的一行数据
    csv_row = {
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": args.dataset,
        "Model": args.model,
        "Seed": args.seed,
        "Epochs": args.epochs,
        "Acc_Mean": summary_results["accuracy_mean"],
        "Acc_Std": summary_results["accuracy_std"],
        "AUC_Mean": summary_results["roc_auc_mean"],
        "AUC_Std": summary_results["roc_auc_std"],
        "F1_Mean": summary_results["f1_weighted_mean"],
        "F1_Std": summary_results["f1_weighted_std"],
        "Kappa_Mean": summary_results["cohen_kappa_mean"],
        "Kappa_Std": summary_results["cohen_kappa_std"]
    }
    
    fieldnames = list(csv_row.keys())
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)
        
    print(f"\n[Success] Summary saved to CSV: {csv_file_path}")

    # 3. 方法二：保存为独立的 TXT 文件 (适合备份查看)
    # 文件名格式: results/Model_Seed.txt
    txt_file_name = f"{args.dataset}_{args.model}_Seed{args.seed}_results.txt"
    txt_file_path = os.path.join(results_dir, txt_file_name)
    
    with open(txt_file_path, "w", encoding='utf-8') as f:
        f.write(f"Experiment Summary\n")
        f.write(f"==================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Parameters: {vars(args)}\n\n")
        f.write(f"Metrics (Mean ± Std over 5 Folds):\n")
        for key, values in fold_metrics.items():
            f.write(f"{key:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        f.write(f"\nRaw Values per Fold:\n")
        for key, values in fold_metrics.items():
            f.write(f"{key}: {values}\n")
            
    print(f"[Success] Detailed report saved to TXT: {txt_file_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="PKL文件根目录")
    parser.add_argument("--dataset", type=str, default="SubLevelExp", help="实验/数据集名称")
    parser.add_argument("--model", type=str, default="BIOT", 
                        choices=["BIOT", "SPaRCNet", "ContraWR", "CNNTransformer", "FFCL", "STTransformer"])
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--in_channels", type=int, default=18)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=200)
    parser.add_argument("--token_size", type=int, default=200)
    parser.add_argument("--hop_length", type=int, default=100)
    parser.add_argument("--sample_length", type=float, default=10)
    parser.add_argument("--pretrain_model_path", type=str, default="")

    # 权重和结果保存位置
    parser.add_argument("--save_dir", type=str, default="logs_subject_level_cv", help="日志和权重保存的根目录")

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    run_subject_level_cv(args)