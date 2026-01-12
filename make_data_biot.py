# -*- coding: utf-8 -*-
# @Time     : 2025/10/29
# @Notes    : BIOT 模型专用预处理脚本
#             1. 滤波 + 陷波 + 重采样 (200Hz)
#             2. Z-Score 标准化 (per channel)
#             3. 固定长度填充/截断 (Padding/Cropping)
#             4. 保存为 PKL (含数据、标签、通道名)

import os
import re
import gc
import pickle
import numpy as np
from pathlib import Path
from fractions import Fraction
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import resample_poly
import mne

# 尝试导入 h5py 以支持 v7.3 mat 文件
try:
    import h5py
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

mne.set_log_level("WARNING")


def now():
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_channel_name(n: str) -> str:
    """
    BIOT 核心步骤：标准化通道名
    将 'EEG Fp1-REF' -> 'FP1'，以便模型能查找到 Embedding
    """
    # 移除 EEG, -REF, 空格，转大写
    clean = str(n).replace('EEG', '').replace('-REF', '').replace(' ', '').upper()
    return clean


def _load_mat_auto(mat_path: str):
    """自动加载 .mat (v5/v7.3)"""
    try:
        d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return d
    except NotImplementedError:
        if not _HAS_H5PY: raise
        with h5py.File(mat_path, "r") as f:
            d = {}
            # h5py 读取的数据往往需要转置或解码，这里做基础读取
            for key in f.keys():
                d[key] = np.array(f[key])
            return d


def _extract_fields(d: dict):
    """从字典中提取 eeg, labels, fs, ch_names"""
    # 1. 提取 EEG 数据
    if 'eeg_data' in d:
        eeg = np.asarray(d['eeg_data'])
    elif 'data' in d: # 兼容常见命名
        eeg = np.asarray(d['data'])
    else:
        raise KeyError("未找到 eeg_data")

    # 确保维度 (Epochs, Channels, Time)
    if eeg.ndim == 2:
        # 假设是 (Channels, Time)，增加一个 Epoch 维度
        eeg = eeg[np.newaxis, :, :]
    
    # h5py 读取时维度可能是 (Time, Channels) 或 (Channels, Time)，需根据具体情况转置
    # 这里假设数据已经是 (Epoch, Channel, Time) 或者代码能处理
    
    # 2. 提取 Labels
    if 'labels' in d:
        labels = np.asarray(d['labels']).ravel()
    else:
        # 如果没有标签，默认为 -1
        labels = np.full(eeg.shape[0], -1)

    # 3. 提取采样率
    fs_key = next((k for k in ['fsample', 'fs', 'sfreq', 'Fs'] if k in d), None)
    if fs_key:
        val = np.asarray(d[fs_key])
        fs = float(val.item() if val.ndim == 0 else val.ravel()[0])
    else:
        raise KeyError("未找到采样率 (fsample/fs)")

    # 4. 提取通道名
    ch_names = []
    if 'ch_names' in d:
        raw = np.asarray(d['ch_names'])
        try:
            # 处理 numpy 字符串数组
            if raw.dtype.kind in ('U', 'S'):
                ch_names = [str(x) for x in raw.tolist()]
            # 处理 h5py 对象引用或 v7.3 char 矩阵
            else:
                flat = raw.ravel()
                ch_names = [str(x) for x in flat]
        except:
            pass
    
    # 如果没读到通道名，生成默认的 (BIOT 会受到影响，但至少能跑)
    if not ch_names or len(ch_names) != eeg.shape[1]:
        print(f"[WARN] 通道名丢失或长度不匹配，使用默认 CH0, CH1...")
        ch_names = [f"CH{i}" for i in range(eeg.shape[1])]

    return eeg.astype(np.float64, copy=False), labels, ch_names, fs


def process_mat_file(mat_path: str, dump_folder: str, 
                     band=(0.1, 75.0), notch=50.0, 
                     out_fs=200, target_sec=10):
    """
    处理单个 mat 文件
    target_sec: 目标时长(秒)。数据将通过截断或补零固定到这个长度。
    """
    try:
        os.makedirs(dump_folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(mat_path))[0]
        
        # 1. 加载数据
        d = _load_mat_auto(mat_path)
        eeg, labels, raw_ch_names, fs = _extract_fields(d)
        
        # 2. 清洗通道名 (BIOT 必需)
        clean_ch_names = [_normalize_channel_name(n) for n in raw_ch_names]

        E, C, T = eeg.shape
        target_pts = int(out_fs * target_sec) # 例如 200 * 10 = 2000 点
        saved_count = 0

        for ep in range(E):
            x = eeg[ep] # (C, T)
            
            # --- A. 滤波 (Bandpass + Notch) ---
            # 使用 mne.filter 对 numpy 数组直接操作
            try:
                x = mne.filter.filter_data(x, fs, band[0], band[1], method='iir', 
                                           verbose=False)
                if notch > 0:
                    x = mne.filter.notch_filter(x, fs, [notch], verbose=False)
            except Exception as e:
                print(f"[WARN] 滤波失败 {base}: {e}")
                continue

            # --- B. 重采样 (Resample) ---
            # 计算重采样后的点数
            curr_sec = x.shape[1] / fs
            new_pts = int(curr_sec * out_fs)
            
            if abs(out_fs - fs) > 0.1:
                x = resample_poly(x, up=out_fs, down=int(fs), axis=-1)
            
            # --- C. Z-Score 标准化 (重要：在补零之前做) ---
            # BIOT 要求输入 zero mean, unit variance
            x = zscore(x, axis=-1)
            
            # 检查 NaN (如果某通道全是0，zscore会产生nan)
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x) # 将 NaN 替换为 0

            # --- D. 长度对齐 (Padding / Cropping) ---
            curr_pts = x.shape[1]
            if curr_pts > target_pts:
                # 截断
                x = x[:, :target_pts]
            elif curr_pts < target_pts:
                # 补零 (右侧填充)
                pad_width = target_pts - curr_pts
                # ((通道前, 通道后), (时间前, 时间后))
                x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            
            # 此时 x shape 必定为 (C, target_pts)

            # --- E. 保存 ---
            # 获取标签
            try:
                lbl = labels[ep]
                y = int(lbl) if isinstance(lbl, (int, float, np.number)) else 0
            except:
                y = 0

            out_name = f"{base}_ep{ep:04d}.pkl"
            save_path = os.path.join(dump_folder, out_name)
            
            data_dict = {
                "X": x.astype(np.float32), # 转为 float32 节省空间
                "y": y,
                "ch_names": clean_ch_names,
                "fs": out_fs
            }
            
            with open(save_path, "wb") as f:
                pickle.dump(data_dict, f)
            
            saved_count += 1

        return saved_count

    except Exception as e:
        print(f"[ERROR] 处理文件失败 {mat_path}: {e}")
        return 0


def _gather_files(root_dir):
    root = Path(root_dir)
    # 查找所有 .mat 文件
    mats = list(root.glob("sub_*_simplified.mat"))
    # 按文件名排序
    mats.sort(key=lambda x: x.name)
    return [str(p) for p in mats]


def main():
    # ================= 配置区域 =================
    # BIOT 标准配置
    OUT_FS = 200           # 目标采样率
    TARGET_SEC = 10        # 固定时间窗口 (秒)，不足补零
    
    # 滤波配置
    BAND_LOW = 0.1
    BAND_HIGH = 75.0
    NOTCH = 50.0
    
    # 路径配置
    BASE_DATA_PATH = '/usr/data/yeqi3/data_clean_easy'
    BASE_OUT_PATH  = '/usr/data/yeqi3/biot_processed_fixed' # 建议用个新名字

    # 任务列表 (Input Dir, Output Dir)
    tasks = [
        (os.path.join(BASE_DATA_PATH, 'read'), os.path.join(BASE_OUT_PATH, 'read')),
        (os.path.join(BASE_DATA_PATH, 'type'), os.path.join(BASE_OUT_PATH, 'type')),
        (os.path.join(BASE_DATA_PATH, 'read_new'), os.path.join(BASE_OUT_PATH, 'read_new')),
        (os.path.join(BASE_DATA_PATH, 'type_new'), os.path.join(BASE_OUT_PATH, 'type_new')),
    ]
    
    WORKERS = 16 # 并行进程数
    # ===========================================

    total_samples = 0
    
    for in_dir, out_dir in tasks:
        if not os.path.exists(in_dir):
            print(f"[SKIP] 目录不存在: {in_dir}")
            continue
            
        print(f"\n>>> 开始处理: {in_dir}")
        print(f"    输出至: {out_dir}")
        
        mats = _gather_files(in_dir)
        if not mats:
            print("    未找到 .mat 文件")
            continue
            
        # 准备并行参数
        # 参数结构: (mat_path, dump_folder, band, notch, out_fs, target_sec)
        params = []
        for p in mats:
            params.append((p, out_dir, (BAND_LOW, BAND_HIGH), NOTCH, OUT_FS, TARGET_SEC))
            
        # 并行执行
        from multiprocessing import Pool
        
        # 实际使用的 worker 数量
        real_workers = min(len(mats), os.cpu_count(), WORKERS)
        
        if real_workers > 1:
            with Pool(real_workers) as pool:
                # starmap 用于自动解包参数元组
                results = pool.starmap(process_mat_file, params)
                count = sum(results)
        else:
            # 单进程调试用
            count = 0
            for param in params:
                count += process_mat_file(*param)
        
        print(f"    完成。该目录生成样本数: {count}")
        total_samples += count

    print(f"\nAll Done! 总共生成样本: {total_samples}")
    print(f"数据已按照 BIOT 格式 (200Hz, {TARGET_SEC}s, Z-Scored, Channel Names) 保存。")


if __name__ == "__main__":
    main()