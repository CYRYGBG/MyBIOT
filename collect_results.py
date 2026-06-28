import os
import re
import pandas as pd
import glob

# ================= 配置区域 =================
# 结果文件所在的文件夹路径
RESULTS_DIR = '/home/yeqi3/cyr/code/MyBIOT/summary_results/'

# 输出 Excel 文件的名称
OUTPUT_FILE = 'experiment_summary_tables.xlsx'

# [新增] 指定数据集在表格中显示的顺序
DATASET_ORDER = ['read', 'type', 'read_new', 'type_new']
# ===========================================

def parse_result_file(file_path):
    """
    解析单个结果txt文件
    """
    data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 1. 提取模型名称
        model_match = re.search(r'Model:\s*(.+)', content)
        data['Model'] = model_match.group(1).strip() if model_match else 'Unknown'

        # 2. 提取随机种子
        seed_match = re.search(r'Seed:\s*(\d+)', content)
        data['Seed'] = int(seed_match.group(1)) if seed_match else -1

        # 3. 提取数据集名称 (作为列名)
        dataset_match = re.search(r'Dataset:\s*(.+)', content)
        data['Dataset'] = dataset_match.group(1).strip() if dataset_match else 'Unknown'

        # 4. 提取准确率 (Accuracy)
        acc_match = re.search(r'^accuracy\s*:\s*([\d\.]+)', content, re.MULTILINE)
        data['Accuracy'] = float(acc_match.group(1)) * 100 if acc_match else None

        # 5. 提取平衡准确率 (Balanced Accuracy)
        bal_acc_match = re.search(r'^balanced_accuracy\s*:\s*([\d\.]+)', content, re.MULTILINE)
        data['Balanced_Accuracy'] = float(bal_acc_match.group(1)) * 100 if bal_acc_match else None

    return data

def generate_formatted_table(df, metric_col):
    """
    生成包含平均行的格式化表格，并按指定顺序排列列
    :param df: 原始 DataFrame
    :param metric_col: 要处理的指标列名 ('Accuracy' 或 'Balanced_Accuracy')
    :return: 处理好的 DataFrame
    """
    # 1. 创建透视表：行=[Model, Seed], 列=[Dataset], 值=metric_col
    pivot_df = df.pivot_table(index=['Model', 'Seed'], 
                              columns='Dataset', 
                              values=metric_col)
    
    # 重置索引以便处理
    pivot_df = pivot_df.reset_index()
    
    # 确保 Seed 是数值类型以便正确排序
    pivot_df['Seed'] = pd.to_numeric(pivot_df['Seed'], errors='coerce')
    pivot_df = pivot_df.sort_values(by=['Model', 'Seed'])
    
    # 获取当前实际存在的数据集列 (排除 Model 和 Seed)
    existing_dataset_cols = [c for c in pivot_df.columns if c not in ['Model', 'Seed']]
    
    final_rows = []
    
    # 2. 按模型分组，计算平均值并插入
    # 获取唯一的模型列表
    models = pivot_df['Model'].unique()
    
    for model in models:
        # 获取该模型的所有行
        model_data = pivot_df[pivot_df['Model'] == model].copy()
        
        # 将原始行加入结果
        for _, row in model_data.iterrows():
            final_rows.append(row.to_dict())
            
        # 计算平均值行
        mean_values = model_data[existing_dataset_cols].mean()
        mean_row = mean_values.to_dict()
        mean_row['Model'] = model
        mean_row['Seed'] = '平均' # 标记为平均
        
        final_rows.append(mean_row)
        
    # 3. 构建最终 DataFrame
    result_df = pd.DataFrame(final_rows)
    
    # ================= [关键修改：强制列排序] =================
    # 1. 找出存在于 DATASET_ORDER 中的列，并按顺序排列
    sorted_cols = [ds for ds in DATASET_ORDER if ds in existing_dataset_cols]
    
    # 2. 防止有额外的数据集不在列表中 (作为容错，加在后面)
    others = [ds for ds in existing_dataset_cols if ds not in sorted_cols]
    
    final_dataset_order = sorted_cols + others
    
    # 3. 组合最终列顺序：Model, Seed, 然后是排序后的数据集
    cols = ['Model', 'Seed'] + final_dataset_order
    
    # 应用列顺序
    result_df = result_df[cols]
    # =========================================================
    
    # 格式化数字：保留2位小数
    return result_df.round(2)

def main():
    # 1. 读取文件
    file_pattern = os.path.join(RESULTS_DIR, '*_results.txt')
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"没有在 {RESULTS_DIR} 找到文件。")
        return

    print(f"正在处理 {len(files)} 个文件...")

    all_data = []
    for f in files:
        try:
            all_data.append(parse_result_file(f))
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    raw_df = pd.DataFrame(all_data)
    
    if raw_df.empty:
        print("没有提取到数据。")
        return

    # 2. 生成两个表
    print("正在生成准确率表...")
    acc_table = generate_formatted_table(raw_df, 'Accuracy')
    
    print("正在生成平衡准确率表...")
    bal_acc_table = generate_formatted_table(raw_df, 'Balanced_Accuracy')

    # 3. 写入 Excel (同一个文件的不同 Sheet)
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        acc_table.to_excel(writer, sheet_name='准确率 (Accuracy)', index=False)
        bal_acc_table.to_excel(writer, sheet_name='平衡准确率 (Balanced Acc)', index=False)

    print("="*30)
    print(f"完成！结果已保存至: {OUTPUT_FILE}")
    print(f"强制列顺序: {DATASET_ORDER}")
    print("Excel中包含两个Sheet：'准确率' 和 '平衡准确率'")
    print("="*30)
    
    # 打印预览
    print("\n[预览: 准确率表]")
    print(acc_table.head(10))

if __name__ == '__main__':
    main()