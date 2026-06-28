import os
import re
import pandas as pd
import glob
import ast

# ================= 配置区域 =================
# 结果文件所在的文件夹路径
RESULTS_DIR = '/home/yeqi3/cyr/code/MyBIOT/summary_results/'

# 输出 Excel 文件的名称
OUTPUT_FILE = 'experiment_summary_tables_fixed.xlsx'

# 指定数据集在表格中显示的顺序
DATASET_ORDER = ['read', 'type', 'read_new', 'type_new']
# ===========================================

def parse_result_file(file_path):
    """
    解析单个结果txt文件，自动提取所有出现的指标 (支持带数字的指标名如 f1_weighted)
    """
    data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # --- 1. 基础信息提取 ---
        model_match = re.search(r'Model:\s*(.+)', content)
        data['Model'] = model_match.group(1).strip() if model_match else 'Unknown'

        seed_match = re.search(r'Seed:\s*(\d+)', content)
        data['Seed'] = int(seed_match.group(1)) if seed_match else -1

        dataset_match = re.search(r'Dataset:\s*(.+)', content)
        data['Dataset'] = dataset_match.group(1).strip() if dataset_match else 'Unknown'

        # --- 2. 策略A：优先从 "Metrics" 汇总区域提取 ---
        # [修改点] 将 [a-z_]+ 改为 [a-z0-9_]+ 以支持数字 (如 f1_weighted)
        summary_matches = re.findall(r'^\s*([a-z0-9_]+)\s*:\s*([\d\.]+)', content, re.MULTILINE)
        
        for metric_name, metric_value in summary_matches:
            data[metric_name] = float(metric_value) * 100

        # --- 3. 策略B：补救措施 (从 Raw Values 列表计算) ---
        # [修改点] 同样增加对数字的支持 [a-z0-9_]+
        raw_matches = re.findall(r'([a-z0-9_]+):\s*(\[.*?\])', content)
        
        for metric_name, raw_list_str in raw_matches:
            if metric_name not in data:
                try:
                    values = ast.literal_eval(raw_list_str)
                    if values and isinstance(values, list):
                        avg_val = sum(values) / len(values)
                        data[metric_name] = avg_val * 100
                except:
                    pass

    return data

def generate_formatted_table(df, metric_col):
    """
    生成格式化表格
    """
    pivot_df = df.pivot_table(index=['Model', 'Seed'], 
                              columns='Dataset', 
                              values=metric_col)
    
    pivot_df = pivot_df.reset_index()
    pivot_df['Seed'] = pd.to_numeric(pivot_df['Seed'], errors='coerce')
    pivot_df = pivot_df.sort_values(by=['Model', 'Seed'])
    
    existing_dataset_cols = [c for c in pivot_df.columns if c not in ['Model', 'Seed']]
    final_rows = []
    
    models = pivot_df['Model'].unique()
    for model in models:
        model_data = pivot_df[pivot_df['Model'] == model].copy()
        for _, row in model_data.iterrows():
            final_rows.append(row.to_dict())
            
        mean_values = model_data[existing_dataset_cols].mean()
        mean_row = mean_values.to_dict()
        mean_row['Model'] = model
        mean_row['Seed'] = '平均' 
        final_rows.append(mean_row)
        
    result_df = pd.DataFrame(final_rows)
    
    sorted_cols = [ds for ds in DATASET_ORDER if ds in existing_dataset_cols]
    others = [ds for ds in existing_dataset_cols if ds not in sorted_cols]
    
    cols = ['Model', 'Seed'] + sorted_cols + others
    result_df = result_df[cols]
    
    return result_df.round(2)

def main():
    file_pattern = os.path.join(RESULTS_DIR, '*_results.txt')
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"未找到文件: {RESULTS_DIR}")
        return

    print(f"正在扫描 {len(files)} 个文件...")
    all_data = []
    for f in files:
        all_data.append(parse_result_file(f))

    raw_df = pd.DataFrame(all_data)
    
    if raw_df.empty:
        print("未提取到数据")
        return

    # 排除非指标列
    meta_cols = ['Model', 'Seed', 'Dataset']
    metric_columns = [col for col in raw_df.columns if col not in meta_cols]

    print(f"修正后提取到的指标: {metric_columns}")
    
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        for metric in metric_columns:
            print(f"正在生成表格 Sheet: {metric} ...")
            try:
                table = generate_formatted_table(raw_df, metric)
                # 格式化 Sheet 名称
                sheet_name = metric.replace('_', ' ').title()[:31]
                table.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                print(f"跳过指标 {metric}: {e}")

    print("="*30)
    print(f"结果已保存: {OUTPUT_FILE}")
    print("="*30)

if __name__ == '__main__':
    main()