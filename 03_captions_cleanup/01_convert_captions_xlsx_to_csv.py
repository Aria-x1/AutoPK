#!/usr/bin/env python3
"""
批量转换所有captions.xlsx为captions.csv
"""
from pathlib import Path
import pandas as pd

# 配置
BASE_DIR = Path("/Users/xxy/Documents/MML/AutoPK/extracted_images")

def convert_xlsx_to_csv():
    """批量转换"""
    
    print("="*70)
    print("Batch Converting captions.xlsx → captions.csv")
    print("="*70)
    print(f"Base directory: {BASE_DIR}\n")
    
    # 统计
    stats = {
        'found': 0,
        'converted': 0,
        'skipped': 0,
        'errors': 0
    }
    
    errors = []
    
    # 遍历所有子文件夹
    for pmid_dir in sorted(BASE_DIR.iterdir()):
        # 跳过非文件夹和隐藏文件夹
        if not pmid_dir.is_dir() or pmid_dir.name.startswith('.'):
            continue
        
        pmid = pmid_dir.name
        xlsx_file = pmid_dir / "captions.xlsx"
        csv_file = pmid_dir / "captions.csv"
        
        # 检查是否有xlsx文件
        if not xlsx_file.exists():
            continue
        
        stats['found'] += 1
        
        # 如果CSV已存在，跳过
        if csv_file.exists():
            print(f"⏭️  {pmid}: CSV already exists, skipping")
            stats['skipped'] += 1
            continue
        
        # 转换
        try:
            print(f"🔄 {pmid}: Converting...", end=' ')
            
            # 读取Excel
            df = pd.read_excel(xlsx_file)
            
            # 验证必需的列
            required_columns = ['image_filename', 'caption']
            missing = [col for col in required_columns if col not in df.columns]
            
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            # 保存为CSV
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Done ({len(df)} rows)")
            stats['converted'] += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            stats['errors'] += 1
            errors.append({'pmid': pmid, 'error': str(e)})
    
    # 打印总结
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Total xlsx files found:  {stats['found']}")
    print(f"Successfully converted:  {stats['converted']}")
    print(f"Skipped (csv exists):    {stats['skipped']}")
    print(f"Errors:                  {stats['errors']}")
    
    if errors:
        print("\nErrors detail:")
        for err in errors:
            print(f"  - {err['pmid']}: {err['error']}")
    
    print("="*70 + "\n")
    
    return stats

if __name__ == "__main__":
    stats = convert_xlsx_to_csv()
    
    if stats['errors'] == 0:
        print("✅ All conversions successful!")
    else:
        print(f"⚠️ Completed with {stats['errors']} errors")
