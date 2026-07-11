"""BC5CDR 資料集前處理：把原始 CSV 轉成標準 Dataset CSV（BC5CDR 格式）供 Pipeline 使用。

輸出 Dataset CSV 欄位（依 PMID 分組，每組一筆 task）：
  - taskID  : 批次層級識別碼（PMID）
  - title   : 文章標題（對應 taskTemplate 的 {title} 佔位符）
  - abstract: 文章摘要（對應 taskTemplate 的 {abstract} 佔位符）
  - items   : JSON array，含該 PMID 下所有 entity pair（sentID/label/e1/e2）

使用方式：python DatasetPreprocess/bc5cdr.py
"""
import json
import logging
import pandas as pd
from pathlib import Path

INPUT_PATH  = "data\\bcvcdr\\BCVCDR_Processed.csv"
OUTPUT_PATH = "data\\bcvcdr\\sample_tasks.csv"


def preprocess():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', on_bad_lines='warn')

    requiredCols = {'ID', 'PMID', 'Title', 'Abstract', 'E1_Name', 'E2_Name', 'Relation_Type'}
    missing = requiredCols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tasks = []
    for pmid, group in df.groupby('PMID', sort=False):
        # gold label 須與 labelSet.classes (['no','yes']) 對齊：Relation_Type==CID → yes。
        items = [
            {
                'sentID': str(row['ID']),
                'label':  "yes" if str(row['Relation_Type']).strip().upper() == "CID" else "no",
                'e1':     str(row['E1_Name']),
                'e2':     str(row['E2_Name'])
            }
            for _, row in group.iterrows()
        ]
        tasks.append({
            'taskID':   str(pmid),
            'title':    str(group.iloc[0]['Title']),
            'abstract': str(group.iloc[0]['Abstract']),
            'items':    json.dumps(items, ensure_ascii=False)
        })

    outPath = Path(OUTPUT_PATH)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    taskDf = pd.DataFrame(tasks)
    taskDf.to_csv(str(outPath), index=False, encoding='utf-8-sig')

    logging.info(f"Preprocessing complete: {len(tasks)} tasks -> {outPath}")
    return taskDf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
    preprocess()
