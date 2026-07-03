"""
LLL 資料集前處理腳本。
將 LLL-test.csv 轉為標準 Task CSV 格式（PPI），供 Pipeline 使用。

LLL 原始欄位：
  - docid, isValid, passage, passageid

標準 Task CSV 欄位（輸出，PPI 格式）：
  - taskID:  唯一識別碼，使用 passageid + row index
  - passage: 單句文本（對應 taskTemplate 的 {passage} 佔位符）
  - label:   true label 字串（對應 config.labelColumn，由 Pipeline 自動包成 items）

使用方式：
  python Preprocess/lll.py
"""

import logging
import pandas as pd
from pathlib import Path

INPUT_PATH  = "data\PPI\IEPA-test.csv"
OUTPUT_PATH = "data\PPI\IEPA\IEPA_test.csv"


def preprocess():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    required = {'isValid', 'passage', 'passageid'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tasks = []
    for i, row in df.iterrows():
        # gold label 須與 config.labelSet.classes (["no", "yes"]) 完全對齊
        label = "yes" if str(row['isValid']).strip().upper() == "TRUE" else "no"
        tasks.append({
            "taskID":  str(row['passageid']) + f"_{i}",
            "passage": str(row['passage']),
            "label":   label,
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
