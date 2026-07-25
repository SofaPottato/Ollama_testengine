"""IEPA 資料集前處理：把 IEPA-{split}.csv 轉成標準 Dataset CSV（PPI 格式）供 Pipeline 使用。

IEPA 原始欄位：docid, isValid, passage, passageid
輸出 Dataset CSV（PPI 格式）：
  - taskID : 唯一識別碼（passageid + row index）
  - passage: 單句文本（對應 taskTemplate 的 {passage} 佔位符）
  - label  : true label 字串（對應 Main 的 labelColumn，由 Pipeline 自動包成 items）

使用方式：python DatasetPreprocess/iepa.py（train / test 一次轉完）
"""
import logging
import pandas as pd
from pathlib import Path

# 路徑一律以專案根目錄為基準組出來，不隨執行時的 cwd 變動而失效。
_ROOT = Path(__file__).resolve().parent.parent

DATASET_NAME = "IEPA"                                    # data/PPIDataset 下五套資料集的欄位格式相同，改這個常數即可轉 AIMed / BioInfer / HPRD50 / LLL
INPUT_DIR    = _ROOT / "data" / "PPIDataset"             # 原始攤平 CSV：{DATASET_NAME}-{split}.csv
OUTPUT_DIR   = INPUT_DIR / DATASET_NAME                  # 轉檔輸出：{DATASET_NAME}/{DATASET_NAME}_{split}.csv
SPLIT_LIST   = ["train", "test"]


def preprocess(inputPath: Path, outputPath: Path) -> pd.DataFrame:
    df = pd.read_csv(inputPath, encoding='utf-8-sig')

    required = {'isValid', 'passage', 'passageid'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{inputPath.name} missing columns: {missing}")

    tasks = []
    for i, row in df.iterrows():
        # gold label 須與 labelSet.classes (['no','yes']) 對齊：isValid==TRUE → yes。
        label = "yes" if str(row['isValid']).strip().upper() == "TRUE" else "no"
        tasks.append({
            "taskID":  str(row['passageid']) + f"_{i}",
            "passage": str(row['passage']),
            "label":   label,
        })

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    taskDf = pd.DataFrame(tasks)
    taskDf.to_csv(str(outputPath), index=False, encoding='utf-8-sig')

    logging.info(f"Preprocessing complete: {len(tasks)} tasks -> {outputPath}")
    return taskDf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
    # 每個 split 各轉一次；缺哪個原始檔就在該檔的 read_csv 直接報錯，不靜默略過。
    for splitName in SPLIT_LIST:
        preprocess(
            INPUT_DIR / f"{DATASET_NAME}-{splitName}.csv",
            OUTPUT_DIR / f"{DATASET_NAME}_{splitName}.csv",
        )
