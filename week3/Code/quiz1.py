"""Week3 Quiz1：把 data/Dataset 的十類影像切出 10% Test，其餘 90% 做 5-Fold 交叉驗證。"""

import os
import shutil

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# 允許的影像副檔名
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def list_image_files(data_dir, image_exts=IMAGE_EXTS):
    """掃描 data_dir 底下每個類別資料夾，回傳 DataFrame(filepath, label)。

    預期結構：
        data_dir/
            cat/xxx.jpg
            dog/yyy.png
            ...
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'找不到資料夾: {data_dir}')

    records = []
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(image_exts):
                records.append({'filepath': os.path.join(class_dir, fname),
                                'label': label})

    if not records:
        raise ValueError(f'{data_dir} 底下沒有找到任何影像檔')

    return pd.DataFrame(records)


def split_dataset(data_dir,
                  test_size=0.1,
                  n_splits=5,
                  random_state=42,
                  stratify=True,
                  image_exts=IMAGE_EXTS):
    """先切出 Test，再對剩下的資料做 K-Fold，回傳多了 split / fold 兩欄的 DataFrame。

    - split：'train'（交叉驗證用的資料池）或 'test'（獨立測試集，不參與訓練）
    - fold ：train 資料所屬的 fold 編號 0 ~ n_splits-1；test 資料為 -1

    預設 test_size=0.1、n_splits=5 → 每個 fold 的 Train:Val:Test = 72:18:10 ≈ 7:2:1。
    切分時都以 label 做分層抽樣（stratify），確保每個類別在各子集中比例一致。
    """
    df = list_image_files(data_dir, image_exts=image_exts)

    strat = df['label'] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat,
    )

    train_df = train_df.assign(split='train', fold=-1)
    test_df = test_df.assign(split='test', fold=-1)

    kf_cls = StratifiedKFold if stratify else KFold
    kf = kf_cls(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_col = train_df.columns.get_loc('fold')
    for k, (_, val_idx) in enumerate(kf.split(train_df, train_df['label'])):
        train_df.iloc[val_idx, fold_col] = k

    return (pd.concat([train_df, test_df])
              .sort_values(['split', 'fold', 'label'])
              .reset_index(drop=True))


def get_fold_split(split_df, fold):
    """取出第 fold 折：該折當 val、其餘折當 train，test 不變。

    回傳的 DataFrame 其 split 欄位為 'train' / 'val' / 'test'，可直接拿去建 DataLoader。
    """
    n_splits = split_df['fold'].max() + 1
    if not 0 <= fold < n_splits:
        raise ValueError(f'fold 必須介於 0 ~ {n_splits - 1}，目前為 {fold}')

    df = split_df.copy()
    is_val = (df['split'] == 'train') & (df['fold'] == fold)
    df.loc[is_val, 'split'] = 'val'
    return df


def split_summary(split_df):
    """回傳各類別在每個 fold 與 test 的張數統計表（含總計）。"""
    group = split_df['fold'].map(lambda k: f'fold{k}')
    group = group.where(split_df['split'] != 'test', 'test')
    table = pd.crosstab(split_df['label'], group)
    # 固定欄位順序：fold0 ~ foldK、test
    cols = sorted(c for c in table.columns if c.startswith('fold')) + ['test']
    table = table[cols]
    table['total'] = table.sum(axis=1)
    table.loc['total'] = table.sum(axis=0)
    return table


def copy_split_to_dirs(split_df, output_dir, move=False, overwrite=False):
    """依照 split 欄位把檔案複製（或搬移）成 output_dir/{split}/{label}/ 結構。

    若要輸出某一折的 train / val，先用 get_fold_split(split_df, fold) 再傳進來。
    """
    if os.path.isdir(output_dir):
        if not overwrite:
            raise FileExistsError(f'{output_dir} 已存在，若要覆蓋請設 overwrite=True')
        shutil.rmtree(output_dir)

    for row in split_df.itertuples(index=False):
        dst_dir = os.path.join(output_dir, row.split, row.label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(row.filepath))
        if move:
            shutil.move(row.filepath, dst)
        else:
            shutil.copy2(row.filepath, dst)

    return output_dir
