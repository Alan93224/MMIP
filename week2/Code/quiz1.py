"""Week2 Quiz1：貓狗影像特徵工程 + 門檻(Threshold)調校與評估工具箱。

模組分三區：
  1. 特徵工程：讀圖 -> 抽取數值特徵 -> 組成特徵矩陣 X 與標籤 y (cat=0, dog=1)
  2. 評估模組：指定門檻評估、自動搜尋最佳門檻（以 F1 最大化為目標）
  3. 視覺化：ROC 曲線、PR 曲線、雙門檻混淆矩陣對照圖
"""

import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# 類別對應：cat -> 0 (Negative)，dog -> 1 (Positive)
CLASS_MAP = {'cat': 0, 'dog': 1}
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# 特徵欄位名稱（順序必須與 extract_features 回傳的向量一致）
FEATURE_NAMES = [
    'R_mean', 'G_mean', 'B_mean',
    'R_std', 'G_std', 'B_std',
    'H_mean', 'S_mean', 'V_mean',
    'H_std', 'S_std', 'V_std',
    'gray_mean', 'gray_std',
    'width', 'height', 'aspect_ratio',
    'edge_density',
]


# ============================================================
# 1. 特徵工程
# ============================================================
def extract_features(image_path, resize=(128, 128)):
    """讀取單張影像並回傳 1D 特徵向量；讀取失敗回傳 None。

    共 18 個數值特徵：RGB 均值/標準差、HSV 均值/標準差、灰階統計、
    原始尺寸與長寬比、Canny 邊緣密度。
    """
    # 用 imdecode 讀檔，避免 Windows 中文或特殊字元路徑讀不到
    raw = np.fromfile(image_path, dtype=np.uint8)
    if raw.size == 0:
        return None
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:                       # 損毀或非影像檔
        return None

    h, w = img_bgr.shape[:2]                  # 先記錄「原始」尺寸資訊
    aspect_ratio = w / h if h else 0.0

    # 統一縮放，讓顏色與紋理統計量彼此可比較，也加快計算
    img_bgr = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 邊緣密度：邊緣像素佔比，粗略描述紋理複雜度
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    feats = np.concatenate([
        img_rgb.mean(axis=(0, 1)),            # R/G/B 均值
        img_rgb.std(axis=(0, 1)),             # R/G/B 標準差
        img_hsv.mean(axis=(0, 1)),            # H/S/V 均值
        img_hsv.std(axis=(0, 1)),             # H/S/V 標準差
        [gray.mean(), gray.std()],            # 灰階均值/標準差
        [w, h, aspect_ratio],                 # 原始寬、高、長寬比
        [edge_density],
    ]).astype(np.float32)
    return feats


def list_images(class_dir, max_per_class=None, seed=42):
    """列出資料夾中的影像路徑；max_per_class 可隨機抽樣以控制資料量。"""
    files = sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(IMG_EXTS)
    )
    if max_per_class is not None and len(files) > max_per_class:
        rng = np.random.default_rng(seed)     # 固定亂數種子，抽樣結果可重現
        idx = rng.choice(len(files), size=max_per_class, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def build_dataset(data_dir, max_per_class=None, resize=(128, 128), seed=42, verbose=True):
    """掃描 data_dir/cat 與 data_dir/dog，回傳 (X, y, paths)。

    X: (n_samples, 18) float32 特徵矩陣
    y: (n_samples,) int 標籤，cat=0 / dog=1
    """
    X, y, paths, n_failed = [], [], [], 0

    for cls_name, label in CLASS_MAP.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f'找不到類別資料夾: {os.path.abspath(cls_dir)}')

        files = list_images(cls_dir, max_per_class=max_per_class, seed=seed)
        for i, fp in enumerate(files, 1):
            feats = extract_features(fp, resize=resize)
            if feats is None:                 # 跳過損毀檔案
                n_failed += 1
                continue
            X.append(feats)
            y.append(label)
            paths.append(fp)
            if verbose and i % 500 == 0:
                print(f'  [{cls_name}] 已處理 {i}/{len(files)} 張')

        if verbose:
            print(f'[{cls_name}] 完成，標籤={label}，有效樣本數={y.count(label)}')

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if verbose:
        print(f'\n資料集完成：X={X.shape}, y={y.shape}, 讀取失敗 {n_failed} 張')
    return X, y, paths


def to_dataframe(X, y):
    """把特徵矩陣包成 DataFrame，方便 describe() 檢視分布。"""
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    return df


# ============================================================
# 2. 評估模組
# ============================================================
def evaluate_threshold(y_true, y_probs, threshold=0.5, title='', verbose=True):
    """以指定門檻把機率轉成預測，印出混淆矩陣與四大指標，回傳 dict。"""
    y_pred = (np.asarray(y_probs) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'confusion_matrix': cm,
    }

    if verbose:
        head = f'[{title}] ' if title else ''
        print(f'{head}Threshold = {threshold:.4f}')
        print('Confusion Matrix (row=真實, col=預測, 0=cat, 1=dog):')
        print('            pred_cat  pred_dog')
        print(f'  true_cat  {tn:8d}  {fp:8d}')
        print(f'  true_dog  {fn:8d}  {tp:8d}')
        print(f'  Accuracy : {metrics["accuracy"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall   : {metrics["recall"]:.4f}')
        print(f'  F1-Score : {metrics["f1"]:.4f}\n')
    return metrics


def find_best_threshold(y_true, y_probs, n_steps=1001, metric='f1', verbose=True):
    """在 [0,1] 上掃描門檻，回傳使指定指標（預設 F1）最大的 (best_th, best_score, table)。"""
    scorers = ['f1', 'accuracy', 'precision', 'recall']
    if metric not in scorers:
        raise ValueError(f'metric 需為 {scorers} 之一，收到 {metric}')

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    thresholds = np.linspace(0.0, 1.0, n_steps)

    rows = []
    for th in thresholds:
        y_pred = (y_probs >= th).astype(int)
        rows.append({
            'threshold': th,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        })
    table = pd.DataFrame(rows)

    best_idx = int(table[metric].idxmax())    # 同分時取最小的門檻
    best_th = float(table.loc[best_idx, 'threshold'])
    best_score = float(table.loc[best_idx, metric])

    if verbose:
        print(f'最佳門檻搜尋（目標={metric}，掃描 {n_steps} 點）')
        print(f'  best threshold = {best_th:.4f} | best {metric} = {best_score:.4f}')
        print(f'  該門檻下 Acc={table.loc[best_idx, "accuracy"]:.4f}, '
              f'P={table.loc[best_idx, "precision"]:.4f}, '
              f'R={table.loc[best_idx, "recall"]:.4f}\n')
    return best_th, best_score, table


def compare_thresholds(y_true, y_probs, th_a=0.5, th_b=None,
                       name_a='Default 0.5', name_b='Tuned'):
    """同時評估兩組門檻，並以 DataFrame 並排比較。"""
    ma = evaluate_threshold(y_true, y_probs, th_a, title=name_a)
    mb = evaluate_threshold(y_true, y_probs, th_b, title=name_b)
    cols = ['threshold', 'accuracy', 'precision', 'recall', 'f1']
    return pd.DataFrame(
        [{c: ma[c] for c in cols}, {c: mb[c] for c in cols}],
        index=[name_a, name_b],
    )


# ============================================================
# 3. 視覺化
# ============================================================
def plot_roc_curve(y_true, y_probs, title='ROC Curve', mark_threshold=None, ax=None):
    """繪製 ROC 曲線並標註 AUC；mark_threshold 會在曲線上標出該門檻位置。"""
    fpr, tpr, ths = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    if mark_threshold is not None:
        i = int(np.argmin(np.abs(ths - mark_threshold)))
        ax.scatter(fpr[i], tpr[i], c='red', zorder=5,
                   label=f'th = {mark_threshold:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    return roc_auc


def plot_pr_curve(y_true, y_probs, title='Precision-Recall Curve', mark_threshold=None, ax=None):
    """繪製 PR 曲線並標註 Average Precision。"""
    precision, recall, ths = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    baseline = float(np.mean(y_true))         # 隨機猜測時的 precision 基準線

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f'PR (AP = {ap:.4f})')
    ax.axhline(baseline, ls='--', c='k', lw=1, label=f'Baseline = {baseline:.3f}')

    if mark_threshold is not None and len(ths):
        i = int(np.argmin(np.abs(ths - mark_threshold)))
        ax.scatter(recall[i], precision[i], c='red', zorder=5,
                   label=f'th = {mark_threshold:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    return ap


def plot_confusion_matrix(cm, ax, title, class_names=('cat(0)', 'dog(1)')):
    """在指定 ax 上畫單一混淆矩陣（含數值標註）。"""
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'pred {c}' for c in class_names])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'true {c}' for c in class_names])
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            # 依格子實際底色亮度決定字色：淺底用深色字，只有很深的底才用白字
            red, green, blue, _ = im.cmap(im.norm(cm[r, c]))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(c, r, f'{cm[r, c]:d}', ha='center', va='center',
                    color='white' if luminance < 0.45 else '#111111',
                    fontsize=13, fontweight='bold')
    ax.set_title(title)
    return im


def plot_dual_confusion_matrix(y_true, y_probs, th_a=0.5, th_b=None,
                               name_a='Default 0.5', name_b='Tuned', figsize=(10, 4.5)):
    """左右並排比較兩個門檻的混淆矩陣。"""
    ma = evaluate_threshold(y_true, y_probs, th_a, verbose=False)
    mb = evaluate_threshold(y_true, y_probs, th_b, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_confusion_matrix(ma['confusion_matrix'], axes[0],
                          f'{name_a} (th={th_a:.3f})\nF1={ma["f1"]:.4f}')
    plot_confusion_matrix(mb['confusion_matrix'], axes[1],
                          f'{name_b} (th={th_b:.3f})\nF1={mb["f1"]:.4f}')
    fig.tight_layout()
    plt.show()
    return ma, mb


def plot_threshold_curves(table, best_th=None, title='Metrics vs. Threshold'):
    """把 find_best_threshold 的掃描表畫成指標隨門檻變化的曲線。"""
    _, ax = plt.subplots(figsize=(7, 5))
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        ax.plot(table['threshold'], table[col], lw=1.8, label=col)
    if best_th is not None:
        ax.axvline(best_th, ls='--', c='red', lw=1.2, label=f'best th = {best_th:.3f}')
    ax.axvline(0.5, ls=':', c='gray', lw=1.2, label='default th = 0.5')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.show()


def plot_evaluation_curves(y_true, y_probs, best_th=None, suptitle='Validation'):
    """一次畫出 ROC 與 PR 曲線，回傳 (auc, ap)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    roc_auc = plot_roc_curve(y_true, y_probs, f'{suptitle} - ROC', best_th, axes[0])
    ap = plot_pr_curve(y_true, y_probs, f'{suptitle} - PR', best_th, axes[1])
    fig.tight_layout()
    plt.show()
    return roc_auc, ap


# ============================================================
# 4. 多模型比較
# ============================================================
def compare_models(y_true, prob_dict, threshold_dict=None, verbose=True):
    """在同一份資料上比較多個模型。

    prob_dict:      {模型名稱: 正類機率}
    threshold_dict: {模型名稱: 門檻}，未指定者一律用 0.5
    回傳含 Accuracy/Precision/Recall/F1 與 FP/FN 的 DataFrame。
    """
    threshold_dict = threshold_dict or {}
    rows = []
    for name, probs in prob_dict.items():
        th = threshold_dict.get(name, 0.5)
        m = evaluate_threshold(y_true, probs, th, verbose=False)
        rows.append({
            'model': name, 'threshold': m['threshold'],
            'accuracy': m['accuracy'], 'precision': m['precision'],
            'recall': m['recall'], 'f1': m['f1'],
            'FP(cat誤判成dog)': m['fp'], 'FN(dog誤判成cat)': m['fn'],
        })
    table = pd.DataFrame(rows).set_index('model')
    if verbose:
        print(table.round(4).to_string())
    return table


def plot_model_comparison(y_true, prob_dict, threshold_dict=None, figsize=(13, 5)):
    """左圖疊合各模型 ROC 曲線，右圖用長條圖比較四大指標。"""
    threshold_dict = threshold_dict or {}
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for name, probs in prob_dict.items():                 # 左：ROC 疊圖
        fpr, tpr, _ = roc_curve(y_true, probs)
        axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve Comparison')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    table = compare_models(y_true, prob_dict, threshold_dict, verbose=False)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):     # 右：指標長條圖
        offset = (i - (len(table) - 1) / 2) * width
        bars = axes[1].bar(x + offset, [row[m] for m in metrics], width,
                           label=f'{name} (th={row["threshold"]:.3f})')
        axes[1].bar_label(bars, fmt='%.3f', fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics Comparison (same validation set)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    plt.show()
    return table


def plot_error_breakdown(y_true, prob_dict, threshold_dict=None, figsize=(7, 5)):
    """比較各模型的兩類錯誤數量：FP(cat→dog) 與 FN(dog→cat)。"""
    table = compare_models(y_true, prob_dict, threshold_dict, verbose=False)
    names = list(table.index)
    fp_vals = table['FP(cat誤判成dog)'].values
    fn_vals = table['FN(dog誤判成cat)'].values

    x = np.arange(len(names))
    _, ax = plt.subplots(figsize=figsize)
    b1 = ax.bar(x - 0.2, fp_vals, 0.4, label='FP: cat -> dog')
    b2 = ax.bar(x + 0.2, fn_vals, 0.4, label='FN: dog -> cat')
    ax.bar_label(b1, fontsize=9)
    ax.bar_label(b2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(th={table.loc[n, "threshold"]:.3f})' for n in names])
    ax.set_ylabel('Error count')
    ax.set_title('Error Type Breakdown')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.show()

    for n in names:                                        # 文字結論：哪一類錯誤較多
        fp_n, fn_n = int(table.loc[n, 'FP(cat誤判成dog)']), int(table.loc[n, 'FN(dog誤判成cat)'])
        worse = 'FP (把 cat 判成 dog)' if fp_n > fn_n else 'FN (把 dog 判成 cat)'
        print(f'{n:<20s} FP={fp_n:4d}, FN={fn_n:4d} -> 較多的錯誤是 {worse}')
    return table
