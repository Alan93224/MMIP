"""Week4 Quiz3：FER 灰階臉部表情七分類（微調 ResNet vs. Vision Transformer）。

模組分五區：
  1. 資料處理：掃描 train / test 資料夾、train 以 8:2 分層切出 Train / Valid、
     影像一次讀進記憶體（48×48 灰階 uint8），資料增強與 Resize 在 GPU 上以整個 batch 進行
  2. 模型：torchvision 的 ImageNet 預訓練 ResNet-50 / ViT-B/16，替換分類層
  3. 訓練：AdamW（分類層較大 LR）、Linear Warmup + Cosine、AMP、Label Smoothing、
     依 Valid Macro-AUC 選最佳權重（Early Stopping），可從 models/ 直接載入已訓練好的權重
  4. 評估：Accuracy / Macro-Precision / Recall / F1 / Macro-AUC、各類別 AUC、
     ROC Curve、混淆矩陣、參數量與推論速度
  5. 視覺化：樣本影像、測試集預測結果
"""

import copy
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# ImageNet 的均值 / 標準差（預訓練模型需要相同的正規化）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def get_device():
    """有 GPU 就用 GPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """固定亂數種子，讓結果可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=False):
    """參數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ============================================================
# 1. 資料處理
# ============================================================
def list_image_files(data_dir, image_exts=IMAGE_EXTS):
    """掃描 data_dir/<類別>/<影像>，回傳 DataFrame(filepath, label)。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'找不到資料夾: {data_dir}')

    records = []
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(image_exts):
                records.append({'filepath': os.path.join(class_dir, fname), 'label': label})

    if not records:
        raise ValueError(f'{data_dir} 底下沒有找到任何影像檔')
    return pd.DataFrame(records)


def split_train_valid(df, valid_size=0.2, random_state=42):
    """以 label 分層抽樣切出 Train:Valid = 8:2，回傳 (train_df, valid_df)。"""
    train_df, valid_df = train_test_split(
        df, test_size=valid_size, stratify=df['label'], random_state=random_state,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def split_summary(splits, class_names):
    """splits = {'train': df, 'valid': df, 'test': df} → 各類別在各子集的張數表。"""
    summary = pd.DataFrame({name: d['label'].value_counts() for name, d in splits.items()})
    summary = summary.reindex(class_names).fillna(0).astype(int)
    summary['total'] = summary.sum(axis=1)
    summary.loc['total'] = summary.sum()
    return summary


def load_images(df, img_size=48, cache_path=None):
    """把 df 的影像全部讀成灰階 uint8 tensor (N, 1, img_size, img_size)。

    FER 影像只有 48×48，整個資料集約 60 MB，一次讀進記憶體可省去每個 epoch 重複讀檔。
    讀幾萬個小檔案很慢，給 cache_path 時會把結果存成 .npz，下次（檔案清單相同）直接讀取。
    """
    paths = np.array(df['filepath'].tolist())
    if cache_path and os.path.isfile(cache_path):
        cache = np.load(cache_path)
        if cache['images'].shape[-1] == img_size and np.array_equal(cache['paths'], paths):
            return torch.from_numpy(cache['images'])

    arr = np.empty((len(df), 1, img_size, img_size), dtype=np.uint8)
    for i, path in enumerate(paths):
        img = Image.open(path).convert('L')
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BILINEAR)
        arr[i, 0] = np.asarray(img)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path, images=arr, paths=paths)
    return torch.from_numpy(arr)


def build_dataloaders(images, targets, batch_size=64):
    """images / targets = {'train': tensor, 'valid': ..., 'test': ...} → 各自的 DataLoader。

    影像已在記憶體中，前處理放在 GPU 上做（gpu_preprocess），因此不需要多個 worker。
    """
    return {
        split: DataLoader(TensorDataset(images[split], targets[split]),
                          batch_size=batch_size, shuffle=(split == 'train'),
                          pin_memory=torch.cuda.is_available())
        for split in images
    }


def gpu_preprocess(x, img_size=224, train=False, max_rotate=10, max_translate=0.08,
                   scale_range=(0.9, 1.1), flip_p=0.5):
    """uint8 灰階 batch (B, 1, h, w) → 模型輸入 (B, 3, img_size, img_size)。

    train=True 時對每張影像各自做隨機資料增強（整個 batch 一次在 GPU 上完成）：
      - 水平翻轉（臉部表情左右對稱，翻轉不改變情緒）
      - 小角度旋轉、縮放、平移（affine_grid + grid_sample）
    接著 Resize 到預訓練模型的輸入大小、灰階複製成 3 通道、以 ImageNet 均值 / 標準差正規化。
    """
    x = x.float() / 255.0
    if train:
        b = x.size(0)
        angle = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(max_rotate)
        scale = torch.empty(b, device=x.device).uniform_(*scale_range)
        flip = torch.where(torch.rand(b, device=x.device) < flip_p, -1.0, 1.0)
        tx, ty = [(torch.rand(b, device=x.device) * 2 - 1) * max_translate * 2 for _ in range(2)]
        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([
            torch.stack([cos * flip, -sin, tx], dim=1),
            torch.stack([sin * flip, cos, ty], dim=1),
        ], dim=1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ============================================================
# 2. 模型
# ============================================================
def build_model(name='resnet50', num_classes=7, pretrained=True, dropout=0.2):
    """建立 ImageNet 預訓練模型並把最後的分類層換成 num_classes 類。

    - resnet18 / resnet50：替換 fc
    - vit_b_16：替換 heads.head（ViT-B/16：patch 16×16、12 層 Transformer Encoder、hidden 768）
    """
    if name in ('resnet18', 'resnet50'):
        weights = {'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                   'resnet50': models.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
        model = getattr(models, name)(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(model.heads.head.in_features, num_classes))
    else:
        raise ValueError(f'不支援的模型: {name}')
    return model


def head_parameters(model):
    """分類層的參數（與 backbone 分開，用較大的 learning rate）。"""
    head = model.fc if hasattr(model, 'fc') else model.heads
    return list(head.parameters())


# ============================================================
# 3. 訓練
# ============================================================
@torch.no_grad()
def predict_proba(model, loader, device, img_size=224, tta=False):
    """回傳 (softmax 機率 (N, C), 真實標籤 (N,))。tta=True 時再加上水平翻轉影像的預測取平均。"""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = gpu_preprocess(x.to(device, non_blocking=True), img_size, train=False)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            p = torch.softmax(model(x).float(), dim=1)
            if tta:
                p = (p + torch.softmax(model(x.flip(3)).float(), dim=1)) / 2
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def compute_metrics(y_true, prob):
    """Accuracy、Macro-Precision / Recall / F1 與 Macro-AUC（One-vs-Rest）。"""
    y_pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': p,
        'macro_recall': r,
        'macro_f1': f1,
        'macro_auc': roc_auc_score(y_true, prob, multi_class='ovr', average='macro',
                                   labels=list(range(prob.shape[1]))),
    }


def class_weights(targets, num_classes, power=0.5):
    """依類別數量的反比給權重（樣本少的類別權重大），平均為 1。

    power=1 為完全反比；FER 的 happy 與 disgusted 張數差約 16 倍，完全反比權重過於極端，
    預設取平方根（power=0.5）緩和。
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    w = (counts.sum() / (num_classes * counts)) ** power
    return w / w.mean()


def train_model(model, loaders, device, epochs=10, lr=1e-4, head_lr_mult=10.0, weight_decay=0.05,
                warmup_epochs=1, label_smoothing=0.1, weights=None, img_size=224, patience=3,
                save_path=None, resume=True, verbose=True):
    """微調預訓練模型，以 Valid Macro-AUC 選最佳權重，回傳 (最佳模型, history DataFrame)。

    - Optimizer：AdamW，分類層 LR = lr × head_lr_mult（新初始化的層需要學得較快）
    - LR：前 warmup_epochs 線性 warmup，之後 cosine 遞減到 0（每個 step 更新）
    - AMP 混合精度加速；weights 可傳入類別權重處理類別不平衡
    - resume=True 且 save_path 已存在：直接載入權重與 history，不重新訓練
    """
    hist_path = None if save_path is None else os.path.splitext(save_path)[0] + '_history.csv'
    if resume and save_path and os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
        history = pd.read_csv(hist_path) if os.path.isfile(hist_path) else pd.DataFrame()
        if verbose:
            print(f'載入已訓練的權重: {save_path}')
        return model.to(device), history

    model = model.to(device)
    head_ids = {id(p) for p in head_parameters(model)}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in head_ids], 'lr': lr},
        {'params': head_parameters(model), 'lr': lr * head_lr_mult},
    ], weight_decay=weight_decay)

    steps_per_epoch = len(loaders['train'])
    total_steps, warmup_steps = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device), label_smoothing=label_smoothing,
    )

    best_auc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loaders['train']:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = gpu_preprocess(x, img_size, train=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        valid_prob, valid_y = predict_proba(model, loaders['valid'], device, img_size)
        valid_loss = F.nll_loss(torch.tensor(valid_prob).clamp_min(1e-12).log(),
                                torch.tensor(valid_y, dtype=torch.long)).item()
        m = compute_metrics(valid_y, valid_prob)
        history.append({
            'epoch': epoch,
            'train_loss': total_loss / n,
            'valid_loss': valid_loss,
            'train_acc': correct / n,
            'valid_acc': m['accuracy'],
            'valid_f1': m['macro_f1'],
            'valid_auc': m['macro_auc'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': time.time() - t0,
        })
        if verbose:
            h = history[-1]
            print(f"epoch {epoch:2d} | loss {h['train_loss']:.4f}/{h['valid_loss']:.4f} | "
                  f"acc {h['train_acc']:.4f}/{h['valid_acc']:.4f} | valid F1 {h['valid_f1']:.4f} | "
                  f"valid AUC {h['valid_auc']:.4f} | {h['time']:.0f}s")

        if m['macro_auc'] > best_auc:
            best_auc, best_state, bad_epochs = m['macro_auc'], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}（最佳 Valid Macro-AUC = {best_auc:.4f}）')
                break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(best_state, save_path)
        history.to_csv(hist_path, index=False)
    return model, history


# ============================================================
# 4. 評估
# ============================================================
def compare_models(results):
    """results = {name: (prob, y_true)}（即 predict_proba 的回傳值）→ 各模型指標的比較表。"""
    rows = {name: compute_metrics(y, prob) for name, (prob, y) in results.items()}
    return pd.DataFrame(rows).T.round(4)


def per_class_auc(results, class_names):
    """各模型在每個類別的 One-vs-Rest AUC，最後一列為 Macro 平均。"""
    rows = {}
    for name, (prob, y) in results.items():
        y_bin = label_binarize(y, classes=range(len(class_names)))
        rows[name] = [roc_auc_score(y_bin[:, i], prob[:, i]) for i in range(len(class_names))]
    df = pd.DataFrame(rows, index=class_names)
    df.loc['macro'] = df.mean()
    return df.round(4)


def report(y_true, prob, class_names):
    """各類別的 Precision / Recall / F1 / Support。"""
    rep = classification_report(y_true, prob.argmax(1), target_names=class_names,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


@torch.no_grad()
def measure_inference_time(model, device, img_size=224, batch_size=64, n_iters=20):
    """平均每張影像的推論時間（ms），含 GPU 前處理，用 AMP。"""
    model.eval()
    x = torch.randint(0, 256, (batch_size, 1, 48, 48), dtype=torch.uint8, device=device)

    def step():
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            model(gpu_preprocess(x, img_size))

    for _ in range(3):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / (n_iters * batch_size) * 1000


def plot_histories(histories, figsize=(16, 4)):
    """histories = {name: history_df} → Loss / Accuracy / Valid Macro-AUC 訓練曲線。"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for (name, h), color in zip(histories.items(), plt.cm.tab10.colors):
        if h.empty:
            continue
        for ax, key in zip(axes[:2], ['loss', 'acc']):
            ax.plot(h['epoch'], h[f'train_{key}'], '--', color=color, label=f'{name} train')
            ax.plot(h['epoch'], h[f'valid_{key}'], '-', color=color, label=f'{name} valid')
        axes[2].plot(h['epoch'], h['valid_auc'], '-o', color=color, label=f'{name} valid')
    for ax, title in zip(axes, ['Loss', 'Accuracy', 'Valid Macro-AUC']):
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curves(results, class_names, figsize=None):
    """每個模型一張圖（各類別 OvR ROC + Macro-average ROC），最後一張疊加比較各模型的 Macro ROC。"""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=figsize or (5.5 * (n + 1), 5))
    grid = np.linspace(0, 1, 500)
    for ax_i, (name, (prob, y)) in enumerate(results.items()):
        ax = axes[ax_i]
        y_bin = label_binarize(y, classes=range(len(class_names)))
        mean_tpr = np.zeros_like(grid)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
            mean_tpr += np.interp(grid, fpr, tpr)
            ax.plot(fpr, tpr, lw=1, label=f'{cls} ({auc(fpr, tpr):.3f})')
        mean_tpr /= len(class_names)
        macro = roc_auc_score(y, prob, multi_class='ovr', average='macro')
        ax.plot(grid, mean_tpr, 'k-', lw=2.5, label=f'macro ({macro:.3f})')
        axes[-1].plot(grid, mean_tpr, lw=2.5, label=f'{name} ({macro:.3f})')
        ax.set_title(f'{name} ROC (One-vs-Rest)')

    axes[-1].set_title('Macro-average ROC')
    for ax in axes:
        ax.plot([0, 1], [0, 1], ':', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results, class_names, normalize=True, figsize=None):
    """並排畫出各模型的混淆矩陣；類別不平衡，預設以列（真實類別）正規化 = 各類別 Recall。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.5 * n, 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, (prob, y)) in zip(axes, results.items()):
        cm = confusion_matrix(y, prob.argmax(1), labels=range(len(class_names)),
                              normalize='true' if normalize else None)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='.2f' if normalize else 'd',
            xticks_rotation=45,
        )
        ax.set_title(f'{name} (acc={accuracy_score(y, prob.argmax(1)):.3f})')
    plt.tight_layout()
    plt.show()


def plot_metric_bars(compare_df, metrics=('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_auc'),
                     title='Test metrics', figsize=(10, 4)):
    """把 compare_models 的比較表畫成長條圖。"""
    ax = compare_df[list(metrics)].T.plot(kind='bar', figsize=figsize, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 視覺化
# ============================================================
def show_samples(images, targets, class_names, n_per_class=6, seed=0):
    """每個類別隨機顯示 n_per_class 張影像。"""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    fig, axes = plt.subplots(len(class_names), n_per_class,
                             figsize=(1.5 * n_per_class, 1.6 * len(class_names)))
    for i, cls in enumerate(class_names):
        idx = rng.choice(np.where(targets == i)[0], n_per_class, replace=False)
        for j, k in enumerate(idx):
            ax = axes[i, j]
            ax.imshow(images[k, 0], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(cls, fontsize=10)
    plt.tight_layout()
    plt.show()

def show_predictions(images, results, class_names, n=12, seed=0):
    """隨機挑 n 張測試影像，標出真實類別與各模型的預測（綠 = 正確、紅 = 錯誤）。"""
    rng = np.random.default_rng(seed)
    y = next(iter(results.values()))[1]
    idx = rng.choice(len(y), n, replace=False)
    cols = 6
    rows = math.ceil(n / cols)
    
    # 1. 將每列的高度倍率從 3.0 調大至 3.6，給文字更多縱向空間
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.6 * rows))
    
    for ax, k in zip(axes.flat, idx):
        ax.imshow(images[k, 0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        lines = [f'true: {class_names[y[k]]}']
        colors = ['black']
        for name, (prob, _) in results.items():
            p = prob[k].argmax()
            lines.append(f'{name}: {class_names[p]} ({prob[k, p]:.2f})')
            colors.append('green' if p == y[k] else 'red')
        for i, (line, c) in enumerate(zip(lines, colors)):
            ax.text(0.5, -0.08 - 0.14 * i, line, color=c, fontsize=8,
                    ha='center', va='top', transform=ax.transAxes)
                    
    for ax in list(axes.flat)[len(idx):]:
        ax.axis('off')
        
    # 2. 增加 h_pad，強制讓上下列子圖之間拉開垂直距離
    plt.tight_layout(h_pad=4.0)
    plt.show()