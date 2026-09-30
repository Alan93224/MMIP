"""Week3 Quiz2：Animals-10 影像分類（Plain CNN vs. 經典 CNN Backbone）。

模組分五區：
  1. 資料處理：由 Quiz1 的 split_df 建立 Dataset / DataLoader（含資料增強）
  2. 模型：自行設計的 Plain CNN、torchvision 經典 Backbone（ResNet 等）
  3. 訓練與評估：訓練迴圈（AMP、Cosine LR、Early Stopping）、Top-1 / Top-5 Accuracy、
     5-Fold 交叉驗證（每折一個模型，測試集以各折模型平均機率做 Ensemble）
  4. 預測結果整理：測試集預測表（Top-5 類別 + 機率）、各類別報告、混淆矩陣
  5. 模型比較：各類別 ROC Curve、Macro-AUC、Accuracy 與參數量比較
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .quiz1 import get_fold_split

# ImageNet 的均值 / 標準差（預訓練 Backbone 需要相同的正規化）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    """有 GPU 就用 GPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """固定亂數種子，讓結果可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 資料處理
# ============================================================
class AnimalDataset(Dataset):
    """讀取 DataFrame(filepath, label) 的影像資料集。"""

    def __init__(self, df, class_to_idx, transform=None):
        self.paths = df['filepath'].tolist()
        self.targets = [class_to_idx[c] for c in df['label']]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 部分圖檔是灰階 / CMYK / 調色盤模式，統一轉成 RGB
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def build_transforms(img_size=224):
    """回傳 (train_tf, eval_tf)。

    Train：隨機裁切縮放 + 水平翻轉 + 色彩抖動，增加資料多樣性
    Eval ：縮放後中心裁切，確保評估結果固定
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(split_df, img_size=224, batch_size=64, num_workers=4):
    """依 split 欄位建立 train / val / test 的 DataLoader。

    split_df 的 split 欄位需為 'train' / 'val' / 'test'
    （5-Fold 的 split_df 請先用 quiz1.get_fold_split 取出某一折）。
    回傳 (loaders: dict, class_names: list)。

    注意（Windows）：每個 worker 都是獨立的 Python 行程，各自載入 torch 約佔 1.5 GB 虛擬記憶體。
    train / val 每個 epoch 都會用到，設 persistent_workers 省去重開行程的時間；
    test 只在最後用一次，不保留 worker。用完請呼叫 shutdown_loaders 釋放。
    """
    class_names = sorted(split_df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train_tf, eval_tf = build_transforms(img_size)

    loaders = {}
    for split in ('train', 'val', 'test'):
        sub = split_df[split_df['split'] == split].reset_index(drop=True)
        ds = AnimalDataset(sub, class_to_idx,
                           transform=train_tf if split == 'train' else eval_tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0 and split != 'test',
        )
    return loaders, class_names


def shutdown_loaders(loaders):
    """立即關閉 DataLoader 的 persistent worker 行程（不等 Python 垃圾回收）。

    loaders 可以是單一 DataLoader 或 dict。
    """
    if isinstance(loaders, DataLoader):
        loaders = {'_': loaders}
    for loader in loaders.values():
        it = getattr(loader, '_iterator', None)
        if it is not None and hasattr(it, '_shutdown_workers'):
            it._shutdown_workers()      # pylint: disable=protected-access
        loader._iterator = None         # pylint: disable=protected-access


# ============================================================
# 2. 模型
# ============================================================
def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlainCNN(nn.Module):
    """自行設計的 Plain CNN（VGG 風格、無殘差連接）。

    結構：5 個 stage，每個 stage = 2 × (Conv3x3-BN-ReLU) + MaxPool，
    通道數 32→64→128→256→256，空間尺寸每個 stage 減半（224 → 7）。
    最後以 Global Average Pooling 取代大型全連接層，大幅減少參數並降低過擬合。
    """

    def __init__(self, num_classes=10, channels=(32, 64, 128, 256, 256), dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers += [conv_bn_relu(in_ch, out_ch),
                       conv_bn_relu(out_ch, out_ch),
                       nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_backbone(name='resnet50', num_classes=10, pretrained=True):
    """建立 torchvision 經典 CNN Backbone，並把最後分類層換成 num_classes 類。

    支援：resnet18 / resnet50 / vgg16_bn / densenet121 / efficientnet_b0 / mobilenet_v3_large
    """
    weights = 'DEFAULT' if pretrained else None
    kwargs = {}
    if name.startswith('resnet') and not pretrained:
        # 從頭訓練時，把每個殘差區塊最後一層 BN 的 gamma 初始化為 0，
        # 讓區塊一開始近似恆等映射，訓練初期更穩定（torchvision 內建選項）
        kwargs['zero_init_residual'] = True
    model = getattr(models, name)(weights=weights, **kwargs)

    if name.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name.startswith(('vgg', 'efficientnet', 'mobilenet')):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    else:
        raise ValueError(f'不支援的 backbone: {name}')
    return model


def count_parameters(model, trainable_only=False):
    """計算參數量。"""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


# ============================================================
# 3. 訓練與評估
# ============================================================
def topk_correct(logits, targets, ks=(1, 5)):
    """回傳各 k 的預測正確數（Top-k：真實類別落在機率最高的 k 個類別內即算對）。"""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)                 # (N, maxk)
    hit = pred.eq(targets.view(-1, 1))                 # (N, maxk)
    return {k: hit[:, :k].any(dim=1).sum().item() for k in ks}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, ks=(1, 5)):
    """跑一個 epoch；有給 optimizer 就是訓練模式，否則為評估模式。

    回傳 dict(loss, top1, top5)。
    """
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    correct = {k: 0 for k in ks}
    use_amp = device.type == 'cuda'

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
            for k, c in topk_correct(logits.float(), y, ks).items():
                correct[k] += c

    out = {'loss': total_loss / n}
    out.update({f'top{k}': correct[k] / n for k in ks})
    return out


def train_model(model, loaders, epochs=20, lr=1e-3, weight_decay=1e-4,
                label_smoothing=0.1, patience=5, device=None, save_path=None,
                verbose=True):
    """訓練模型並以 Val Top-1 選最佳權重。

    - Optimizer：AdamW；LR Scheduler：Cosine Annealing
    - AMP 混合精度加速（GPU）
    - Early Stopping：Val Top-1 連續 patience 個 epoch 沒進步就停止

    回傳 (best_model, history_df)。
    """
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    best_acc, best_state, wait = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders['train'], criterion, device, optimizer, scaler)
        va = run_epoch(model, loaders['val'], criterion, device)
        scheduler.step()

        history.append({'epoch': epoch,
                        'train_loss': tr['loss'], 'train_top1': tr['top1'], 'train_top5': tr['top5'],
                        'val_loss': va['loss'], 'val_top1': va['top1'], 'val_top5': va['top5']})

        improved = va['top1'] > best_acc
        if improved:
            best_acc, wait = va['top1'], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if verbose:
            print(f'[{epoch:02d}/{epochs}] '
                  f'train loss {tr["loss"]:.4f} top1 {tr["top1"]:.4f} | '
                  f'val loss {va["loss"]:.4f} top1 {va["top1"]:.4f} top5 {va["top5"]:.4f} | '
                  f'{time.time() - t0:.0f}s' + (' *' if improved else ''))

        if wait >= patience:
            if verbose:
                print(f'Early stopping：Val Top-1 已 {patience} 個 epoch 未進步')
            break

    model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(best_state, save_path)
    if verbose:
        print(f'最佳 Val Top-1 = {best_acc:.4f}')
    return model, pd.DataFrame(history)


def evaluate_topk(model, loaders, device=None, splits=('train', 'val', 'test')):
    """計算各 split 的 Loss、Top-1 / Top-5 Accuracy，回傳 DataFrame。"""
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    rows = {}
    for split in splits:
        rows[split] = run_epoch(model, loaders[split], criterion, device)
    return pd.DataFrame(rows).T.rename(columns={'top1': 'Top-1 Acc', 'top5': 'Top-5 Acc'})


def plot_history(history, title=''):
    """畫 Loss 與 Top-1 Accuracy 的訓練曲線。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['epoch'], history['train_loss'], label='train')
    axes[0].plot(history['epoch'], history['val_loss'], label='val')
    axes[0].set_title(f'{title} Loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    axes[1].plot(history['epoch'], history['train_top1'], label='train top-1')
    axes[1].plot(history['epoch'], history['val_top1'], label='val top-1')
    axes[1].plot(history['epoch'], history['val_top5'], label='val top-5', linestyle='--')
    axes[1].set_title(f'{title} Accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def cross_validate(model_fn, split_df, name='model', folds=None, img_size=224,
                   batch_size=64, num_workers=4, device=None, save_dir=None,
                   resume=False, **train_kwargs):
    """K-Fold 交叉驗證：每一折都用 model_fn() 建新模型，以該折當 val 訓練。

    - model_fn：無參數、回傳新模型的函式，例如 lambda: PlainCNN(10)
    - folds   ：要跑哪幾折（預設全部），時間不夠可先跑部分折，例如 [0, 1]
    - save_dir：每折最佳權重存成 {name}_fold{k}.pt、訓練紀錄存成 {name}_fold{k}_history.csv
    - resume  ：True 時若該折權重已存在就直接載入、跳過訓練（中斷後可接著跑）
    - train_kwargs 直接傳給 train_model（epochs、lr、patience…）

    回傳 dict：
      models          各折最佳模型（放在 CPU）
      histories       各折訓練紀錄（resume 載入但找不到紀錄檔時為 None）
      metrics         各折 train / val / test 的 Loss、Top-1、Top-5（DataFrame）
                      train 取自最佳 epoch 的訓練紀錄（含資料增強），val / test 為重新評估
      test_probs      各折模型在測試集的機率 list[(N, C)]
      ensemble_probs  各折模型平均後的測試集機率 (N, C)
      y_test、test_paths、class_names
    """
    device = device or get_device()
    n_splits = int(split_df['fold'].max()) + 1
    folds = range(n_splits) if folds is None else folds

    out = {'models': [], 'histories': [], 'test_probs': [], 'metrics': []}
    for k in folds:
        print(f'========== {name} | Fold {k + 1}/{n_splits} ==========')
        fold_df = get_fold_split(split_df, k)
        loaders, class_names = build_dataloaders(fold_df, img_size=img_size,
                                                 batch_size=batch_size, num_workers=num_workers)
        save_path = hist_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{name}_fold{k}.pt')
            hist_path = os.path.join(save_dir, f'{name}_fold{k}_history.csv')

        try:
            if resume and save_path is not None and os.path.isfile(save_path):
                print(f'載入已存在的權重: {save_path}')
                model = model_fn()
                model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
                hist = pd.read_csv(hist_path) if os.path.isfile(hist_path) else None
            else:
                model, hist = train_model(model_fn(), loaders, device=device,
                                          save_path=save_path, **train_kwargs)
                if hist_path is not None:
                    hist.to_csv(hist_path, index=False)
            # 訓練結束就先關掉 train 的 worker，釋放記憶體
            shutdown_loaders({'train': loaders['train']})

            print('評估 val / test ...')
            metrics = evaluate_topk(model, loaders, device=device, splits=('val', 'test'))
            if hist is not None:
                best = hist.loc[hist['val_top1'].idxmax()]
                metrics.loc['train'] = {'loss': best['train_loss'],
                                        'Top-1 Acc': best['train_top1'],
                                        'Top-5 Acc': best['train_top5']}
            metrics.insert(0, 'fold', k)
            probs, y_test = predict_proba(model, loaders['test'], device=device)
        finally:
            shutdown_loaders(loaders)

        print(metrics.drop(columns='fold').round(4).to_string())
        out['models'].append(model.cpu())
        out['histories'].append(None if hist is None else hist.assign(fold=k))
        out['metrics'].append(metrics.rename_axis('split').reset_index())
        out['test_probs'].append(probs)
        del loaders
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out['metrics'] = pd.concat(out['metrics'], ignore_index=True)
    out['ensemble_probs'] = np.mean(out['test_probs'], axis=0)
    out['y_test'] = y_test
    out['test_paths'] = fold_df.loc[fold_df['split'] == 'test', 'filepath'].tolist()
    out['class_names'] = class_names
    return out


def cv_summary(cv_result):
    """整理交叉驗證結果：各折 Top-1 / Top-5，外加 mean ± std 與 Ensemble 的測試集表現。"""
    m = cv_result['metrics']
    table = m.pivot(index='fold', columns='split', values=['Top-1 Acc', 'Top-5 Acc'])
    table = table.reindex(columns=['train', 'val', 'test'], level=1)
    table.index = [f'fold{k}' for k in table.index]

    stats = pd.DataFrame({'mean': table.mean(), 'std': table.std()}).T
    table = pd.concat([table, stats])

    # 各折模型平均機率後的 Ensemble 只在 test 上有意義
    probs, y = cv_result['ensemble_probs'], cv_result['y_test']
    top5 = np.argsort(-probs, axis=1)[:, :5]
    table.loc['ensemble'] = np.nan
    table.loc['ensemble', ('Top-1 Acc', 'test')] = (top5[:, 0] == y).mean()
    table.loc['ensemble', ('Top-5 Acc', 'test')] = (top5 == y[:, None]).any(axis=1).mean()
    return table.round(4)


def plot_cv_history(histories, title=''):
    """把各折的 Loss 與 Val Top-1 / Top-5 曲線疊在一起畫。"""
    _, axes = plt.subplots(1, 3, figsize=(16, 4))
    for h in histories:
        if h is None:           # resume 載入、沒有訓練紀錄的折
            continue
        k = int(h['fold'].iloc[0])
        line, = axes[0].plot(h['epoch'], h['train_loss'], label=f'fold{k} train')
        axes[0].plot(h['epoch'], h['val_loss'], linestyle='--', color=line.get_color(),
                     label=f'fold{k} val')
        axes[1].plot(h['epoch'], h['val_top1'], label=f'fold{k}')
        axes[2].plot(h['epoch'], h['val_top5'], label=f'fold{k}')
    names = ['Loss (solid=train, dashed=val)', 'Val Top-1 Acc', 'Val Top-5 Acc']
    for ax, name in zip(axes, names):
        ax.set_title(f'{title} {name}')
        ax.set_xlabel('epoch')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 預測結果整理
# ============================================================
@torch.no_grad()
def predict_proba(model, loader, device=None):
    """回傳 (probs: (N, C) ndarray, y_true: (N,) ndarray)。"""
    device = device or get_device()
    model = model.to(device).eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def build_prediction_df(paths, probs, y_true, class_names, topk=5):
    """把測試集預測整理成表格。

    欄位：檔名、真實類別、模型給真實類別的機率、預測類別與信心度、
    top1 ~ top5（格式「類別 (機率)」）、Top-1 是否正確、是否落在 Top-5 內。
    """
    order = np.argsort(-probs, axis=1)[:, :topk]
    names = np.array(class_names)
    rows = np.arange(len(y_true))
    pred = order[:, 0]
    df = pd.DataFrame({
        'filepath': paths,
        'true_label': names[y_true],
        'true_prob': probs[rows, y_true].round(4),
        'pred_label': names[pred],
        'confidence': probs[rows, pred].round(4),
    })
    for i in range(topk):
        idx = order[:, i]
        df[f'top{i + 1}'] = [f'{names[c]} ({p:.4f})' for c, p in zip(idx, probs[rows, idx])]
    df['correct'] = pred == y_true
    df[f'in_top{topk}'] = (order == y_true[:, None]).any(axis=1)
    return df


def per_class_report(y_true, probs, class_names):
    """各類別 Precision / Recall / F1 / Support。"""
    report = classification_report(y_true, probs.argmax(axis=1),
                                   target_names=class_names, output_dict=True, digits=4)
    return pd.DataFrame(report).T.round(4)


def plot_confusion_matrix(y_true, probs, class_names, title='', normalize=True):
    """畫混淆矩陣（預設以真實類別做列正規化 = 各類別 Recall）。"""
    cm = confusion_matrix(y_true, probs.argmax(axis=1))
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else int(cm[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} Confusion Matrix')
    plt.tight_layout()
    plt.show()


def show_topk_probs(paths, probs, y_true, class_names, indices, k=5):
    """逐張顯示影像與 Top-k 類別機率長條圖（綠色 = 真實類別）。"""
    names = np.array(class_names)
    _, axes = plt.subplots(len(indices), 2, figsize=(9, 2.6 * len(indices)),
                           gridspec_kw={'width_ratios': [1, 1.6]}, squeeze=False)
    for (ax_img, ax_bar), i in zip(axes, indices):
        order = np.argsort(-probs[i])[:k]
        ax_img.imshow(Image.open(paths[i]).convert('RGB'))
        ax_img.set_title(f'True: {names[y_true[i]]}', fontsize=9)
        ax_img.axis('off')

        colors = ['tab:green' if c == y_true[i] else 'tab:gray' for c in order]
        bars = ax_bar.barh(names[order][::-1], probs[i, order][::-1], color=colors[::-1])
        ax_bar.bar_label(bars, fmt='%.4f', fontsize=8)
        ax_bar.set_xlim(0, 1.15)
        ax_bar.set_title(f'Top-{k} probabilities', fontsize=9)
    plt.tight_layout()
    plt.show()


def show_predictions(pred_df, n=12, only_wrong=False, cols=6, random_state=42):
    """隨機展示 n 張測試影像與其預測結果（綠色=正確、紅色=錯誤）。"""
    df = pred_df[~pred_df['correct']] if only_wrong else pred_df
    df = df.sample(min(n, len(df)), random_state=random_state)
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, r in zip(np.atleast_1d(axes).ravel(), df.itertuples()):
        ax.imshow(Image.open(r.filepath).convert('RGB'))
        ax.set_title(f'T: {r.true_label}\nP: {r.pred_label} ({r.confidence:.2f})',
                     fontsize=9, color='green' if r.correct else 'red')
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 模型比較：ROC / Macro-AUC / Accuracy / 參數量
# ============================================================
def plot_multiclass_roc(y_true, probs, class_names, title='', ax=None):
    """One-vs-Rest 畫各類別 ROC Curve，並回傳 (各類別 AUC Series, Macro-AUC)。"""
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    aucs = {}
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        aucs[name] = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f'{name} (AUC={aucs[name]:.4f})')

    macro_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} ROC (Macro-AUC={macro_auc:.4f})')
    ax.legend(loc='lower right', fontsize=8)
    return pd.Series(aucs, name=title), macro_auc


def compare_models(results):
    """整理模型比較表。

    results: {模型名稱: dict(model=..., probs=..., y_true=...)}
    回傳欄位：Top-1 / Top-5 Acc、Macro-AUC、參數量（M）、模型大小（MB, float32）
    """
    rows = []
    for name, r in results.items():
        probs, y_true = r['probs'], r['y_true']
        top5 = np.argsort(-probs, axis=1)[:, :5]
        n_params = count_parameters(r['model'])
        rows.append({
            'model': name,
            'Top-1 Acc': (top5[:, 0] == y_true).mean(),
            'Top-5 Acc': (top5 == y_true[:, None]).any(axis=1).mean(),
            'Macro-AUC': roc_auc_score(y_true, probs, average='macro', multi_class='ovr'),
            'Params (M)': n_params / 1e6,
            'Size (MB)': n_params * 4 / 1024 ** 2,
        })
    return pd.DataFrame(rows).set_index('model').round(4)


def plot_model_comparison(compare_df):
    """左：Top-1 / Top-5 / Macro-AUC 長條圖；右：參數量長條圖。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics = compare_df[['Top-1 Acc', 'Top-5 Acc', 'Macro-AUC']]
    metrics.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_ylim(max(0.0, metrics.values.min() - 0.1), 1.0)
    axes[0].set_title('Test Performance')
    axes[0].legend(loc='lower right')
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt='%.3f', fontsize=8)

    compare_df['Params (M)'].plot(kind='bar', ax=axes[1], rot=0, color='gray')
    axes[1].set_title('Parameters (M)')
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt='%.2f', fontsize=8)
    plt.tight_layout()
    plt.show()
