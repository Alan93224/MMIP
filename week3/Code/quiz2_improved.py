"""Week3 Quiz2 Improved：針對原版 Plain CNN（underfitting）與 ResNet-18（overfitting）的改良版訓練流程。

不修改 quiz2.py，只在這裡新增改良需要的部分，其餘（Dataset、評估、畫圖）直接沿用 quiz2：
  1. 資料增強：新增較強的 train transform（TrivialAugmentWide + RandomErasing + 較大裁切範圍）
  2. 模型：ResNet 分類層前加 Dropout；Plain CNN 直接用 quiz2.PlainCNN 調整 channels / dropout
  3. 訓練：Linear Warmup + Cosine LR、可選 MixUp / CutMix
  4. 交叉驗證：train 的 Top-1 / Top-5 改用「不做資料增強」重新評估，才能公平衡量 train-val 落差
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import v2

from .quiz1 import get_fold_split
from .quiz2 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AnimalDataset,
    build_transforms,
    evaluate_topk,
    get_device,
    predict_proba,
    shutdown_loaders,
    topk_correct,
)


# ============================================================
# 1. 資料處理
# ============================================================
def build_strong_train_transform(img_size=224):
    """較強的資料增強（對抗 overfitting）。

    - RandomResizedCrop scale 0.6 → 0.35：裁切範圍更大，模型要看局部特徵也能判斷
    - TrivialAugmentWide：每張圖隨機套一種幾何 / 色彩變換（旋轉、剪切、對比、曝光…），不需調參
    - RandomErasing：隨機遮住一塊區域，迫使模型不要只依賴單一部位
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_dataloaders(split_df, img_size=224, batch_size=64, num_workers=4, strong_aug=False):
    """同 quiz2.build_dataloaders，另外多兩點：

    - strong_aug=True 時 train 使用 build_strong_train_transform
    - 多一個 'train_eval'：train 資料但用 eval transform，用來量「沒有增強」時的 train 表現
    """
    class_names = sorted(split_df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train_tf, eval_tf = build_transforms(img_size)
    if strong_aug:
        train_tf = build_strong_train_transform(img_size)

    specs = {'train': ('train', train_tf, True),
             'train_eval': ('train', eval_tf, False),
             'val': ('val', eval_tf, False),
             'test': ('test', eval_tf, False)}
    loaders = {}
    for key, (split, tf, shuffle) in specs.items():
        sub = split_df[split_df['split'] == split].reset_index(drop=True)
        loaders[key] = DataLoader(
            AnimalDataset(sub, class_to_idx, transform=tf),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            # train / val 每個 epoch 都用 → 保留 worker；train_eval / test 只在最後用一次
            persistent_workers=num_workers > 0 and key in ('train', 'val'),
        )
    return loaders, class_names


# ============================================================
# 2. 模型
# ============================================================
def build_resnet(name='resnet18', num_classes=10, dropout=0.2):
    """從頭訓練的 torchvision ResNet，分類層改成 Dropout → Linear（參數量不變）。"""
    model = getattr(models, name)(weights=None, zero_init_residual=True)
    model.fc = nn.Sequential(nn.Dropout(dropout),
                             nn.Linear(model.fc.in_features, num_classes))
    return model


# ============================================================
# 3. 訓練
# ============================================================
class RandomMixUpCutMix:
    """以機率 p 對整個 batch 套用 MixUp 或 CutMix（兩者各半），否則維持原樣。

    回傳 (x, soft_targets)；soft_targets 是 (N, C) 的機率分佈，可直接丟進 CrossEntropyLoss。
    """

    def __init__(self, num_classes, p=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
        self.num_classes = num_classes
        self.p = p
        self.mix = v2.RandomChoice([v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
                                    v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)])

    def __call__(self, x, y):
        if np.random.rand() < self.p:
            return self.mix(x, y)
        return x, nn.functional.one_hot(y, self.num_classes).float()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              mix_fn=None, ks=(1, 5)):
    """同 quiz2.run_epoch，另外支援 mix_fn（MixUp / CutMix，只在訓練時使用）。

    有 mix 時 Top-k 仍以原始 hard label 計算，所以 train acc 會偏低，僅供看趨勢。
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
            target = y
            if training and mix_fn is not None:
                x, target = mix_fn(x, y)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, target)

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


def train_model(model, loaders, epochs=40, lr=1e-3, weight_decay=5e-4,
                label_smoothing=0.1, patience=10, warmup_epochs=2,
                mix_prob=0.0, device=None, save_path=None, verbose=True):
    """同 quiz2.train_model，改良處：

    - LR：前 warmup_epochs 個 epoch 從 0.1×lr 線性升到 lr，之後 Cosine 衰減到 0
      （epoch 拉長後，一開始用較穩定的步伐，避免從頭訓練時前幾個 epoch 震盪）
    - mix_prob > 0 時，訓練 batch 以該機率套用 MixUp / CutMix
    """
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                               total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine],
                                                      milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    mix_fn = None
    if mix_prob > 0:
        num_classes = len(set(loaders['train'].dataset.targets))
        mix_fn = RandomMixUpCutMix(num_classes, p=mix_prob)

    best_acc, best_state, wait = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        tr = run_epoch(model, loaders['train'], criterion, device, optimizer, scaler, mix_fn)
        va = run_epoch(model, loaders['val'], criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'lr': lr_now,
                        'train_loss': tr['loss'], 'train_top1': tr['top1'], 'train_top5': tr['top5'],
                        'val_loss': va['loss'], 'val_top1': va['top1'], 'val_top5': va['top5']})

        improved = va['top1'] > best_acc
        if improved:
            best_acc, wait = va['top1'], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if verbose:
            print(f'[{epoch:02d}/{epochs}] lr {lr_now:.2e} | '
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


# ============================================================
# 4. 交叉驗證
# ============================================================
def cross_validate(model_fn, split_df, name='model', folds=None, img_size=224,
                   batch_size=64, num_workers=4, device=None, save_dir=None,
                   resume=False, strong_aug=False, **train_kwargs):
    """同 quiz2.cross_validate（回傳格式相同，可直接用 quiz2 的 cv_summary / plot_cv_history），
    差別在 metrics 的 train 列：改成最佳模型在「無資料增強」train 上重新評估的結果，
    與 val / test 的評估方式一致，train-val 落差才代表真正的 overfitting 程度。
    """
    device = device or get_device()
    n_splits = int(split_df['fold'].max()) + 1
    folds = range(n_splits) if folds is None else folds

    out = {'models': [], 'histories': [], 'test_probs': [], 'metrics': []}
    for k in folds:
        print(f'========== {name} | Fold {k + 1}/{n_splits} ==========')
        fold_df = get_fold_split(split_df, k)
        loaders, class_names = build_dataloaders(fold_df, img_size=img_size, batch_size=batch_size,
                                                 num_workers=num_workers, strong_aug=strong_aug)
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
            shutdown_loaders({'train': loaders['train']})

            print('評估 train(無增強) / val / test ...')
            metrics = evaluate_topk(model, loaders, device=device,
                                    splits=('train_eval', 'val', 'test'))
            metrics = metrics.rename(index={'train_eval': 'train'})
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


# ============================================================
# 5. 原版 vs. Improved 比較
# ============================================================
def clean_train_gap(cv_result, split_df, img_size=224, batch_size=64, num_workers=4, device=None):
    """用「無資料增強」重新評估 cv_result 各折模型在 train / val 的 Top-1，回傳各折與平均的落差。

    原版 quiz2.cross_validate 的 train acc 取自訓練紀錄（有增強），與 val 不在同一基準；
    這裡統一用 eval transform 評估，原版與 Improved 的 overfitting 程度才能直接比較。
    """
    device = device or get_device()
    rows = []
    for model, k in zip(cv_result['models'], cv_result['metrics']['fold'].unique()):
        fold_df = get_fold_split(split_df, int(k))
        loaders, _ = build_dataloaders(fold_df, img_size=img_size, batch_size=batch_size,
                                       num_workers=num_workers)
        try:
            m = evaluate_topk(model, loaders, device=device, splits=('train_eval', 'val'))
        finally:
            shutdown_loaders(loaders)
        model.cpu()
        rows.append({'fold': f'fold{int(k)}',
                     'train Top-1': m.loc['train_eval', 'Top-1 Acc'],
                     'val Top-1': m.loc['val', 'Top-1 Acc']})
    df = pd.DataFrame(rows).set_index('fold')
    df['gap'] = df['train Top-1'] - df['val Top-1']
    df.loc['mean'] = df.mean()
    return df.round(4)
