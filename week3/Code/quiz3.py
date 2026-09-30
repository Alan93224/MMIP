"""Week3 Quiz3：資料增強前後比較、Kernel 視覺化、Grad-CAM 可解釋性分析。

模組分四區（訓練流程直接沿用 quiz2_improved.cross_validate，這裡只放分析用的函式）：
  1. 模型載入：從 models/ 讀回已訓練好的各折權重，不必重跑交叉驗證
  2. 資料增強：展示擴增後的影像、增強前後的指標與各類別 Recall 比較
  3. Kernel 視覺化：第一層卷積核的權重、統計量（亮度/色彩、方向性）與在影像上的響應
  4. XAI：Grad-CAM，看模型做預測時關注影像的哪些區域
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2
from PIL import Image
from sklearn.metrics import recall_score, roc_auc_score

from .quiz2 import IMAGENET_MEAN, IMAGENET_STD, PlainCNN, build_transforms, get_device
from .quiz2_improved import build_strong_train_transform


# ============================================================
# 1. 模型載入
# ============================================================
def load_fold_model(model_fn, name, fold=0, model_dir='models'):
    """讀取 {model_dir}/{name}_fold{fold}.pt，回傳 eval 模式的模型（CPU）。"""
    path = os.path.join(model_dir, f'{name}_fold{fold}.pt')
    model = model_fn()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def load_image(path, img_size=224):
    """回傳 (評估用的正規化 tensor (1, 3, H, W), 顯示用的 RGB ndarray (H, W, 3)，範圍 0~1)。

    與 val / test 相同的 eval transform（Resize → CenterCrop），熱力圖才能與原圖對齊。
    """
    _, eval_tf = build_transforms(img_size)
    x = eval_tf(Image.open(path).convert('RGB')).unsqueeze(0)
    return x, denormalize(x[0])


def denormalize(x):
    """把正規化後的 (3, H, W) tensor 轉回 0~1 的 (H, W, 3) ndarray。"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ============================================================
# 2. 資料增強
# ============================================================
def show_augmentations(path, img_size=224, n=7, strong=True, seed=0):
    """同一張影像套用 n 次隨機增強（第一格為 eval transform 的原圖），直觀看出擴增的變化幅度。"""
    torch.manual_seed(seed)
    train_tf, eval_tf = build_transforms(img_size)
    if strong:
        train_tf = build_strong_train_transform(img_size)
    img = Image.open(path).convert('RGB')

    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    axes[0].imshow(denormalize(eval_tf(img)))
    axes[0].set_title('original', fontsize=9)
    for ax in axes[1:]:
        ax.imshow(denormalize(train_tf(img)))
    for ax in axes:
        ax.axis('off')
    fig.suptitle('strong augmentation' if strong else 'basic augmentation', fontsize=10)
    plt.tight_layout()
    plt.show()


def show_mix(path_a, path_b, label_a, label_b, class_names, img_size=224, lam=0.6, seed=0):
    """把兩張影像分別做 MixUp 與 CutMix，並標出混合後的 soft label。

    MixUp 以固定的 lam 示範（訓練時 lam ~ Beta(0.2, 0.2)，大多接近 0 或 1，隨機抽到的效果不明顯）；
    CutMix 直接用訓練時的 v2.CutMix(alpha=1.0)，soft label 依貼上區域的面積比例計算。
    """
    torch.manual_seed(seed)
    _, eval_tf = build_transforms(img_size)
    x = torch.stack([eval_tf(Image.open(p).convert('RGB')) for p in (path_a, path_b)])
    idx = {c: i for i, c in enumerate(class_names)}
    y = torch.tensor([idx[label_a], idx[label_b]])

    xc, yc = v2.CutMix(alpha=1.0, num_classes=len(class_names))(x, y)
    panels = [('A', x[0], {label_a: 1.0}),
              ('B', x[1], {label_b: 1.0}),
              (f'MixUp (lam={lam})', lam * x[0] + (1 - lam) * x[1],
               {label_a: lam, label_b: 1 - lam}),
              ('CutMix', xc[0],
               {class_names[c]: yc[0, c].item() for c in yc[0].nonzero().flatten().tolist()})]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 3.2))
    for ax, (name, img, soft) in zip(axes, panels):
        ax.imshow(denormalize(img))
        ax.set_title(name + '\n' + ', '.join(f'{k} {v:.2f}' for k, v in soft.items()), fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def aug_summary(results, y_true):
    """比較增強前後的整體指標。

    results: {模型名稱: dict(cv=cross_validate 的回傳值, train_top1=可選)}
      - train_top1 需為「無增強」評估的 train Top-1；v2 之後的 cv['metrics'] 本來就是，
        原版（quiz2.cross_validate）的 train 含增強，請傳入 clean_train_gap 算出的值
    回傳欄位：train / val / test Top-1（5 折平均）、train-val 落差、Ensemble test Top-1、Macro-AUC、
    以及答錯樣本的平均信心度（越低代表錯的時候越不武斷）。
    """
    rows = {}
    for name, r in results.items():
        cv = r['cv']
        t = cv['metrics'].pivot(index='fold', columns='split', values='Top-1 Acc').mean()
        train = r.get('train_top1', t['train'])
        probs = cv['ensemble_probs']
        pred = probs.argmax(axis=1)
        wrong = pred != y_true
        rows[name] = {
            'train Top-1': train,
            'val Top-1': t['val'],
            'gap (train-val)': train - t['val'],
            'test Top-1 (mean)': t['test'],
            'test Top-1 (ensemble)': (~wrong).mean(),
            'Macro-AUC': roc_auc_score(y_true, probs, multi_class='ovr', average='macro'),
            'wrong conf (mean)': probs[wrong].max(axis=1).mean(),
        }
    return pd.DataFrame(rows).T.round(4)


def plot_recall_compare(y_true, probs_dict, class_names, title=''):
    """各類別 Recall 的並排長條圖，回傳 Recall 表（最後一列附上 Macro）。"""
    df = pd.DataFrame({name: recall_score(y_true, p.argmax(axis=1), average=None)
                       for name, p in probs_dict.items()}, index=class_names)
    ax = df.plot(kind='bar', figsize=(11, 3.8), width=0.8)
    lo = max(0.0, df.values.min() - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel('recall')
    ax.set_title(title or 'Per-class recall')
    ax.legend(loc='lower right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    df.loc['Macro'] = df.mean()
    return df.round(4)


# ============================================================
# 3. Kernel 視覺化
# ============================================================
def first_conv(model):
    """回傳模型的第一個 Conv2d（直接作用在 RGB 像素上，最容易解讀）。"""
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d))


def _to_rgb(w):
    """(3, k, k) 權重 → 以 0 為中心、線性縮放到 0~1 的 (k, k, 3) 影像。

    0 對應灰色 0.5，正值偏亮、負值偏暗；每個 kernel 各自縮放，只看形狀與相對大小。
    """
    w = w.permute(1, 2, 0).numpy()
    return 0.5 + 0.5 * w / (np.abs(w).max() + 1e-8)


def plot_kernels(model, title='', cols=16, highlight=()):
    """畫出第一層所有卷積核（RGB 顯示），highlight 內的編號以紅框標示。"""
    w = first_conv(model).weight.detach().cpu()
    rows = int(np.ceil(len(w) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.9))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis('off')
        if i >= len(w):
            continue
        ax.imshow(_to_rgb(w[i]), interpolation='nearest')
        ax.set_title(str(i), fontsize=7, color='red' if i in highlight else 'black')
        if i in highlight:
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('red')
                s.set_linewidth(2)
    k = w.shape[-1]
    fig.suptitle(f'{title} first conv: {len(w)} kernels, {k}x{k}x3', fontsize=10)
    plt.tight_layout()
    plt.show()


def kernel_stats(model):
    """第一層每個 kernel 的統計量，用來挑選要分析的 kernel：

    - norm      ：權重 L2 norm，越大代表這個 kernel 的輸出越強
    - color     ：色彩成分比例 = ||w - 通道平均|| / ||w||；
                  接近 0 → 三通道權重相同，只看亮度（灰階邊緣 / 紋理）；越大 → 對顏色差異敏感
    - orient    ：亮度成分的主要梯度方向（度，0 = 左右亮暗變化 → 偵測垂直邊緣，90 = 上下變化 → 水平邊緣）
    - coherence ：方向一致性 0~1（structure tensor），越接近 1 越像單一方向的邊緣偵測器，
                  低則代表斑點 / 中心-周圍 / 紋理型
    - dc        ：權重總和 / ||w||，接近 0 表示對均勻區域無反應（純邊緣），大則代表會響應整體亮度或顏色
    """
    w = first_conv(model).weight.detach().cpu().numpy()         # (C, 3, k, k)
    rows = []
    for i, wi in enumerate(w):
        norm = np.linalg.norm(wi) + 1e-8
        gray = wi.mean(axis=0)
        gy, gx = np.gradient(gray)
        jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
        coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-8)
        orient = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) % 180
        rows.append({'kernel': i,
                     'norm': norm,
                     'color': np.linalg.norm(wi - gray) / norm,
                     'orient': orient,
                     'coherence': coherence,
                     'dc': wi.sum() / norm})
    return pd.DataFrame(rows).set_index('kernel').round(3)


def show_kernel_response(model, kernel_ids, paths, img_size=224, title=''):
    """每個 kernel 一列：RGB 權重、R / G / B 三通道權重、以及在各張影像上的響應（feature map）。

    響應直接用該 kernel 對正規化影像做卷積（BN / ReLU 之前），
    紅色 = 正響應（與 kernel 圖樣相符），藍色 = 負響應（相反圖樣）。
    """
    conv = first_conv(model)
    w = conv.weight.detach().cpu()
    imgs = [load_image(p, img_size) for p in paths]
    ncols = 4 + len(paths)
    fig, axes = plt.subplots(len(kernel_ids) + 1, ncols,
                             figsize=(2.1 * ncols, 2.2 * (len(kernel_ids) + 1)))
    for ax in axes.ravel():
        ax.axis('off')
    for j, (_, rgb) in enumerate(imgs):
        axes[0, 4 + j].imshow(rgb)
        axes[0, 4 + j].set_title(os.path.basename(os.path.dirname(paths[j])), fontsize=9)

    for r, k in enumerate(kernel_ids, start=1):
        wk = w[k]
        axes[r, 0].imshow(_to_rgb(wk), interpolation='nearest')
        axes[r, 0].set_title(f'kernel {k} (RGB)', fontsize=9)
        vmax = wk.abs().max().item()
        for c, name in enumerate('RGB'):
            ax = axes[r, 1 + c]
            ax.imshow(wk[c].numpy(), cmap='bwr', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_title(f'{name} weights', fontsize=9)
            if wk.shape[-1] <= 3:
                for (yy, xx), v in np.ndenumerate(wk[c].numpy()):
                    ax.text(xx, yy, f'{v:.2f}', ha='center', va='center', fontsize=7)
        for j, (x, _) in enumerate(imgs):
            with torch.no_grad():
                fmap = F.conv2d(x, wk[None], stride=conv.stride, padding=conv.padding)[0, 0].numpy()
            m = np.abs(fmap).max() + 1e-8
            axes[r, 4 + j].imshow(fmap, cmap='bwr', vmin=-m, vmax=m)
            axes[r, 4 + j].set_title(f'response k{k}', fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    plt.show()


# ============================================================
# 4. XAI：Grad-CAM
# ============================================================
def gradcam_layer(model):
    """預設的 Grad-CAM 目標層：最後一個卷積 stage 的輸出（空間資訊仍在、語意最高）。

    - ResNet   ：layer4（224 輸入 → 7×7）
    - PlainCNN ：最後一個 Conv-BN-ReLU（MaxPool 之前 → 14×14，解析度比 ResNet 高一倍）
    """
    if hasattr(model, 'layer4'):
        return model.layer4
    if isinstance(model, PlainCNN):
        return model.features[-2]
    raise ValueError('請自行指定 target_layer')


class GradCAM:
    """Grad-CAM（Selvaraju et al., 2017）。

    對目標類別的 logit 求目標層 feature map 的梯度，
    以「梯度的空間平均」當作各通道的權重，加權總和後過 ReLU，得到該類別的關注熱力圖。
    """

    def __init__(self, model, target_layer=None, device=None):
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        layer = target_layer if target_layer is not None else gradcam_layer(model)
        self.acts = self.grads = None
        self._hook = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, 'grads', g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (1, 3, H, W)。回傳 (cam: (H, W) 0~1, probs: (C,), class_idx)；class_idx=None 用預測類別。"""
        x = x.to(self.device)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
            if class_idx is None:
                class_idx = int(probs.argmax())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
        return cam, probs, class_idx

    def remove(self):
        self._hook.remove()


def _overlay(ax, rgb, cam, title, color='black', alpha=0.45):
    ax.imshow(rgb)
    ax.imshow(cam, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, color=color)
    ax.axis('off')


def show_gradcam(models, paths, true_labels, class_names, img_size=224, device=None):
    """每張影像一列：原圖 + 各模型對「自己預測類別」的 Grad-CAM（綠字 = 預測正確、紅字 = 錯誤）。

    models: {名稱: 模型}
    """
    cams = {name: GradCAM(m, device=device) for name, m in models.items()}
    ncols = 1 + len(cams)
    fig, axes = plt.subplots(len(paths), ncols, figsize=(2.8 * ncols, 2.9 * len(paths)),
                             squeeze=False)
    try:
        for row, path, label in zip(axes, paths, true_labels):
            x, rgb = load_image(path, img_size)
            row[0].imshow(rgb)
            row[0].set_title(f'True: {label}', fontsize=9)
            row[0].axis('off')
            for ax, (name, cam_fn) in zip(row[1:], cams.items()):
                cam, probs, c = cam_fn(x)
                pred = class_names[c]
                _overlay(ax, rgb, cam, f'{name}\nP: {pred} ({probs[c]:.2f})',
                         color='green' if pred == label else 'red')
    finally:
        for cam_fn in cams.values():
            cam_fn.remove()
    plt.tight_layout()
    plt.show()


def show_gradcam_true_vs_pred(model, paths, true_labels, class_names, img_size=224,
                              device=None, name=''):
    """針對預測錯誤的樣本：同一個模型分別對「預測類別」與「真實類別」做 Grad-CAM，

    比較模型把錯誤類別的證據放在哪裡、正確類別的證據又被忽略在哪裡。
    """
    cam_fn = GradCAM(model, device=device)
    idx = {c: i for i, c in enumerate(class_names)}
    fig, axes = plt.subplots(len(paths), 3, figsize=(8.4, 2.9 * len(paths)), squeeze=False)
    try:
        for row, path, label in zip(axes, paths, true_labels):
            x, rgb = load_image(path, img_size)
            cam_p, probs, c = cam_fn(x)
            cam_t, _, _ = cam_fn(x, class_idx=idx[label])
            row[0].imshow(rgb)
            row[0].set_title(f'True: {label}', fontsize=9)
            row[0].axis('off')
            _overlay(row[1], rgb, cam_p, f'pred: {class_names[c]} ({probs[c]:.2f})',
                     color='green' if class_names[c] == label else 'red')
            _overlay(row[2], rgb, cam_t, f'true: {label} ({probs[idx[label]]:.2f})')
    finally:
        cam_fn.remove()
    if name:
        fig.suptitle(name, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.98) if name else None)
    plt.show()
