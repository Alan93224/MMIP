"""Week2 Quiz2：UCI 信用卡違約預測（PyTorch MLP）。

模組分四區：
  1. 資料處理：讀 CSV、清理類別欄、7:2:1 切分、Feature Scaling
  2. 模型：可設定隱藏層的 Multi-Layer Perceptron
  3. 訓練與評估：訓練迴圈（記錄 train/val loss）、Stratified 5-Fold CV、預測
  4. 視覺化：Loss 曲線、單筆樣本預測展示
"""

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = 'default.payment.next.month'       # 1 = 下個月違約, 0 = 正常
# 類別型欄位（數值只是代碼，沒有大小關係）→ 做 one-hot
CATEGORICAL_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']


# ============================================================
# 1. 資料處理
# ============================================================
def load_data(csv_path, verbose=True):
    """讀取 UCI_Credit_Card.csv，回傳原始 DataFrame。"""
    df = pd.read_csv(csv_path)
    if verbose:
        print(f'資料筆數: {len(df)}, 欄位數: {df.shape[1]}')
        print(f'缺失值總數: {int(df.isna().sum().sum())}')
        counts = df[TARGET_COL].value_counts().sort_index()
        print(f'標籤分布: 0(正常)={counts.get(0, 0)}, 1(違約)={counts.get(1, 0)} '
              f'(違約率 {df[TARGET_COL].mean():.2%})')
    return df


def preprocess(df, log_amount=True, verbose=True):
    """清理資料並回傳 (X, y, feature_names)。

    處理內容：
      - 丟掉 ID（純流水號，沒有預測力）
      - EDUCATION 的 0/5/6 與 MARRIAGE 的 0 併入「其他」類
      - LIMIT_BAL / BILL_AMT / PAY_AMT 做對稱對數壓縮（可關閉）：金額分布極度
        右偏，少數百萬級大戶會讓標準化後仍留下 70 倍的極端值，壓縮後接近常態
      - SEX / EDUCATION / MARRIAGE 做 one-hot（避免模型誤把代碼當成數值大小）
    """
    df = df.copy()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # 官方文件只定義 EDUCATION 1~4、MARRIAGE 1~3，其餘代碼統一歸到「其他」
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    if log_amount:
        # sign(x) * log1p(|x|)：保留負數帳單（溢繳）的方向，同時壓掉長尾
        amount_cols = [c for c in df.columns
                       if c.startswith(('BILL_AMT', 'PAY_AMT')) or c == 'LIMIT_BAL']
        for col in amount_cols:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    y = df[TARGET_COL].to_numpy(dtype=np.int64)
    X_df = df.drop(columns=[TARGET_COL])
    X_df = pd.get_dummies(X_df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)
    if verbose:
        print(f'前處理後特徵矩陣: {X.shape}（one-hot 後共 {len(feature_names)} 個特徵）'
              f'{" | 金額欄已做 log 壓縮" if log_amount else ""}')
    return X, y, feature_names


def split_data(X, y, random_state=42, verbose=True):
    """7:2:1 切成 Train / Validation / Test，全程 stratify 保持違約比例一致。"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=random_state)
    # 從剩下的 90% 取 20/90，最終比例才是 70:20:10
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20 / 0.90, stratify=y_temp, random_state=random_state)

    if verbose:
        total = len(y)
        for name, yy in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            print(f'{name:<5s}: {len(yy):6d} 筆 ({len(yy)/total:.1%}) | 違約率 {yy.mean():.2%}')
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, verbose=True):
    """Feature Scaling：只用訓練集 fit StandardScaler，避免資料外洩。

    LIMIT_BAL、BILL_AMT 動輒數十萬，AGE 只有兩位數，不縮放會讓大數值欄位
    主導梯度，神經網路難以收斂。
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    if verbose:
        print('縮放前 訓練集各欄位數值範圍:'
              f' min={X_train.min():.1f}, max={X_train.max():.1f}')
        print('縮放後 訓練集各欄位數值範圍:'
              f' min={X_train_s.min():.2f}, max={X_train_s.max():.2f}'
              f' | mean={X_train_s.mean():.4f}, std={X_train_s.std():.4f}')
    return X_train_s, X_val_s, X_test_s, scaler


# ============================================================
# 2. 模型
# ============================================================
class MLP(nn.Module):
    """多層感知器：Linear -> (BatchNorm) -> ReLU -> Dropout，堆疊數層後輸出 1 個 logit。

    use_batchnorm=True 會讓模型在 1~2 個 epoch 內就衝到最低點，loss 曲線看不到
    下降過程；表格資料量不大時建議關閉，改用較小的學習率慢慢收斂。
    """

    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3, use_batchnorm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))          # 輸出 logit，搭配 BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed=42):
    """固定 numpy 與 torch 的亂數種子，讓訓練結果可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(input_dim, hidden_dims=(64, 32), dropout=0.3, use_batchnorm=False,
                seed=42, verbose=True):
    """建立 MLP 並印出結構與參數量。"""
    set_seed(seed)
    model = MLP(input_dim, hidden_dims, dropout, use_batchnorm)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f'可訓練參數量: {n_params:,}')
    return model


# ============================================================
# 3. 訓練與評估
# ============================================================
def train_model(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None,
                epochs=100, batch_size=128, lr=3e-4, weight_decay=1e-4,
                optimizer_name='adam', scheduler_name='cosine', pos_weight=None,
                patience=None, restore_best=True, seed=42, verbose=True, log_every=10):
    """訓練 MLP，每個 epoch 記錄 Training Loss 與 Validation Loss。

    scheduler_name: 'cosine'（學習率隨 epoch 餘弦衰減）/ 'plateau' / 'none'
    patience:       連續幾個 epoch 沒進步就 early stopping（None = 不啟用）
    restore_best:   True 會載回 val_loss 最低的權重（也是一種 early stopping）；
                    基準模型要設成 False，直接使用最後一個 epoch 的權重
    X_test/y_test:  選填。傳入後每個 epoch 也會算一次 Test Loss，方便和 train/val
                    畫在同一張圖上。注意：test 只用來「觀察」，不參與任何訓練決策
                    （early stopping、選權重、調參都只看 val），否則就不是乾淨的測試集。
    回傳 (model, history)；history 含 train_loss / val_loss / test_loss / val_auc / lr。
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    has_test = X_test is not None and y_test is not None
    if has_test:
        Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
        yte = torch.tensor(y_test, dtype=torch.float32, device=device)

    # 違約樣本只佔約 22%，用 pos_weight 提高正類權重可改善 Recall
    pw = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizers = {
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        'adamw': lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        'sgd': lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                       weight_decay=weight_decay),
        'rmsprop': lambda: torch.optim.RMSprop(model.parameters(), lr=lr,
                                               weight_decay=weight_decay),
    }
    if optimizer_name not in optimizers:
        raise ValueError(f'optimizer_name 需為 {list(optimizers)} 之一')
    optimizer = optimizers[optimizer_name]()

    if scheduler_name == 'cosine':       # 學習率隨訓練逐步降低，後期微調更細緻
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'plateau':    # val_loss 停滯就把學習率砍半
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               factor=0.5, patience=5)
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError("scheduler_name 需為 'cosine' / 'plateau' / 'none'")

    n = len(Xtr)
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': [],
               'clean_train_loss': [], 'test_loss': []}
    best_val, best_state, best_epoch, bad_epochs = float('inf'), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)     # 每個 epoch 重新洗牌
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():                        # 驗證階段不計算梯度
            val_logits = model(Xv)
            val_loss = criterion(val_logits, yv).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            clean_train = criterion(model(Xtr), ytr).item()
            # 只做觀察用：不影響 early stopping 與權重選擇
            test_loss = criterion(model(Xte), yte).item() if has_test else None
        val_auc = roc_auc_score(y_val, val_probs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['clean_train_loss'].append(clean_train)
        if has_test:
            history['test_loss'].append(test_loss)

        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val - 1e-5:               # 記住 val loss 最低的權重
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f'Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | '
                  f'clean_train={clean_train:.4f} | val_loss={val_loss:.4f} | '
                  + (f'test_loss={test_loss:.4f} | ' if has_test else '')
                  + f'val_auc={val_auc:.4f} | lr={history["lr"][-1]:.2e}')

        if patience is not None and bad_epochs >= patience:
            if verbose:
                print(f'Early stopping：連續 {patience} 個 epoch 沒有進步，停在 epoch {epoch}')
            break

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val
    history['final_val_loss'] = history['val_loss'][-1]

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)            # 還原 val_loss 最低的權重
        if verbose:
            print(f'\n訓練完成，最佳 epoch = {best_epoch} '
                  f'(val_loss={best_val:.4f})，已載回該權重')
    elif verbose:
        # 基準模型不做任何挑選，直接用最後一個 epoch 的權重
        print(f'\n訓練完成，共 {len(history["train_loss"])} 個 epoch，'
              f'使用最後一個 epoch 的權重 (val_loss={history["val_loss"][-1]:.4f})；'
              f'僅供參考，最低點在 epoch {best_epoch} (val_loss={best_val:.4f})')
    return model, history


def predict_proba(model, X):
    """回傳正類（違約）機率，形狀 (n_samples,)。"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate(y_true, y_probs, threshold=0.5, title='', verbose=True):
    """以指定門檻計算 Accuracy / Precision / Recall / F1 / AUC 與混淆矩陣。"""
    y_pred = (np.asarray(y_probs) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_probs),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'confusion_matrix': cm,
    }
    if verbose:
        head = f'[{title}] ' if title else ''
        print(f'{head}Threshold = {threshold:.4f}')
        print('Confusion Matrix (row=真實, col=預測, 0=正常, 1=違約):')
        print('              pred_0    pred_1')
        print(f'  true_0    {tn:8d}  {fp:8d}')
        print(f'  true_1    {fn:8d}  {tp:8d}')
        print(f'  Accuracy : {metrics["accuracy"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall   : {metrics["recall"]:.4f}')
        print(f'  F1-Score : {metrics["f1"]:.4f}')
        print(f'  ROC-AUC  : {metrics["auc"]:.4f}\n')
    return metrics


def cross_validate_mlp(X_train, y_train, input_dim=None, n_splits=5,
                       hidden_dims=(64, 32), dropout=0.3, use_batchnorm=False,
                       epochs=100, batch_size=128, lr=3e-4, optimizer_name='adam',
                       scheduler_name='cosine', pos_weight=None,
                       random_state=42, verbose=True):
    """在訓練集上做 Stratified K-Fold 交叉驗證，每折重新建模訓練。

    注意：每折內部各自 fit 一次 StandardScaler，確保驗證折不參與縮放統計量的計算。
    """
    input_dim = input_dim or X_train.shape[1]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        scaler = StandardScaler()                       # 每折獨立縮放，避免折間外洩
        X_tr = scaler.fit_transform(X_train[tr_idx]).astype(np.float32)
        X_va = scaler.transform(X_train[va_idx]).astype(np.float32)
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = build_model(input_dim, hidden_dims, dropout, use_batchnorm,
                            seed=random_state + fold, verbose=False)
        model, hist = train_model(model, X_tr, y_tr, X_va, y_va,
                                  epochs=epochs, batch_size=batch_size, lr=lr,
                                  optimizer_name=optimizer_name,
                                  scheduler_name=scheduler_name, pos_weight=pos_weight,
                                  seed=random_state + fold, verbose=False)
        probs = predict_proba(model, X_va)
        m = evaluate(y_va, probs, 0.5, verbose=False)
        rows.append({'fold': fold, 'accuracy': m['accuracy'], 'precision': m['precision'],
                     'recall': m['recall'], 'f1': m['f1'], 'auc': m['auc'],
                     'best_epoch': hist['best_epoch']})
        if verbose:
            print(f'Fold {fold}/{n_splits}: Acc={m["accuracy"]:.4f} P={m["precision"]:.4f} '
                  f'R={m["recall"]:.4f} F1={m["f1"]:.4f} AUC={m["auc"]:.4f} '
                  f'(best epoch {hist["best_epoch"]})')

    table = pd.DataFrame(rows).set_index('fold')
    if verbose:
        mean, std = table.mean(), table.std()
        print('\n5-Fold 平均:')
        for col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            print(f'  {col:<10s} {mean[col]:.4f} ± {std[col]:.4f}')
    return table


def predict_one_sample(model, X_scaled, y_true, feature_names, index=0,
                       threshold=0.5, X_raw=None, top_k=8):
    """展示單一筆樣本的預測過程：輸入特徵 -> 模型機率 -> 預測類別 -> 對照真實標籤。"""
    x = X_scaled[index:index + 1]                       # 保持 2D 形狀 (1, n_features)
    prob = float(predict_proba(model, x)[0])
    pred = int(prob >= threshold)
    truth = int(y_true[index])

    print(f'=== Validation 第 {index} 筆樣本 ===')
    source = X_raw if X_raw is not None else X_scaled
    tag = '前處理後' if X_raw is not None else '縮放後'
    print(f'部分輸入特徵（{tag}，前 {top_k} 欄）:')
    for name, val in list(zip(feature_names, source[index]))[:top_k]:
        print(f'  {name:<12s} = {val:12.2f}')
    print(f'\n模型輸出違約機率 = {prob:.4f}  (門檻 {threshold})')
    print(f'預測結果 = {pred} ({"違約" if pred else "正常"})')
    print(f'真實標籤 = {truth} ({"違約" if truth else "正常"})')
    print('判斷:', '預測正確 (O)' if pred == truth else '預測錯誤 (X)')
    return {'index': index, 'prob': prob, 'pred': pred, 'true': truth}


# ============================================================
# 4. 視覺化
# ============================================================
def plot_loss_curves(history, title='MLP Training History', figsize=(12, 4.5)):
    """左圖：Training / Validation / Test Loss 畫在同一張；右圖：Validation AUC。

    history 若沒有 test_loss（訓練時沒傳 X_test/y_test）就只畫 train 與 val。
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, history['train_loss'], lw=1.8, label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], lw=1.8, label='Validation Loss')
    if history.get('test_loss'):               # 有傳 test 資料才會有這條線
        axes[0].plot(epochs, history['test_loss'], lw=1.8, ls='--', label='Test Loss')
    if 'best_epoch' in history:
        axes[0].axvline(history['best_epoch'], ls='--', c='red', lw=1.2,
                        label=f'best epoch = {history["best_epoch"]}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['val_auc'], lw=1.8, c='green', label='Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title(f'{title} - Validation AUC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_history_comparison(hist_dict, figsize=(15, 4.5)):
    """比較多個模型的訓練歷程：Training Loss、Validation Loss、Validation AUC。

    hist_dict: {模型名稱: history}
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [('train_loss', 'Training Loss'),
              ('val_loss', 'Validation Loss'),
              ('val_auc', 'Validation AUC')]
    if any(h.get('test_loss') for h in hist_dict.values()):
        panels[2] = ('test_loss', 'Test Loss')      # 有 test 紀錄時改畫 Test Loss

    for ax, (key, label) in zip(axes, panels):
        for name, hist in hist_dict.items():
            if not hist.get(key):
                continue
            ax.plot(range(1, len(hist[key]) + 1), hist[key], lw=1.8, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()

    # 文字摘要：各模型跑了幾個 epoch、最低 val_loss 落在哪
    for name, hist in hist_dict.items():
        print(f'{name:<24s} epochs={len(hist["train_loss"]):3d} | '
              f'最低 val_loss={hist["best_val_loss"]:.4f} @ epoch {hist["best_epoch"]:3d} | '
              f'最後 train={hist["train_loss"][-1]:.4f} / val={hist["val_loss"][-1]:.4f} | '
              f'過擬合差距={hist["val_loss"][-1] - hist["train_loss"][-1]:+.4f}')


def compare_models(metrics_dict, plot=True, figsize=(8, 4.5)):
    """把多個模型的評估結果整理成表格並畫長條圖比較。

    metrics_dict: {模型名稱: evaluate() 回傳的 dict}
    """
    cols = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    table = pd.DataFrame(
        [{c: m[c] for c in cols} | {'FP': m['fp'], 'FN': m['fn']}
         for m in metrics_dict.values()],
        index=list(metrics_dict),
    )

    if plot:
        x = np.arange(len(cols))
        width = 0.8 / len(table)
        _, ax = plt.subplots(figsize=figsize)
        for i, (name, row) in enumerate(table.iterrows()):
            offset = (i - (len(table) - 1) / 2) * width
            bars = ax.bar(x + offset, [row[c] for c in cols], width, label=name)
            ax.bar_label(bars, fmt='%.3f', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cols)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison (same validation set)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.show()
    return table


def plot_confusion(cm, ax, title, class_names=('normal(0)', 'default(1)')):
    """畫混淆矩陣，字色依格子底色亮度自動切換，淺底用深色字。"""
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'pred {c}' for c in class_names])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'true {c}' for c in class_names])
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            red, green, blue, _ = im.cmap(im.norm(cm[r, c]))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(c, r, f'{cm[r, c]:d}', ha='center', va='center',
                    color='white' if luminance < 0.45 else '#111111',
                    fontsize=13, fontweight='bold')
    ax.set_title(title)
    return im


def plot_confusion_pair(m_val, m_test, figsize=(10, 4.5)):
    """並排顯示 Validation 與 Test 的混淆矩陣。"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_confusion(m_val['confusion_matrix'], axes[0],
                   f'Validation (th={m_val["threshold"]:.3f})\nF1={m_val["f1"]:.4f}')
    plot_confusion(m_test['confusion_matrix'], axes[1],
                   f'Test (th={m_test["threshold"]:.3f})\nF1={m_test["f1"]:.4f}')
    fig.tight_layout()
    plt.show()
