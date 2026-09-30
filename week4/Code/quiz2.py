"""Week4 Quiz2：Emotion 文字情緒分類（RNN vs. LSTM）。

模組分五區：
  1. 資料處理：讀取 csv、Train:Valid = 8:2 分層切分、斷詞、建立字典、Dataset / DataLoader
  2. 模型：以同一個 TextRNN 類別切換 nn.RNN / nn.LSTM，其餘架構完全相同，比較才公平
  3. 訓練與評估：訓練迴圈（梯度裁剪、Early Stopping，依 Valid Macro-F1 選最佳權重）
  4. 結果比較：Accuracy / Macro-Precision / Macro-Recall / Macro-F1 / Macro-AUC、
     各類別報告、混淆矩陣、訓練曲線
  5. 推論與介面：單句預測、Gradio 互動介面
"""

import copy
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PAD, UNK = '<pad>', '<unk>'
PAD_IDX, UNK_IDX = 0, 1


def get_device():
    """有 GPU 就用 GPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """固定亂數種子，讓結果可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """可訓練參數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 1. 資料處理
# ============================================================
def load_data(csv_path, text_col='Comment', label_col='Emotion'):
    """讀取 csv，去除空值與重複句子，回傳 DataFrame(text, label)。"""
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df = df.dropna().drop_duplicates(subset='text').reset_index(drop=True)
    return df


def split_train_valid(df, valid_size=0.2, random_state=42):
    """以 label 分層抽樣切出 Train:Valid = 8:2，回傳 (train_df, valid_df)。"""
    train_df, valid_df = train_test_split(
        df, test_size=valid_size, stratify=df['label'], random_state=random_state,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def tokenize(text):
    """小寫化後只保留英文字母、數字與撇號，以空白切詞。"""
    return re.findall(r"[a-z0-9']+", str(text).lower())


class Vocab:
    """詞 ↔ 索引對照表。0 = <pad>、1 = <unk>（字典外的詞）。"""

    def __init__(self, texts, min_freq=2, max_size=None):
        counter = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=None):
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        return ids or [UNK_IDX]   # 空字串至少給一個 token，避免序列長度為 0


class TextDataset(Dataset):
    """把每句話轉成詞索引序列，label 轉成類別索引。"""

    def __init__(self, df, vocab, class_to_idx, max_len=64):
        self.seqs = [vocab.encode(t, max_len) for t in df['text']]
        self.targets = [class_to_idx[c] for c in df['label']]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def collate_batch(batch):
    """把一個 batch 補齊（pad）到該 batch 最長的長度，回傳 (x, lengths, y)。"""
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = torch.full((len(seqs), int(lengths.max())), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x, lengths, torch.tensor(targets, dtype=torch.long)


def build_dataloaders(train_df, valid_df, vocab, class_to_idx, max_len=64, batch_size=64):
    """回傳 {'train': ..., 'valid': ...} 兩個 DataLoader。"""
    train_ds = TextDataset(train_df, vocab, class_to_idx, max_len)
    valid_ds = TextDataset(valid_df, vocab, class_to_idx, max_len)
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch),
        'valid': DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch),
    }


# ============================================================
# 2. 模型
# ============================================================
class TextRNN(nn.Module):
    """Embedding → (Bi)RNN / (Bi)LSTM → 最後時間步的 hidden state → Dropout → FC。

    rnn_type='rnn' 用 nn.RNN（tanh），'lstm' 用 nn.LSTM，其餘超參數相同。
    以 pack_padded_sequence 包裝，讓 RNN 只看真實長度，<pad> 不會影響最後的 hidden state。
    """

    def __init__(self, vocab_size, num_classes, rnn_type='lstm', embed_dim=128,
                 hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM}[self.rnn_type]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = rnn_cls(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.rnn_type == 'lstm':
            h = h[0]                       # LSTM 回傳 (h_n, c_n)，只取 h_n
        # h: (num_layers * num_directions, B, H) → 取最後一層（雙向則串接正反兩方向）
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(h))


# ============================================================
# 3. 訓練與評估
# ============================================================
@torch.no_grad()
def predict_proba(model, loader, device):
    """回傳 (softmax 機率 (N, C), 真實標籤 (N,))。"""
    model.eval()
    probs, targets = [], []
    for x, lengths, y in loader:
        logits = model(x.to(device), lengths)
        probs.append(torch.softmax(logits, dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def compute_metrics(y_true, prob):
    """Accuracy、Macro-Precision / Recall / F1 與 Macro-AUC（One-vs-Rest）。"""
    y_pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': p,
        'recall': r,
        'f1': f1,
        'auc': roc_auc_score(y_true, prob, multi_class='ovr', average='macro',
                             labels=list(range(prob.shape[1]))),
    }


def train_model(model, loaders, device, epochs=30, lr=1e-3, weight_decay=1e-5,
                clip_norm=1.0, patience=5, save_path=None, verbose=True):
    """訓練模型，以 Valid Macro-F1 選最佳權重（Early Stopping），回傳 (最佳模型, history DataFrame)。

    RNN 容易梯度爆炸，因此每步做 gradient clipping（clip_norm）。
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for x, lengths, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)

        train_prob, train_y = predict_proba(model, loaders['train'], device)
        valid_prob, valid_y = predict_proba(model, loaders['valid'], device)
        valid_loss = nn.functional.nll_loss(torch.tensor(valid_prob).clamp_min(1e-12).log(),
                                            torch.tensor(valid_y, dtype=torch.long)).item()
        train_m = compute_metrics(train_y, train_prob)
        valid_m = compute_metrics(valid_y, valid_prob)
        scheduler.step(valid_m['f1'])

        history.append({
            'epoch': epoch,
            'train_loss': total_loss / n,
            'valid_loss': valid_loss,
            'train_acc': train_m['accuracy'],
            'valid_acc': valid_m['accuracy'],
            'train_f1': train_m['f1'],
            'valid_f1': valid_m['f1'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        if verbose:
            h = history[-1]
            print(f"[{model.rnn_type.upper():4s}] epoch {epoch:2d} | "
                  f"loss {h['train_loss']:.4f}/{h['valid_loss']:.4f} | "
                  f"acc {h['train_acc']:.4f}/{h['valid_acc']:.4f} | "
                  f"f1 {h['train_f1']:.4f}/{h['valid_f1']:.4f} | {time.time() - t0:.1f}s")

        if valid_m['f1'] > best_f1:
            best_f1, best_state, bad_epochs = valid_m['f1'], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}（最佳 Valid Macro-F1 = {best_f1:.4f}）')
                break

    model.load_state_dict(best_state)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(best_state, save_path)
    return model, pd.DataFrame(history)


# ============================================================
# 4. 結果比較
# ============================================================
def compare_models(results):
    """results = {name: (prob, y_true)}（即 predict_proba 的回傳值）→ 各模型指標的比較表。"""
    rows = {name: compute_metrics(y, prob) for name, (prob, y) in results.items()}
    return pd.DataFrame(rows).T.round(4)


def report(y_true, prob, class_names):
    """各類別的 Precision / Recall / F1 / Support。"""
    rep = classification_report(y_true, prob.argmax(1), target_names=class_names,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


def plot_histories(histories, figsize=(15, 4)):
    """histories = {name: history_df} → Loss / Accuracy / Macro-F1 三張訓練曲線。"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for (name, h), color in zip(histories.items(), plt.cm.tab10.colors):
        for ax, key in zip(axes, ['loss', 'acc', 'f1']):
            ax.plot(h['epoch'], h[f'train_{key}'], '--', color=color, label=f'{name} train')
            ax.plot(h['epoch'], h[f'valid_{key}'], '-', color=color, label=f'{name} valid')
    for ax, title in zip(axes, ['Loss', 'Accuracy', 'Macro-F1']):
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results, class_names, figsize=None):
    """results = {name: (prob, y_true)} → 並排畫出各模型的混淆矩陣（格內為筆數）。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, (prob, y)) in zip(axes, results.items()):
        cm = confusion_matrix(y, prob.argmax(1), labels=range(len(class_names)))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name} (acc={np.trace(cm) / cm.sum():.3f})')
    plt.tight_layout()
    plt.show()


def plot_metric_bars(compare_df, metrics=('accuracy', 'precision', 'recall', 'f1', 'auc'), figsize=(9, 4)):
    """把 compare_models 的比較表畫成長條圖。"""
    ax = compare_df[list(metrics)].T.plot(kind='bar', figsize=figsize, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title('Valid metrics: RNN vs. LSTM')
    ax.grid(axis='y', alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=8)
    plt.tight_layout()
    plt.show()


def misclassified_examples(valid_df, y_true, prob, class_names, n=10):
    """列出模型預測錯誤的句子，方便做錯誤分析。"""
    y_pred = prob.argmax(1)
    df = valid_df.copy()
    df['pred'] = [class_names[i] for i in y_pred]
    df['confidence'] = prob.max(1).round(4)
    return df[y_pred != y_true].head(n)


# ============================================================
# 5. 推論與介面
# ============================================================
@torch.no_grad()
def predict_text(model, text, vocab, class_names, device, max_len=64):
    """輸入一句話，回傳 {類別: 機率}。"""
    model.eval()
    ids = vocab.encode(text, max_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    prob = torch.softmax(model(x, torch.tensor([len(ids)])), dim=1)[0].cpu().numpy()
    return {c: float(p) for c, p in zip(class_names, prob)}


def build_gradio_app(models, vocab, class_names, device, max_len=64, examples=None):
    """models = {name: model}。輸入一段英文，同時顯示每個模型預測的情緒機率。"""
    import gradio as gr

    def infer(text):
        if not text or not text.strip():
            return [{} for _ in models]
        return [predict_text(m, text, vocab, class_names, device, max_len) for m in models.values()]

    return gr.Interface(
        fn=infer,
        inputs=gr.Textbox(lines=3, label='輸入一段英文句子', placeholder='e.g. i feel so happy today'),
        outputs=[gr.Label(num_top_classes=len(class_names), label=f'{name} 預測') for name in models],
        examples=examples,
        title='Emotion Classification：RNN vs. LSTM',
        description=f'類別：{", ".join(class_names)}。輸入句子後，同時比較 RNN 與 LSTM 的預測機率。',
        flagging_mode='never',
    )
