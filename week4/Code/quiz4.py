"""Week4 Quiz4：Flickr8k Image Captioning（CNN Encoder + Transformer Decoder）。

模組分六區：
  1. 資料處理：讀取 Flickr8k.token.txt、清理描述文字、以「影像」為單位切 Train:Valid:Test = 7:2:1、建立字典
  2. 影像特徵：凍結的 ImageNet 預訓練 ConvNeXt-Tiny 抽出 7×7×768 的特徵網格，快取成 .npy
     （以 memory-map 讀取，不必整個載入記憶體）
  3. 模型：影像特徵 → 線性投影 + 位置編碼 → Transformer Decoder（Masked Self-Attention + Cross-Attention）
  4. 訓練：Teacher Forcing、Label Smoothing、Warmup + Cosine、依 Valid BLEU-4 選最佳權重（Early Stopping）
  5. 產生描述與評估：Greedy / Beam Search、BLEU-1~4（nltk）、BERTScore（bert_score）
  6. 視覺化：訓練曲線、測試影像 + 模型描述 + 正確描述
"""

import copy
import math
import os
import random
import re
import time
from collections import Counter
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm  # auto 會自動適配 Jupyter Notebook 與一般終端機

PAD, START, END, UNK = '<pad>', '<start>', '<end>', '<unk>'
PAD_IDX, START_IDX, END_IDX, UNK_IDX = 0, 1, 2, 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    """有 GPU 就用 GPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """固定亂數種子，讓結果可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=True):
    """參數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ============================================================
# 1. 資料處理
# ============================================================
def clean_caption(text):
    """小寫化、只保留英文字母組成的詞（去掉標點與數字），以空白串接。"""
    return ' '.join(re.findall(r'[a-z]+', str(text).lower()))


def load_captions(token_path, image_dir):
    """讀取 Flickr8k.token.txt（每行：`影像名#編號<TAB>描述`），回傳 DataFrame(image, caption)。

    只保留 image_dir 中實際存在的影像（原始檔有一筆 `2258277193_586949ec62.jpg.1` 找不到影像）。
    """
    rows = []
    with open(token_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            key, caption = line.split('\t', 1)
            rows.append({'image': key.split('#')[0], 'caption': clean_caption(caption)})
    df = pd.DataFrame(rows)
    existing = set(os.listdir(image_dir))
    return df[df['image'].isin(existing) & (df['caption'].str.len() > 0)].reset_index(drop=True)


def split_images(images, ratios=(0.7, 0.2, 0.1), random_state=42):
    """以「影像」為單位隨機切成 train / valid / test（同一張影像的 5 句描述必在同一個子集，避免資料洩漏）。"""
    images = np.array(sorted(set(images)))
    rng = np.random.default_rng(random_state)
    rng.shuffle(images)
    n_train = int(round(len(images) * ratios[0]))
    n_valid = int(round(len(images) * ratios[1]))
    return {
        'train': sorted(map(str, images[:n_train])),
        'valid': sorted(map(str, images[n_train:n_train + n_valid])),
        'test': sorted(map(str, images[n_train + n_valid:])),
    }


class Vocab:
    """詞 ↔ 索引對照表。0 = <pad>、1 = <start>、2 = <end>、3 = <unk>。"""

    def __init__(self, captions, min_freq=5):
        counter = Counter(w for c in captions for w in c.split())
        words = sorted((w for w, n in counter.items() if n >= min_freq), key=lambda w: (-counter[w], w))
        self.itos = [PAD, START, END, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, caption, max_len=None):
        """'a dog runs' → [<start>, a, dog, runs, <end>]（超過 max_len 會截斷，但保留 <end>）。"""
        ids = [self.stoi.get(w, UNK_IDX) for w in caption.split()]
        if max_len is not None:
            ids = ids[:max_len - 2]
        return [START_IDX] + ids + [END_IDX]

    def decode(self, ids):
        """索引序列 → 句子（遇到 <end> 停止，略過特殊 token）。"""
        words = []
        for i in ids:
            i = int(i)
            if i == END_IDX:
                break
            if i in (PAD_IDX, START_IDX):
                continue
            words.append(self.itos[i])
        return ' '.join(words)


def references_by_image(captions_df, images):
    """{影像名: [5 句正確描述]}，評估 BLEU / BERTScore 用。"""
    groups = captions_df.groupby('image')['caption'].apply(list)
    return {img: groups[img] for img in images}


# ============================================================
# 2. 影像特徵
# ============================================================
def build_encoder():
    """凍結的 ImageNet 預訓練 ConvNeXt-Tiny（去掉分類頭），輸出 (B, 768, 7, 7) 的特徵網格。"""
    cnn = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    encoder = cnn.features.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def image_transform(img_size=224):
    """整張影像縮放成 img_size×img_size（不裁切，避免把描述中提到的物體切掉）。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageFileDataset(Dataset):
    def __init__(self, image_names, image_dir, transform):
        self.paths = [os.path.join(image_dir, n) for n in image_names]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))


@torch.no_grad()
def extract_features(image_names, image_dir, cache_path, device, batch_size=64, img_size=224):
    """抽出每張影像的特徵網格並存成 cache_path（.npy, float16, (N, 49, 768)），影像順序存在同名 .txt。

    已存在且影像清單相同時直接讀取。回傳 (以 memory-map 開啟的特徵陣列, {影像名: 索引})。
    """
    names_path = os.path.splitext(cache_path)[0] + '_names.txt'
    image_names = list(image_names)
    if os.path.isfile(cache_path) and os.path.isfile(names_path):
        with open(names_path, encoding='utf-8') as f:
            cached = f.read().split('\n')
        if cached == image_names:
            return np.load(cache_path, mmap_mode='r'), {n: i for i, n in enumerate(image_names)}

    encoder = build_encoder().to(device)
    loader = DataLoader(ImageFileDataset(image_names, image_dir, image_transform(img_size)),
                        batch_size=batch_size, shuffle=False)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    feats = None
    start, t0 = 0, time.time()
    for x in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            f = encoder(x.to(device))                       # (B, 768, 7, 7)
        f = f.flatten(2).transpose(1, 2).float().cpu().numpy().astype(np.float16)   # (B, 49, 768)
        if feats is None:
            feats = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.float16,
                                              shape=(len(image_names),) + f.shape[1:])
        feats[start:start + len(f)] = f
        start += len(f)
    feats.flush()
    del feats, encoder
    with open(names_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(image_names))
    print(f'特徵抽取完成：{len(image_names)} 張，{time.time() - t0:.0f}s → {cache_path}')
    return np.load(cache_path, mmap_mode='r'), {n: i for i, n in enumerate(image_names)}


class CaptionDataset(Dataset):
    """訓練用：每一句描述是一筆樣本 → (影像特徵 (49, 768), token 序列)。"""

    def __init__(self, captions_df, images, features, feat_index, vocab, max_len=40):
        df = captions_df[captions_df['image'].isin(set(images))]
        self.feat_rows = [feat_index[i] for i in df['image']]
        self.seqs = [vocab.encode(c, max_len) for c in df['caption']]
        self.features = features

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        feat = torch.from_numpy(np.asarray(self.features[self.feat_rows[idx]], dtype=np.float32))
        return feat, self.seqs[idx]


class ImageFeatureDataset(Dataset):
    """產生描述 / 評估用：每張影像一筆 → (影像特徵, 影像名)。"""

    def __init__(self, images, features, feat_index):
        self.images = list(images)
        self.rows = [feat_index[i] for i in self.images]
        self.features = features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(np.asarray(self.features[self.rows[idx]], dtype=np.float32)), self.images[idx]


def collate_captions(batch):
    feats, seqs = zip(*batch)
    tokens = torch.full((len(seqs), max(len(s) for s in seqs)), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return torch.stack(feats), tokens


def collate_images(batch):
    feats, names = zip(*batch)
    return torch.stack(feats), list(names)


def build_dataloaders(captions_df, splits, features, feat_index, vocab, max_len=40, batch_size=64):
    """'train'：每句描述一筆（打亂）；'valid' / 'test'：每張影像一筆（產生描述用）。"""
    train_ds = CaptionDataset(captions_df, splits['train'], features, feat_index, vocab, max_len)
    valid_loss_ds = CaptionDataset(captions_df, splits['valid'], features, feat_index, vocab, max_len)
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_captions),
        'valid_loss': DataLoader(valid_loss_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_captions),
    }
    for split in ('valid', 'test'):
        loaders[split] = DataLoader(ImageFeatureDataset(splits[split], features, feat_index),
                                    batch_size=batch_size, shuffle=False, collate_fn=collate_images)
    return loaders


# ============================================================
# 3. 模型
# ============================================================
class CaptionTransformer(nn.Module):
    """影像特徵網格 → Transformer Decoder 逐字產生描述。

    - Encoder 端：49 個影像區塊特徵 (768 維) → Linear 投影到 d_model + 可學習的位置編碼 → 1 層 Transformer Encoder
      （讓影像區塊之間先互相交換資訊）
    - Decoder 端：詞嵌入 + 位置編碼 → num_layers 層 Transformer Decoder
      （Masked Self-Attention 只看已產生的詞；Cross-Attention 決定每產生一個詞時要看影像的哪些區域）
    """

    def __init__(self, vocab_size, feat_dim=768, n_regions=49, d_model=512, nhead=8, num_layers=3,
                 dim_ff=2048, dropout=0.1, max_len=40):
        super().__init__()
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.region_pos = nn.Parameter(torch.zeros(1, n_regions, d_model))
        self.img_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True),
            num_layers=1,
        )
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.word_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model, self.max_len = d_model, max_len
        nn.init.trunc_normal_(self.region_pos, std=0.02)
        nn.init.trunc_normal_(self.word_pos, std=0.02)

    def encode(self, feats):
        """(B, 49, 768) → memory (B, 49, d_model)。"""
        return self.img_encoder(self.feat_proj(feats) + self.region_pos)

    def decode(self, tokens, memory):
        """tokens (B, T) + memory → 每個位置下一個詞的 logits (B, T, V)。"""
        T = tokens.size(1)
        x = self.dropout(self.embed(tokens) * math.sqrt(self.d_model) + self.word_pos[:, :T])
        causal = torch.triu(torch.full((T, T), float('-inf'), device=tokens.device), diagonal=1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=(tokens == PAD_IDX))
        return self.out(self.norm(x))

    def forward(self, feats, tokens):
        return self.decode(tokens, self.encode(feats))


def _next_token_logits(model, tokens, memory):
    """最後一個位置的 logits；產生描述時禁止輸出 <pad> / <start> / <unk>。"""
    logits = model.decode(tokens, memory)[:, -1].float()
    logits[:, [PAD_IDX, START_IDX, UNK_IDX]] = float('-inf')
    return logits


@torch.no_grad()
def greedy_decode(model, feats, max_len=30):
    """每一步都選機率最高的詞（整個 batch 一起產生）。回傳 (B, ≤max_len) 的 token 序列。"""
    model.eval()
    memory = model.encode(feats)
    tokens = torch.full((feats.size(0), 1), START_IDX, dtype=torch.long, device=feats.device)
    done = torch.zeros(feats.size(0), dtype=torch.bool, device=feats.device)
    for _ in range(max_len):
        next_tok = _next_token_logits(model, tokens, memory).argmax(-1)
        next_tok = next_tok.masked_fill(done, PAD_IDX)
        tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
        done |= next_tok == END_IDX
        if done.all():
            break
    return tokens[:, 1:]


@torch.no_grad()
def beam_search(model, feat, beam_size=3, max_len=30, length_penalty=0.7):
    """單張影像的 Beam Search：每一步保留累積 log 機率最高的 beam_size 條候選句。

    結束的句子以 score / 長度^length_penalty 排序，避免模型偏好過短的句子。回傳最佳句的 token list。
    """
    model.eval()
    memory = model.encode(feat.unsqueeze(0))                 # (1, 49, d)
    beams = torch.full((1, 1), START_IDX, dtype=torch.long, device=feat.device)
    scores = torch.zeros(1, device=feat.device)
    finished = []
    for _ in range(max_len):
        logp = torch.log_softmax(_next_token_logits(model, beams, memory.expand(beams.size(0), -1, -1)), -1)
        cand = (scores[:, None] + logp).flatten()
        top_scores, top_idx = cand.topk(beam_size * 2)
        vocab_size = logp.size(1)
        new_beams, new_scores = [], []
        for s, idx in zip(top_scores.tolist(), top_idx.tolist()):
            b, tok = divmod(idx, vocab_size)
            seq = beams[b].tolist() + [tok]
            if tok == END_IDX:
                finished.append((s / (len(seq) - 1) ** length_penalty, seq[1:]))
            else:
                new_beams.append(seq)
                new_scores.append(s)
            if len(new_beams) == beam_size:
                break
        if len(finished) >= beam_size or not new_beams:
            break
        beams = torch.tensor(new_beams, dtype=torch.long, device=feat.device)
        scores = torch.tensor(new_scores, device=feat.device)
    if not finished:
        finished = [(s / beams.size(1) ** length_penalty, seq[1:]) for s, seq in zip(scores.tolist(), beams.tolist())]
    return max(finished, key=lambda t: t[0])[1]


def generate_captions(model, loader, vocab, device, method='greedy', beam_size=3, max_len=30,
                      progress=True, desc=None):
    """對 loader（每張影像一筆）產生描述，回傳 {影像名: 描述}。"""
    model.eval()
    preds = {}
    for feats, names in tqdm(loader, desc=desc or f'Generating ({method})', leave=False, disable=not progress):
        feats = feats.to(device)
        if method == 'greedy':
            for name, ids in zip(names, greedy_decode(model, feats, max_len).tolist()):
                preds[name] = vocab.decode(ids)
        else:
            for name, feat in zip(names, feats):
                preds[name] = vocab.decode(beam_search(model, feat, beam_size, max_len))
    return preds


# ============================================================
# 4. 評估指標
# ============================================================
def bleu_scores(preds, refs):
    """Corpus-level BLEU-1 ~ BLEU-4（每張影像以 5 句正確描述為參考答案）。"""
    from nltk.translate.bleu_score import corpus_bleu

    images = list(preds)
    hyps = [preds[i].split() for i in images]
    references = [[r.split() for r in refs[i]] for i in images]
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25, 0.25, 0.25, 0.25)]
    return {f'BLEU-{n}': corpus_bleu(references, hyps, weights=w) for n, w in enumerate(weights, 1)}


def sentence_bleu4(preds, refs):
    """每張影像各自的 BLEU-4（加 smoothing，短句不會直接變 0），用來挑好 / 差的例子。"""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smooth = SmoothingFunction().method1
    return {i: sentence_bleu([r.split() for r in refs[i]], preds[i].split(), smoothing_function=smooth)
            for i in preds}


def bert_scores(preds, refs, model_type='distilbert-base-uncased', device=None, batch_size=64,
                rescale_with_baseline=True):
    """BERTScore（Precision / Recall / F1）：以 BERT 的 contextual embedding 比對語意相似度，
    同義詞或換句話說也能得分（BLEU 只看 n-gram 是否完全相同）。多個參考答案時取最高分。

    rescale_with_baseline=True 會把分數線性重新縮放，讓數值較有鑑別度（原始分數通常都擠在 0.8~0.9）。
    """
    from bert_score import score

    images = list(preds)
    cands = [preds[i] for i in images]
    references = [refs[i] for i in images]
    device = str(device) if device is not None else None
    P, R, F1 = score(cands, references, model_type=model_type, lang='en', device=device,
                     batch_size=batch_size, rescale_with_baseline=rescale_with_baseline, verbose=False)
    return {'BERTScore-P': P.mean().item(), 'BERTScore-R': R.mean().item(), 'BERTScore-F1': F1.mean().item()}


def evaluate_captions(preds, refs, use_bertscore=True, **bert_kwargs):
    """BLEU-1~4 + BERTScore，回傳 dict。"""
    result = bleu_scores(preds, refs)
    if use_bertscore:
        result.update(bert_scores(preds, refs, **bert_kwargs))
    return result


def caption_stats(preds, refs):
    """模型描述的平均長度、詞彙多樣性（用到幾個不同的詞），與正確描述比較。"""
    pred_words = [p.split() for p in preds.values()]
    ref_words = [r.split() for i in preds for r in refs[i]]
    return pd.DataFrame({
        'avg length': [np.mean([len(w) for w in pred_words]), np.mean([len(w) for w in ref_words])],
        'unique words': [len({w for s in pred_words for w in s}), len({w for s in ref_words for w in s})],
        'unique captions': [len(set(preds.values())), len({r for i in preds for r in refs[i]})],
    }, index=['model', 'reference']).round(2)


# ============================================================
# 5. 訓練
# ============================================================
@torch.no_grad()
def caption_loss(model, loader, criterion, device, progress=True):
    model.eval()
    total, n = 0.0, 0
    for feats, tokens in tqdm(loader, desc='Valid loss', leave=False, disable=not progress):
        feats, tokens = feats.to(device), tokens.to(device)
        logits = model(feats, tokens[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        total += loss.item() * feats.size(0)
        n += feats.size(0)
    return total / n


def train_model(model, loaders, vocab, refs_valid, device, epochs=20, lr=3e-4, weight_decay=0.01,
                warmup_epochs=1, label_smoothing=0.1, patience=4, save_path=None, resume=True, verbose=True):
    """Teacher Forcing 訓練：輸入 <start> w1 … wn，預測 w1 … wn <end>。

    每個 epoch 在 Valid 上以 greedy 產生描述並計算 BLEU-4，以此選最佳權重並做 Early Stopping。
    resume=True 且 save_path 已存在：直接載入權重與 history，不重新訓練。
    """
    hist_path = None if save_path is None else os.path.splitext(save_path)[0] + '_history.csv'
    if resume and save_path and os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
        history = pd.read_csv(hist_path) if os.path.isfile(hist_path) else pd.DataFrame()
        if verbose:
            print(f'載入已訓練的權重: {save_path}（RESUME=True，跳過訓練，因此不會有進度條）')
        return model.to(device), history

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(loaders['train'])
    total_steps, warmup_steps = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        # leave=True：每個 epoch 的進度條跑完後保留在輸出中
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch:2d}/{epochs} [train]", leave=True)
        for feats, tokens in pbar:
            feats, tokens = feats.to(device), tokens.to(device)
            logits = model(feats, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * feats.size(0)
            n += feats.size(0)
            pbar.set_postfix({'train_loss': f"{total / n:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        valid_loss = caption_loss(model, loaders['valid_loss'], eval_criterion, device)
        preds = generate_captions(model, loaders['valid'], vocab, device, method='greedy',
                                  desc=f'Epoch {epoch:2d}/{epochs} [valid BLEU]')
        bleu = bleu_scores(preds, refs_valid)
        history.append({
            'epoch': epoch,
            'train_loss': total / n,
            'valid_loss': valid_loss,
            'valid_bleu1': bleu['BLEU-1'],
            'valid_bleu4': bleu['BLEU-4'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': time.time() - t0,
        })
        if verbose:
            h = history[-1]
            example = preds[next(iter(preds))]
            # tqdm.write 不會打斷進度條的顯示
            tqdm.write(f"epoch {epoch:2d} | loss {h['train_loss']:.4f}/{h['valid_loss']:.4f} | "
                       f"valid BLEU-1 {h['valid_bleu1']:.4f} BLEU-4 {h['valid_bleu4']:.4f} | "
                       f"{h['time']:.0f}s | 例: {example}")

        if bleu['BLEU-4'] > best_bleu:
            best_bleu, best_state, bad_epochs = bleu['BLEU-4'], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    tqdm.write(f'Early stopping at epoch {epoch}（最佳 Valid BLEU-4 = {best_bleu:.4f}）')
                break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(best_state, save_path)
        history.to_csv(hist_path, index=False)
    return model, history


# ============================================================
# 6. 視覺化
# ============================================================
def plot_history(history, figsize=(12, 4)):
    """Loss 與 Valid BLEU 訓練曲線。"""
    if history.empty:
        print('沒有訓練紀錄（權重是直接載入的）')
        return
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(history['epoch'], history['train_loss'], '--', label='train (label smoothing)')
    axes[0].plot(history['epoch'], history['valid_loss'], '-', label='valid')
    axes[0].set_title('Cross-Entropy Loss')
    axes[1].plot(history['epoch'], history['valid_bleu1'], '-o', label='valid BLEU-1')
    axes[1].plot(history['epoch'], history['valid_bleu4'], '-o', label='valid BLEU-4')
    axes[1].set_title('Valid BLEU (greedy)')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_caption_lengths(captions_df, figsize=(7, 3)):
    """描述長度（詞數）分佈，用來決定 max_len。"""
    lengths = captions_df['caption'].str.split().str.len()
    plt.figure(figsize=figsize)
    plt.hist(lengths, bins=range(0, lengths.max() + 2))
    plt.xlabel('number of words')
    plt.ylabel('count')
    plt.title('Caption length')
    plt.tight_layout()
    plt.show()
    return lengths.describe().round(1)


def show_captions(images, image_dir, refs, preds=None, scores=None, n_refs=5, cols=2, img_width=6):
    """顯示影像 + 模型描述（preds = {方法名: {影像名: 描述}}）+ 正確描述。

    scores = {影像名: 分數}（例如 sentence BLEU-4）會顯示在標題上。
    支援長文字自動折行與動態垂直偏移，避免文字與上下子圖重疊。
    """
    preds = preds or {}
    rows = math.ceil(len(images) / cols)
    
    # 根據預測數量與 GT 數量動態分配足夠的每列高度（給長文字充足空間）
    n_lines_base = len(preds) + n_refs + 1
    row_height = 4.0 + 0.38 * n_lines_base
    fig, axes = plt.subplots(rows, cols, figsize=(img_width * cols, rows * row_height))
    axes = np.atleast_1d(axes).flatten()
    
    # 子圖寬度內單行文字的安全折行字元數（避免超出圖寬）
    char_width = 52

    for ax, name in zip(axes, images):
        img_path = os.path.join(image_dir, name)
        ax.imshow(Image.open(img_path).convert('RGB'))
        ax.set_xticks([])
        ax.set_yticks([])
        
        title = name if scores is None else f'{name}  (BLEU-4 = {scores[name]:.3f})'
        ax.set_title(title, fontsize=9.5)

        # 整理文字清單與對應顏色
        lines = [(f'[{method}] {p[name]}', 'tab:blue') for method, p in preds.items()]
        lines.append(('Ground truth:', 'black'))
        lines += [(f'  {j + 1}. {r}', 'dimgray') for j, r in enumerate(refs[name][:n_refs])]

        # 動態累加 y 偏移量，徹底避免長句子換行覆蓋到下一行文字
        y_offset = -0.06
        for text, color in lines:
            wrapped_text = textwrap.fill(text, width=char_width)
            sublines_count = wrapped_text.count('\n') + 1

            ax.text(
                0.0, y_offset, wrapped_text,
                color=color,
                fontsize=8.5,
                ha='left',
                va='top',
                transform=ax.transAxes,
                fontweight='bold' if color == 'tab:blue' else 'normal',
                linespacing=1.25
            )
            # 根據該段文字實際佔據的行數往下推移 y 軸
            y_offset -= (0.048 * sublines_count + 0.015)

    # 隱藏多餘沒有圖片的子圖
    for ax in axes[len(images):]:
        ax.axis('off')

    # 設定子圖間距，保留足夠的上下緩衝區
    plt.tight_layout(h_pad=5.0, w_pad=2.0)
    plt.show()


@torch.no_grad()
def caption_image(model, image_path, vocab, device, encoder=None, method='beam', beam_size=3, max_len=30):
    """任意一張影像 → 描述（現場抽特徵，不需要快取）。"""
    encoder = encoder or build_encoder().to(device)
    x = image_transform()(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    feat = encoder(x).flatten(2).transpose(1, 2).float()[0]
    if method == 'greedy':
        ids = greedy_decode(model, feat.unsqueeze(0), max_len)[0].tolist()
    else:
        ids = beam_search(model, feat, beam_size, max_len)
    return vocab.decode(ids)
