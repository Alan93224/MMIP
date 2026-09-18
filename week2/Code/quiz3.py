"""Week2 Quiz3：MLP 在 Validation Dataset 上的 Prediction Score 與 ROC / AUC。

沿用 Quiz2 訓練好的改良模型（Improved MLP）與同一份 Train / Validation 資料：
  1. 全量預測：輸出整個 Validation Dataset 每一筆的 Prediction Score / Probability
  2. ROC Curve：X 軸 FPR、Y 軸 TPR，並把 AUC 數值標示在圖上
  3. 第二個模型：用同一份 Training Dataset 訓練 XGBoost
     （先前也比過 LogisticRegression / SVM / DecisionTree / RandomForest /
     NaiveBayes，XGBoost 的 AUC 最好，因此只保留它），
     把兩條 ROC 疊在同一張圖比較
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from . import quiz2

# XGBoost 的固定超參數（樹的形狀與取樣策略；learning_rate / n_estimators
# 因為要做實驗，改成 train_xgboost 的獨立參數）
XGB_PARAMS = dict(
    max_depth=4, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, min_child_weight=5,
    eval_metric='auc', n_jobs=-1,
)

# 預設值：小 learning rate + 大 n_estimators 上限，讓早停自己決定要幾棵樹
DEFAULT_LR = 0.02
DEFAULT_N_ESTIMATORS = 2000
DEFAULT_ES_ROUNDS = 100
DEFAULT_ES_HOLDOUT = 0.15

XGB_DISPLAY_NAME = 'XGBoost'
MLP_DISPLAY_NAME = 'MLP (Improved)'


def make_xgboost(random_state=42, scale_pos_weight=1.0, **kwargs):
    """建立（尚未訓練的）XGBClassifier。

    XGBoost 需另外安裝（pip install xgboost），故延後 import。
    kwargs 會覆寫 XGB_PARAMS 的預設值。
    """
    from xgboost import XGBClassifier
    params = dict(XGB_PARAMS)
    params.update(kwargs)
    return XGBClassifier(scale_pos_weight=scale_pos_weight,
                         random_state=random_state, **params)


# ============================================================
# 1. Validation Dataset 全量 Prediction Score
# ============================================================
def build_score_table(probs, y_true, threshold=0.5, feature_names=None, X_raw=None,
                      show=10, sort_by=None, verbose=True, model_name='MLP'):
    """把一組 Prediction Score 整理成明細表（兩個模型共用的核心）。

    參數：
      probs         : 模型對每一筆樣本輸出的正類機率，形狀 (n_samples,)
      y_true        : validation 真實標籤
      threshold     : 分類門檻，機率 >= threshold 判為違約(1)
      feature_names : 特徵名稱；有給且有 X_raw 時會附上前幾欄原始特徵值
      X_raw         : 未縮放的特徵（前處理後），只為了讓表格好讀
      show          : 印出前幾筆；None 表示不印明細
      sort_by       : None（照原順序）/ 'prob_desc' / 'prob_asc'

    回傳：DataFrame，每列一筆樣本，欄位
      index / prob（Prediction Score）/ pred / true / correct
    """
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (probs >= threshold).astype(int)

    table = pd.DataFrame({
        'index': np.arange(len(probs)),
        'prob': probs,                       # Prediction Score / Probability
        'pred': preds,
        'true': y_true,
        'correct': preds == y_true,
    })

    # 附上幾欄原始特徵，方便肉眼對照（不影響計算）
    if X_raw is not None and feature_names is not None:
        for name in list(feature_names)[:4]:
            table[name] = np.asarray(X_raw)[:, list(feature_names).index(name)]

    if sort_by == 'prob_desc':
        table = table.sort_values('prob', ascending=False)
    elif sort_by == 'prob_asc':
        table = table.sort_values('prob', ascending=True)

    if verbose:
        n = len(table)
        print(f'=== [{model_name}] Validation Dataset Prediction Score'
              f'（共 {n} 筆，門檻 {threshold:.4f}）===')
        print(f'Score 範圍: min={probs.min():.4f}, max={probs.max():.4f}, '
              f'mean={probs.mean():.4f}, median={np.median(probs):.4f}')
        print(f'預測為違約(1): {int(preds.sum())} 筆 ({preds.mean():.2%}) | '
              f'真實違約(1): {int(y_true.sum())} 筆 ({y_true.mean():.2%})')
        print(f'預測正確: {int(table["correct"].sum())} / {n} '
              f'({table["correct"].mean():.2%})')
        print('平均 Score（依真實標籤分組）: '
              f'正常(0)={probs[y_true == 0].mean():.4f}, '
              f'違約(1)={probs[y_true == 1].mean():.4f}')
        print(f'ROC-AUC: {roc_auc_score(y_true, probs):.4f}')
        if show:
            print(f'\n前 {show} 筆明細:')
            print(table.head(show).to_string(index=False,
                                             float_format=lambda v: f'{v:.4f}'))
    return table


def predict_validation_scores(model, X_scaled, y_true, threshold=0.5,
                              feature_names=None, X_raw=None,
                              show=10, sort_by=None, verbose=True,
                              model_name='MLP'):
    """MLP 對整個 Validation Dataset 輸出 Prediction Score / Probability。

    X_scaled 是已縮放的 validation 特徵矩陣；其餘參數見 build_score_table。
    """
    probs = quiz2.predict_proba(model, X_scaled)
    return build_score_table(probs, y_true, threshold=threshold,
                             feature_names=feature_names, X_raw=X_raw,
                             show=show, sort_by=sort_by, verbose=verbose,
                             model_name=model_name)


def predict_xgb_scores(model, X_scaled, y_true, threshold=0.5,
                       feature_names=None, X_raw=None,
                       show=10, sort_by=None, verbose=True,
                       model_name=XGB_DISPLAY_NAME):
    """XGBoost 對整個 Validation Dataset 輸出 Prediction Score / Probability。

    用 predict_proba(X)[:, 1] 取正類（違約）機率，輸出格式與 MLP 版完全一致，
    兩者才能直接比較。
    """
    probs = xgb_scores(model, X_scaled)
    return build_score_table(probs, y_true, threshold=threshold,
                             feature_names=feature_names, X_raw=X_raw,
                             show=show, sort_by=sort_by, verbose=verbose,
                             model_name=model_name)


def score_summary_by_bin(table, bins=10, verbose=True):
    """把 Prediction Score 切成等寬區間，看每個區間的樣本數與實際違約率。

    分數校準得好的話，區間平均 Score 會貼近該區間的實際違約率。
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    grouped = table.groupby(pd.cut(table['prob'], edges, include_lowest=True),
                            observed=False)
    summary = pd.DataFrame({
        'n': grouped.size(),
        'mean_score': grouped['prob'].mean(),
        'actual_rate': grouped['true'].mean(),
    })
    if verbose:
        print('Prediction Score 分布（區間平均分數 vs. 實際違約率）:')
        print(summary.to_string(float_format=lambda v: f'{v:.4f}'))
    return summary


def plot_score_distribution(table, threshold=0.5, bins=40, figsize=(7, 4.5),
                            title='Validation Prediction Score Distribution'):
    """依真實標籤分色，畫出 Prediction Score 的分布直方圖。"""
    _, ax = plt.subplots(figsize=figsize)
    ax.hist(table.loc[table['true'] == 0, 'prob'], bins=bins, alpha=0.6,
            label='true = 0 (normal)')
    ax.hist(table.loc[table['true'] == 1, 'prob'], bins=bins, alpha=0.6,
            label='true = 1 (default)')
    ax.axvline(threshold, ls='--', c='red', lw=1.2,
               label=f'threshold = {threshold:.3f}')
    ax.set_xlabel('Prediction Score (probability of default)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


# ============================================================
# 2. ROC Curve / AUC
# ============================================================
def plot_roc_curve(y_true, y_probs, title='MLP ROC Curve (Validation)',
                   mark_threshold=None, ax=None, figsize=(6.5, 5.5), verbose=True):
    """畫單一模型的 ROC 曲線（X 軸 FPR、Y 軸 TPR），並把 AUC 數值標示在圖上。

    mark_threshold 有給的話，會在曲線上標出該門檻對應的 (FPR, TPR)。
    回傳 (roc_auc, fpr, tpr, thresholds)。
    """
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    fpr, tpr, ths = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, lw=2.2, color='#1f77b4', label=f'MLP (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#1f77b4')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')

    if mark_threshold is not None:
        i = int(np.argmin(np.abs(ths - mark_threshold)))
        ax.scatter(fpr[i], tpr[i], c='red', zorder=5, s=55,
                   label=f'threshold = {mark_threshold:.3f}\n'
                         f'FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}')

    # 把 AUC 數值直接標在圖上
    ax.text(0.97, 0.06, f'AUC = {roc_auc:.4f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', fc='#fff5cc', ec='#d9a400', lw=1.2))

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(title)
    ax.legend(loc='center right', fontsize=9)
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    if verbose:
        print(f'{title}: AUC = {roc_auc:.4f}')
    return roc_auc, fpr, tpr, ths


def roc_points(fpr, tpr, ths, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5), verbose=True):
    """列出幾個門檻在 ROC 曲線上的 (FPR, TPR)，說明門檻怎麼沿著曲線移動。"""
    rows = []
    for th in thresholds:
        i = int(np.argmin(np.abs(ths - th)))
        rows.append({'threshold': th, 'FPR': fpr[i], 'TPR (Recall)': tpr[i]})
    table = pd.DataFrame(rows)
    if verbose:
        print('ROC 曲線上的門檻對應點:')
        print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    return table


# ============================================================
# 3. 第二個模型：XGBoost
# ============================================================
def xgb_scores(model, X):
    """取得 XGBoost 對正類（違約）的 Prediction Score。"""
    return model.predict_proba(np.asarray(X))[:, 1]


def train_xgboost(X_train, y_train, random_state=42,
                  learning_rate=DEFAULT_LR, n_estimators=DEFAULT_N_ESTIMATORS,
                  early_stopping_rounds=DEFAULT_ES_ROUNDS,
                  es_holdout=DEFAULT_ES_HOLDOUT, eval_set=None,
                  verbose=True, **kwargs):
    """用與 Quiz2 相同的 Training Dataset 訓練 XGBoost（可選早停）。

    learning_rate        : 每棵樹的步伐。越小越穩但需要越多棵樹
    n_estimators         : 樹的數量；開早停時這只是「上限」，實際用幾棵由早停決定
    early_stopping_rounds: 連續幾輪 eval AUC 沒進步就停；None = 不早停，跑滿 n_estimators
    es_holdout           : 早停集比例。從 **Training Dataset** 再分層切一塊出來當早停集，
                           Validation Dataset 保持乾淨（不能拿要評估的那份資料來早停，
                           否則等於偷看答案，AUC 會偏樂觀）
    eval_set             : (X, y)，自備早停集時傳這個，就不從 train 裡切
    kwargs               : 覆寫其他超參數（例如 max_depth=6）

    傳進來的 X_train 請用 Quiz2 縮放後的 X_tr_s_cc；樹模型本來不需要縮放，
    但兩個模型看同一份矩陣，ROC 才比得起來。
    """
    X_all = np.asarray(X_train)
    y_all = np.asarray(y_train).ravel()
    X_fit, y_fit = X_all, y_all
    X_es = y_es = None

    if early_stopping_rounds:
        if eval_set is not None:
            X_es, y_es = np.asarray(eval_set[0]), np.asarray(eval_set[1]).ravel()
        else:
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_all, y_all, test_size=es_holdout, stratify=y_all,
                random_state=random_state)

    # 違約只佔約 22%，用正負樣本比當權重，效果等同其他模型的 class_weight='balanced'
    pos = max(int(y_fit.sum()), 1)
    model = make_xgboost(
        random_state=random_state, scale_pos_weight=(len(y_fit) - pos) / pos,
        learning_rate=learning_rate, n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds if X_es is not None else None,
        **kwargs)

    t0 = time.perf_counter()
    if X_es is not None:
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    else:
        model.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - t0

    n_used = (model.best_iteration + 1) if X_es is not None else n_estimators

    if verbose:
        print(f'=== {XGB_DISPLAY_NAME} ({type(model).__name__}) ===')
        print(f'訓練資料: {X_fit.shape[0]} 筆 × {X_fit.shape[1]} 個特徵', end='')
        if X_es is not None:
            source = '自備' if eval_set is not None else '從 Training Dataset 切出'
            print(f'（早停集 {len(y_es)} 筆，{source}）')
        else:
            print('（與 MLP 同一份）')
        params = {k: v for k, v in model.get_params().items()
                  if k in ('n_estimators', 'max_depth', 'learning_rate', 'subsample',
                           'colsample_bytree', 'min_child_weight', 'reg_lambda',
                           'scale_pos_weight', 'early_stopping_rounds')}
        print('主要超參數:', params)
        if X_es is not None:
            print(f'早停: 第 {model.best_iteration} 輪最佳'
                  f'（eval AUC = {model.best_score:.4f}），'
                  f'實際使用 {n_used} / {n_estimators} 棵樹')
        train_auc = roc_auc_score(y_fit, xgb_scores(model, X_fit))
        print(f'訓練耗時: {elapsed:.1f} 秒 | Training AUC: {train_auc:.4f}')
    return model


def plot_xgb_learning_curve(model, figsize=(7, 4.5),
                            title='XGBoost Early Stopping (eval AUC vs. trees)'):
    """畫早停集的 eval AUC 隨樹數的變化，並標出早停選中的那一輪。

    只有開早停（有給 eval_set）訓練出來的模型才有 evals_result。
    """
    try:
        results = model.evals_result()
    except Exception:                      # xgboost 沒有 eval_set 時會丟 XGBoostError
        results = None
    if not results:
        print('這個模型沒有 eval_set 紀錄（沒開早停），無法畫學習曲線。')
        return None
    metric, curve = next(iter(next(iter(results.values())).items()))

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, len(curve) + 1), curve, lw=1.8, color='#1f77b4',
            label=f'eval {metric}')
    best_i = int(model.best_iteration)
    ax.axvline(best_i + 1, ls='--', c='red', lw=1.2,
               label=f'best iteration = {best_i} ({curve[best_i]:.4f})')
    ax.set_xlabel('Number of trees (boosting rounds)')
    ax.set_ylabel(f'eval {metric}')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return curve


# 學習率 / 早停實驗的預設組合：(標籤, learning_rate, n_estimators, 是否早停)
LR_CONFIGS = (
    ('lr=0.05 n=400 無早停', 0.05, 400, False),
    ('lr=0.02 n=400 無早停', 0.02, 400, False),
    ('lr=0.02 n=1000 無早停', 0.02, 1000, False),
    ('lr=0.05 + 早停', 0.05, 3000, True),
    ('lr=0.02 + 早停', 0.02, 5000, True),
    ('lr=0.01 + 早停', 0.01, 8000, True),
)


def compare_xgb_configs(X_train, y_train, X_val, y_val, configs=LR_CONFIGS,
                        seeds=(0, 1, 42, 7, 2024), es_holdout=DEFAULT_ES_HOLDOUT,
                        verbose=True, **kwargs):
    """同一份資料、多組 learning rate / 早停設定的對照實驗。

    每組設定跑多個 seed（換模型亂數與早停集切法），回報 Validation AUC 的
    平均 ± 標準差。標準差就是「這個差距到底算不算數」的尺規：
    兩組設定的差距若沒有大過標準差，就只是隨機波動。

    train_auc - val_auc（gap）用來看過擬合程度。
    回傳一張 DataFrame，index 是設定標籤。
    """
    y_val = np.asarray(y_val).ravel()
    rows = []
    for label, lr, n_est, use_es in configs:
        vals, gaps, trees, secs = [], [], [], []
        for seed in seeds:
            t0 = time.perf_counter()
            model = train_xgboost(
                X_train, y_train, random_state=seed, learning_rate=lr,
                n_estimators=n_est,
                early_stopping_rounds=DEFAULT_ES_ROUNDS if use_es else None,
                es_holdout=es_holdout, verbose=False, **kwargs)
            secs.append(time.perf_counter() - t0)
            # gap 要跟模型真正 fit 的那份資料比，早停時是 85% 的 train
            X_seen, y_seen = np.asarray(X_train), np.asarray(y_train).ravel()
            if use_es:
                X_seen, _, y_seen, _ = train_test_split(
                    X_seen, y_seen, test_size=es_holdout, stratify=y_seen,
                    random_state=seed)
            val_auc = roc_auc_score(y_val, xgb_scores(model, X_val))
            vals.append(val_auc)
            gaps.append(roc_auc_score(y_seen, xgb_scores(model, X_seen)) - val_auc)
            trees.append((model.best_iteration + 1) if use_es else n_est)
        rows.append({
            'config': label, 'lr': lr, 'early_stop': use_es,
            'trees': int(np.mean(trees)),
            'val_AUC_mean': np.mean(vals), 'val_AUC_std': np.std(vals),
            'val_AUC_min': np.min(vals), 'val_AUC_max': np.max(vals),
            'train_val_gap': np.mean(gaps), 'sec': np.mean(secs),
        })
        if verbose:
            r = rows[-1]
            print(f'{label:<24s} trees~{r["trees"]:>5d}  '
                  f'val AUC = {r["val_AUC_mean"]:.4f} ± {r["val_AUC_std"]:.4f} '
                  f'[{r["val_AUC_min"]:.4f}, {r["val_AUC_max"]:.4f}]  '
                  f'gap = {r["train_val_gap"]:.4f}  {r["sec"]:.1f}s')

    table = pd.DataFrame(rows).set_index('config')
    if verbose:
        print(f'\n=== {len(seeds)} 個 seed 平均（依 val AUC 排序）===')
        print(table.sort_values('val_AUC_mean', ascending=False)
              .to_string(float_format=lambda v: f'{v:.4f}'))
    return table


def plot_xgb_configs(table, figsize=(9.5, 5)):
    """把 compare_xgb_configs 的結果畫成 Validation AUC（含 ±std 誤差棒）+ 過擬合 gap。"""
    order = table.sort_values('val_AUC_mean', ascending=False)
    x = np.arange(len(order))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = ['#d62728' if es else '#1f77b4' for es in order['early_stop']]
    ax1.bar(x, order['val_AUC_mean'], yerr=order['val_AUC_std'], capsize=4,
            color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order.index, rotation=30, ha='right', fontsize=8)
    lo = (order['val_AUC_mean'] - order['val_AUC_std']).min()
    hi = (order['val_AUC_mean'] + order['val_AUC_std']).max()
    ax1.set_ylim(lo - 0.002, hi + 0.002)          # 差距很小，放大 y 軸才看得出來
    ax1.set_ylabel('Validation AUC (mean ± std)')
    ax1.set_title('Validation AUC（紅=有早停）')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, order['train_val_gap'], color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order.index, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('train AUC - val AUC')
    ax2.set_title('過擬合程度（越低越好）')
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    plt.show()


def feature_importance(model, feature_names, top_k=15, plot=True, figsize=(7, 5)):
    """列出 XGBoost 的 feature_importances_（預設是 gain 的相對比例）。"""
    scores = np.ravel(model.feature_importances_)
    label = 'Feature Importance'

    table = (pd.DataFrame({'feature': list(feature_names), label: scores})
             .sort_values(label, ascending=False).head(top_k))
    if plot:
        _, ax = plt.subplots(figsize=figsize)
        ax.barh(table['feature'][::-1], table[label][::-1], color='#2ca02c')
        ax.set_xlabel(label)
        ax.set_title(f'{type(model).__name__} - Top {top_k} {label}')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    return table


# ============================================================
# 4. 兩模型 ROC 疊圖比較
# ============================================================
def plot_roc_comparison(y_true, prob_dict, mark_thresholds=None,
                        title='Validation ROC: MLP vs. XGBoost',
                        figsize=(7.5, 6.5), verbose=True):
    """把 MLP 與 XGBoost 的 ROC 曲線畫在同一張圖，並把各自的 AUC 標示於圖中。

    y_true     : 同一份 Validation 真實標籤
    prob_dict  : {模型名稱: 該模型對 validation 的 Prediction Score}
    mark_thresholds: {模型名稱: 門檻}，會在對應曲線上標出該門檻的點
    回傳：{模型名稱: AUC} 的 DataFrame
    """
    y_true = np.asarray(y_true).ravel()
    mark_thresholds = mark_thresholds or {}
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    _, ax = plt.subplots(figsize=figsize)
    rows = []
    for i, (name, probs) in enumerate(prob_dict.items()):
        fpr, tpr, ths = roc_curve(y_true, np.asarray(probs).ravel())
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f'{name} (AUC = {roc_auc:.4f})')

        th = mark_thresholds.get(name)
        if th is not None:
            j = int(np.argmin(np.abs(ths - th)))
            ax.scatter(fpr[j], tpr[j], c=color, edgecolors='black', zorder=5, s=60)
            ax.annotate(f'th={th:.2f}', (fpr[j], tpr[j]), textcoords='offset points',
                        xytext=(8, -12), fontsize=8, color=color)
        rows.append({'model': name, 'AUC': roc_auc})

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')

    # 把兩個模型的 AUC 數值直接標在圖中
    text = '\n'.join(f'{r["model"]}: AUC = {r["AUC"]:.4f}'
                     for r in sorted(rows, key=lambda r: -r['AUC']))
    ax.text(0.97, 0.06, text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', fc='#fff5cc', ec='#d9a400', lw=1.2))

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(title)
    ax.legend(loc='center right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    table = pd.DataFrame(rows).set_index('model')
    if verbose:
        print('各模型 AUC（同一份 Validation Dataset）:')
        print(table.to_string(float_format=lambda v: f'{v:.4f}'))
        best = table['AUC'].idxmax()
        print(f'AUC 較高者: {best} ({table.loc[best, "AUC"]:.4f})')
    return table


# ============================================================
# 5. 兩模型比較：Prediction Score、指標表
# ============================================================
def collect_val_scores(xgb_model, X_val, y_val, mlp_model=None,
                       mlp_name=MLP_DISPLAY_NAME, xgb_name=XGB_DISPLAY_NAME,
                       verbose=False):
    """取得兩個模型對同一份 Validation Dataset 的 Prediction Score。

    回傳：{顯示名稱: 機率陣列 (n_samples,)}，有給 mlp_model 時 MLP 排在前面。
    """
    probs = {}
    if mlp_model is not None:
        probs[mlp_name] = quiz2.predict_proba(mlp_model, X_val).ravel()
    probs[xgb_name] = xgb_scores(xgb_model, X_val)
    if verbose:
        y = np.asarray(y_val).ravel()
        for name, p in probs.items():
            print(f'{name:<20s} AUC = {roc_auc_score(y, p):.4f}')
    return probs


def leaderboard(y_true, prob_dict, thresholds=None, default_threshold=0.5,
                sort_by='AUC', verbose=True):
    """把兩個模型在同一份 Validation Dataset 上的指標排成一張表。

    thresholds: {模型名稱: 門檻}，沒給的模型用 default_threshold。
    AUC / AP 與門檻無關；Accuracy / Precision / Recall / F1 會受門檻影響。
    """
    y_true = np.asarray(y_true).ravel()
    thresholds = thresholds or {}
    rows = []
    for name, probs in prob_dict.items():
        probs = np.asarray(probs).ravel()
        th = thresholds.get(name, default_threshold)
        pred = (probs >= th).astype(int)
        rows.append({
            'model': name,
            'AUC': roc_auc_score(y_true, probs),
            'AP': average_precision_score(y_true, probs),
            'threshold': th,
            'Accuracy': accuracy_score(y_true, pred),
            'Precision': precision_score(y_true, pred, zero_division=0),
            'Recall': recall_score(y_true, pred, zero_division=0),
            'F1': f1_score(y_true, pred, zero_division=0),
        })
    table = pd.DataFrame(rows).set_index('model').sort_values(sort_by, ascending=False)
    if verbose:
        print('=== Validation Dataset 模型比較（同一份資料、同一組特徵）===')
        print(table.to_string(float_format=lambda v: f'{v:.4f}'))
        best = table.index[0]
        print(f'\n{sort_by} 最高: {best} ({table.loc[best, sort_by]:.4f})')
    return table


def plot_leaderboard(table, metrics=('AUC', 'F1', 'Recall', 'Precision', 'Accuracy'),
                     figsize=(9, 5)):
    """把指標表畫成分組長條圖，一眼看出兩個模型在各指標上的高低。"""
    metrics = [m for m in metrics if m in table.columns]
    x = np.arange(len(metrics))
    width = 0.8 / len(table)

    _, ax = plt.subplots(figsize=figsize)
    for i, (name, row) in enumerate(table.iterrows()):
        offset = (i - (len(table) - 1) / 2) * width
        bars = ax.bar(x + offset, [row[m] for m in metrics], width, label=name)
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics: MLP vs. XGBoost')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_comparison(X_train, y_train, X_val, y_val, mlp_model=None,
                   random_state=42, mlp_threshold=0.5, default_threshold=0.5,
                   plot=True, verbose=True, **xgb_kwargs):
    """一鍵流程：訓練 XGBoost → 取 Validation Prediction Score → ROC 疊圖 → 指標表。

    回傳 (xgb_model, prob_dict, table)。
    """
    model = train_xgboost(X_train, y_train, random_state=random_state,
                          verbose=verbose, **xgb_kwargs)
    prob_dict = collect_val_scores(model, X_val, y_val, mlp_model=mlp_model)

    marks = {XGB_DISPLAY_NAME: default_threshold}
    thresholds = None
    if mlp_model is not None:
        thresholds = {MLP_DISPLAY_NAME: mlp_threshold}
        marks[MLP_DISPLAY_NAME] = mlp_threshold

    if plot:
        plot_roc_comparison(y_val, prob_dict, mark_thresholds=marks,
                            figsize=(8, 6.8), verbose=False)
    table = leaderboard(y_val, prob_dict, thresholds=thresholds,
                        default_threshold=default_threshold, verbose=verbose)
    if plot:
        plot_leaderboard(table)
    return model, prob_dict, table
