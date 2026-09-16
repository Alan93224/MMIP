# Quiz 1

分類主題： 二分類貓狗辨識

資料集： 各 12500 張貓狗圖片，訓練集、驗證集、測試集比例 7：2：1

使用機器學習模型： RandomForest、LogisticRegression

## Feature Scaling：

    訓練集縮放後 mean≈0, std≈1: 0.0 1.0
    驗證集縮放後 mean/std: 0.0193 1.0042

## 5-Fold 交叉驗證：

    LogisticRegression   Acc=0.6208±0.0113 | F1=0.6073±0.0119 | AUC=0.6688±0.0143
    RandomForest         Acc=0.6371±0.0064 | F1=0.6319±0.0048 | AUC=0.6905±0.0093

## Classification Threshold：

預設都是 0.5，模型訓練完再跑 find_best_threshold 函式看哪個 Threshold 的成效是最好的

RandomForest Threshold == 0.5：

    Accuracy : 0.6456
    Precision: 0.6481
    Recall   : 0.6372
    F1-Score : 0.6426

RandomForest Threshold == 0.3710 (best threshold)：

    Accuracy : 0.6050  
    Precision: 0.5673
    Recall   : 0.8852
    F1-Score : 0.6915

<img src="data/threshold_curves1.png" width="600" alt="Threshold_Curves1">

<img src="data/evaluation_curves1.png" width="600" alt="evaluation_curves1">

<img src="data/confusion_matrix1.png" width="600" alt="confusion_matrix1">

Test：

<img src="data/test_curves.png" width="600" alt="test_curves">

<img src="data/test_confusion_matrix.png" width="600" alt="test_confusion_matrix">

LogisticRegression Threshold == 0.5：

    Accuracy : 0.6256
    Precision: 0.6333
    Recall   : 0.5968
    F1-Score : 0.6145 

LogisticRegression Threshold == 0.3560 (best threshold)：

    Accuracy : 0.5846  
    Precision: 0.5517
    Recall   : 0.9032
    F1-Score : 0.6850

<img src="data/threshold_curves2.png" width="600" alt="threshold_curves2">

<img src="data/confusion_matrix2.png" width="600" alt="confusion_matrix2">

## 結果討論：

<img src="data/error_type1.png" width="600" alt="error_type1">

<img src="data/error_type2.png" width="600" alt="error_type2">

1. **Threshold 0.5 時，兩個模型都是 Precision > Recall**：預測為 dog 的樣本大約六成正確，但真正的 dog 只抓到 56~61%。此時**較多的錯誤是 FN（dog 被判成 cat）**，RandomForest FN=156 > FP=128，LogisticRegression FN=175 > FP=127。原因是模型輸出的機率整體偏低（偏向預測 cat），0.5 這條線切得太嚴格。

2. **調低 Threshold 後兩者都反轉成 Recall >> Precision**：因為 F1 最佳化會 Threshold 往下推（RF 0.407、LR 0.369），更多樣本被判成 dog，Recall 從 0.61→0.81、0.56→0.84，但誤抓的 cat 變多，Precision 下降到 0.60、0.57。此時**較多的錯誤變成 FP（cat 被判成 dog）**，RF FP=216 > FN=76，LR FP=250 > FN=64。這正是 Precision–Recall 的取捨：Threshold 只是在移動兩類錯誤的分配比例。

3. **兩模型之間的差異**：LogisticRegression 的最佳 Threshold 更低、機率分布更集中，因此走得比 RandomForest 更「極端」——Recall 最高（0.84）但 Precision 最低（0.5734），FP 也最多（250）。RandomForest 在兩者間較平衡，Accuracy、Precision、F1（0.6894 vs 0.6815）與 ROC-AUC（0.685 vs 0.679）皆略勝，因為它能捕捉特徵間的非線性關係，而 LogisticRegression 只能畫線性決策邊界。

4. **選擇建議**：若應用上「漏掉 dog」成本較高（例如要盡量找出所有 dog 再人工複核），選 LogisticRegression + 低 Threshold 的高 Recall 設定；若要兼顧判對率，選 RandomForest + Threshold 0.407。整體兩模型絕對表現都只有 0.6 上下，瓶頸在於目前的色彩/紋理統計特徵區分力有限，而非 Threshold 設定。

# Quiz 2

