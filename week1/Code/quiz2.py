import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# BT.601 灰階加權係數（與 quiz1 相同）
_BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def log_run(name, ms, gray, records):
    records.append({'method': name, 'ms': ms, 'shape': gray.shape, 'mean': gray.mean()})
    print(f'{name:<6s} 執行時間: {ms:8.3f} ms | 尺寸: {gray.shape} | 平均灰階: {gray.mean():.2f}')


def _check_path(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'找不到影像檔: {os.path.abspath(image_path)}')
    return image_path


def _to_gray_numpy(img_rgb):
    """用 BT.601 加權把 RGB 轉灰階（uint8）。"""
    gray_float = np.dot(img_rgb.astype(np.float32), _BT601)
    return np.clip(np.round(gray_float), 0, 255).astype(np.uint8)


def _load_rgb(image_path):
    """用 matplotlib 讀圖並正規化成 uint8 的 3 通道 RGB。"""
    _check_path(image_path)
    img_rgb = plt.imread(image_path)
    if img_rgb.ndim == 2:                                # 本來就是灰階圖
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    if img_rgb.shape[2] == 4:                            # 去掉 alpha 通道
        img_rgb = img_rgb[:, :, :3]
    if img_rgb.dtype != np.uint8:                        # PNG 會讀成 0~1 float
        img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    return img_rgb


def hist_cdf(gray):
    """回傳 (hist, cdf)：長度 256 的直方圖與其累積分布。"""
    hist = np.bincount(gray.ravel(), minlength=256)
    return hist, hist.cumsum()


def equalize_lut(gray):
    """依直方圖等化公式算出 256 階的查表 (LUT)。

    s_k = round( (cdf(k) - cdf_min) / (N - cdf_min) * 255 )
    """
    hist, cdf = hist_cdf(gray)
    nonzero = cdf[cdf > 0]
    cdf_min = nonzero[0] if nonzero.size else 0
    denom = gray.size - cdf_min
    if denom <= 0:                                       # 整張圖只有單一灰階值
        return np.zeros(256, dtype=np.uint8)
    lut = np.round((cdf - cdf_min) / denom * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _plot_pair(img_rgb, gray, equalized, title):
    """畫出「原圖 / 原灰階 / 等化後」與對應直方圖 + CDF。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Origin')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Gray (before)')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(equalized, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'{title} equalized')
    axes[0, 2].axis('off')

    axes[1, 0].axis('off')
    for ax, data, name in ((axes[1, 1], gray, 'Before'),
                           (axes[1, 2], equalized, f'After ({title})')):
        hist, cdf = hist_cdf(data)
        ax.bar(np.arange(256), hist, width=1.0, color='steelblue')
        ax.set_title(f'Histogram - {name}')
        ax.set_xlim(0, 255)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Pixel count')
        ax_cdf = ax.twinx()                              # 右軸疊上正規化 CDF
        ax_cdf.plot(np.arange(256), cdf / cdf[-1], color='orangered', linewidth=1.5)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.set_ylabel('Normalized CDF')

    plt.tight_layout()
    plt.show()


def Numpy_Histogram(image_path, records, show=True, gray=None):
    """用 NumPy 實作 Histogram Equalization，回傳 (img_rgb, gray, eq_numpy)。"""
    img_rgb = _load_rgb(image_path)
    if gray is None:
        gray = _to_gray_numpy(img_rgb)

    t0 = time.perf_counter()
    lut = equalize_lut(gray)                             # 建立累積分布查表
    eq_numpy = lut[gray]                                 # 查表映射
    numpy_ms = (time.perf_counter() - t0) * 1000
    log_run('NumPy', numpy_ms, eq_numpy, records)

    if show:
        _plot_pair(img_rgb, gray, eq_numpy, 'NumPy')
    return img_rgb, gray, eq_numpy


def OpenCV_Histogram(image_path, records, show=True, gray=None):
    """用 OpenCV 的 cv2.equalizeHist 做等化，回傳 (img_rgb_cv, gray_cv, eq_cv)。"""
    _check_path(image_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f'OpenCV 無法讀取影像 (路徑含非 ASCII 字元或格式不支援?): {image_path}')
    img_rgb_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if gray is None else gray

    t0 = time.perf_counter()
    eq_cv = cv2.equalizeHist(gray_cv)
    opencv_ms = (time.perf_counter() - t0) * 1000
    log_run('OpenCV', opencv_ms, eq_cv, records)

    if show:
        _plot_pair(img_rgb_cv, gray_cv, eq_cv, 'OpenCV')
    return img_rgb_cv, gray_cv, eq_cv


def analyze_records(records, Histogram_numpy, Histogram_cv, img_rgb):
    """列出歷次執行紀錄，並比較兩種等化結果的差異。"""
    if not records:
        print('尚無執行紀錄，請先執行 Numpy_Histogram / OpenCV_Histogram。')
        return

    print('=== 執行紀錄 ===')
    for i, r in enumerate(records, 1):
        print(f"{i:2d}. {r['method']:<6s} {r['ms']:8.3f} ms | 平均灰階: {r['mean']:.2f}")

    print('\n=== 各方法平均 ===')
    for name in ('NumPy', 'OpenCV'):
        ms_list = [r['ms'] for r in records if r['method'] == name]
        if ms_list:
            print(f'{name:<6s} {sum(ms_list) / len(ms_list):8.3f} ms (共 {len(ms_list)} 次)')

    if Histogram_numpy.shape != Histogram_cv.shape:
        raise ValueError(f'兩張結果尺寸不同: {Histogram_numpy.shape} vs {Histogram_cv.shape}')

    gray = _to_gray_numpy(img_rgb)                       # 等化前的灰階，作為對照基準
    diff = np.abs(Histogram_numpy.astype(np.int16) - Histogram_cv.astype(np.int16))
    print(f'\n最大差異: {diff.max()} | 平均差異: {diff.mean():.4f} '
          f'| 不同的像素: {np.count_nonzero(diff)} / {diff.size}')
    print(f'標準差 (等化前 → 後): {gray.std():.2f} → {Histogram_numpy.std():.2f}')

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Origin')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Gray (before)')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(Histogram_numpy, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('NumPy equalized')
    axes[0, 2].axis('off')
    axes[0, 3].imshow(Histogram_cv, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title('OpenCV equalized')
    axes[0, 3].axis('off')

    for ax, data, name in ((axes[1, 0], gray, 'Before'),
                           (axes[1, 1], Histogram_numpy, 'NumPy'),
                           (axes[1, 2], Histogram_cv, 'OpenCV')):
        hist, cdf = hist_cdf(data)
        ax.bar(np.arange(256), hist, width=1.0, color='steelblue')
        ax.set_title(f'Histogram - {name}')
        ax.set_xlim(0, 255)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Pixel count')
        ax_cdf = ax.twinx()
        ax_cdf.plot(np.arange(256), cdf / cdf[-1], color='orangered', linewidth=1.5)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.set_ylabel('Normalized CDF')

    im = axes[1, 3].imshow(diff, cmap='hot', vmin=0, vmax=5)
    axes[1, 3].set_title('Difference (NumPy vs OpenCV)')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
