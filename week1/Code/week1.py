import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Quiz 1
def log_run(name, ms, gray, records):
    records.append({'method': name, 'ms': ms, 'shape': gray.shape, 'mean': gray.mean()})
    print(f'{name:<6s} 執行時間: {ms:8.3f} ms | 尺寸: {gray.shape} | 平均灰階: {gray.mean():.2f}')


def _check_path(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'找不到影像檔: {os.path.abspath(image_path)}')
    return image_path


def numpy_gray(image_path, records, show=True):
    """用 NumPy 做 BT.601 加權灰階轉換，回傳 (img_rgb, gray_numpy)。"""
    _check_path(image_path)
    img_rgb = plt.imread(image_path)
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:      # 去掉 alpha 通道
        img_rgb = img_rgb[:, :, :3]
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f'預期 3 通道彩色影像，實際 shape={img_rgb.shape}')
    if img_rgb.dtype != np.uint8:                        # PNG 會讀成 0~1 float
        img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    t0 = time.perf_counter()
    gray_float = np.dot(img_rgb.astype(np.float32), weights)
    gray_numpy = np.clip(np.round(gray_float), 0, 255).astype(np.uint8)
    numpy_ms = (time.perf_counter() - t0) * 1000
    log_run('NumPy', numpy_ms, gray_numpy, records)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Origin')
        axes[0].axis('off')
        axes[1].imshow(gray_numpy, cmap='gray')
        axes[1].set_title('NumPy gray')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    return img_rgb, gray_numpy


def opencv_gray(image_path, records, show=True):
    """用 OpenCV 做灰階轉換，回傳 (img_rgb_cv, gray_cv)。"""
    _check_path(image_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f'OpenCV 無法讀取影像 (路徑含非 ASCII 字元或格式不支援?): {image_path}')

    t0 = time.perf_counter()
    gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    opencv_ms = (time.perf_counter() - t0) * 1000
    log_run('OpenCV', opencv_ms, gray_cv, records)

    img_rgb_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_rgb_cv)
        axes[0].set_title('Origin')
        axes[0].axis('off')
        axes[1].imshow(gray_cv, cmap='gray')
        axes[1].set_title('OpenCV gray')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    return img_rgb_cv, gray_cv


def analyze_records(records, gray_numpy, gray_cv, img_rgb):
    """列出歷次執行紀錄，並比較兩種方法的轉換結果。"""
    if not records:
        print('尚無執行紀錄，請先執行 numpy_gray / opencv_gray。')
        return

    print('=== 執行紀錄 ===')
    for i, r in enumerate(records, 1):
        print(f"{i:2d}. {r['method']:<6s} {r['ms']:8.3f} ms | 平均灰階: {r['mean']:.2f}")

    print('\n=== 各方法平均 ===')
    for name in ('NumPy', 'OpenCV'):
        ms_list = [r['ms'] for r in records if r['method'] == name]
        if ms_list:
            print(f'{name:<6s} {sum(ms_list) / len(ms_list):8.3f} ms (共 {len(ms_list)} 次)')

    if gray_numpy.shape != gray_cv.shape:
        raise ValueError(f'兩張灰階圖尺寸不同: {gray_numpy.shape} vs {gray_cv.shape}')

    diff = np.abs(gray_numpy.astype(np.int16) - gray_cv.astype(np.int16))
    print(f'\n最大差異: {diff.max()} | 平均差異: {diff.mean():.4f}')

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Origin')
    axes[0].axis('off')
    axes[1].imshow(gray_numpy, cmap='gray')
    axes[1].set_title('NumPy gray')
    axes[1].axis('off')
    axes[2].imshow(gray_cv, cmap='gray')
    axes[2].set_title('OpenCV gray')
    axes[2].axis('off')
    im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=5)
    axes[3].set_title('Difference')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

#Quiz 2
