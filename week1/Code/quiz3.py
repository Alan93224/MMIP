import glob
import os
import re
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 紙張偵測參數：在縮小的影像上找四角，加速並降低紙面紋理/文字的干擾
DETECT_LONG_SIDE = 1000
MIN_AREA_RATIO = 0.02   # 紙張至少佔畫面 2%
FILL_OK = 0.95          # 輪廓面積 / 四邊形面積 超過此值即視為找到紙張
FALLBACK_F35 = 26.0     # 無 EXIF 且無法由角點估計焦距時，假設的 35mm 等效焦距 (手機主鏡頭)


def log_run(name, ms, gray, records, file=''):
    records.append({'method': name, 'file': file, 'ms': ms, 'shape': gray.shape, 'mean': gray.mean()})
    print(f'{name:<6s} {file:<14s} 執行時間: {ms:8.3f} ms | 尺寸: {gray.shape} | 平均灰階: {gray.mean():.2f}')


def _check_path(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'找不到影像檔: {os.path.abspath(image_path)}')
    return image_path


def find_images(data_dir, prefix='quiz3_', exts=('.jpg', '.jpeg', '.png')):
    """找出 data_dir 下以 prefix 開頭的 jpg/png（排除 *_output 輸出檔），依檔名中的數字排序。"""
    paths = [p for p in glob.glob(os.path.join(data_dir, f'{prefix}*'))
             if os.path.splitext(p)[1].lower() in exts
             and '_output' not in os.path.basename(p).lower()]
    natural = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(p))]
    return sorted(paths, key=natural)


def _order_corners(pts):
    """依繞中心的角度排序，再以 x+y 最小者為起點：左上、右上、右下、左下。"""
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()))]
    return np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)


def _approx_quad(contour):
    """把輪廓的凸包逐步放寬 epsilon 逼近成四邊形，失敗回傳 None。"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 30):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def detect_paper(img_bgr, long_side=DETECT_LONG_SIDE):
    """自動偵測紙張四角，回傳 float32 (4, 2)，順序：左上、右上、右下、左下。

    1. 縮小、灰階、高斯模糊
    2. 從 Otsu 門檻開始逐步提高門檻二值化（紙比背景亮），閉運算填掉文字、開運算去雜點
    3. 取最大輪廓逼近四邊形；背景亮處與紙黏在一起時四邊形填滿率會偏低，就繼續提高門檻
    4. 取填滿率最接近 1 的四邊形，座標放大回原圖
    """
    scale = long_side / max(img_bgr.shape[:2])
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k, open_k = np.ones((15, 15), np.uint8), np.ones((9, 9), np.uint8)

    best_fill, best_quad = None, None
    for t in range(int(otsu), 250, 5):
        _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_k), cv2.MORPH_OPEN, open_k)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            break
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_AREA_RATIO * gray.size:
            break
        quad = _approx_quad(c)
        if quad is None:
            continue
        fill = area / cv2.contourArea(quad)
        if best_fill is None or abs(1 - fill) < abs(1 - best_fill):
            best_fill, best_quad = fill, quad
        if fill > FILL_OK:
            break

    if best_quad is None:
        raise ValueError('偵測不到紙張四邊形')
    return _order_corners(best_quad.astype(np.float32) / scale)


def _exif_focal_px(image_path, w, h):
    """由 EXIF 的 35mm 等效焦距換算成像素焦距；沒有 EXIF 回傳 None。"""
    try:
        f35 = Image.open(image_path).getexif().get_ifd(0x8769).get(41989)  # FocalLengthIn35mmFilm
    except Exception:
        return None
    return f35 / np.hypot(36, 24) * np.hypot(w, h) if f35 else None


def _zhang_vectors(quad):
    """Zhang & He (2007) whiteboard：回傳 n2、n3（對應矩形寬、高兩個方向）。"""
    m1, m2, m4, m3 = [np.array([*p, 1.0]) for p in quad]  # m1 左上 m2 右上 m3 左下 m4 右下
    k2 = np.cross(m1, m4) @ m3 / (np.cross(m2, m4) @ m3)
    k3 = np.cross(m1, m4) @ m2 / (np.cross(m3, m4) @ m2)
    return k2 * m2 - m1, k3 * m3 - m1


def estimate_focal(quad, w, h):
    """由四角點估計焦距（像素），正面拍攝等退化情況回傳 None。"""
    n2, n3 = _zhang_vectors(quad)
    u0, v0 = w / 2, h / 2
    if abs(n2[2]) < 1e-6 or abs(n3[2]) < 1e-6:
        return None
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2)
           + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)) / (n2[2] * n3[2])
    return np.sqrt(f2) if f2 > 0 else None


def estimate_aspect(quad, w, h, f):
    """由四角點與焦距估計紙張真實的 寬/高。"""
    n2, n3 = _zhang_vectors(quad)
    K_inv = np.linalg.inv(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
    a, b = K_inv @ n2, K_inv @ n3
    return float(np.sqrt((a @ a) / (b @ b)))


def paper_dst_points(src_pts, aspect):
    """目標長方形：高取來源左右邊較長者（保留解析度），寬 = 高 × aspect。"""
    out_h = max(np.linalg.norm(src_pts[3] - src_pts[0]), np.linalg.norm(src_pts[2] - src_pts[1]))
    out_w = out_h * aspect
    return np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])


def _draw_quad(ax, pts, color):
    closed = np.vstack([pts, pts[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=30, zorder=3)


def _plot_pair(img_rgb, warped, src_pts, dst_pts, title):
    """左：原圖與來源四邊形；右：校正後與目標長方形。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    _draw_quad(axes[0], src_pts, 'red')
    axes[0].set_title('Origin (src points)')
    axes[0].axis('off')
    axes[1].imshow(warped)
    _draw_quad(axes[1], dst_pts, 'lime')
    axes[1].set_title(f'{title} perspective')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def OpenCV_Perspective(image_path, records, src_pts=None, dst_pts=None, aspect=None, focal=None, show=True):
    """用 cv2.getPerspectiveTransform + cv2.warpPerspective 把紙張校正成正面的長方形。

    src_pts 為 None 時自動偵測紙張四角；dst_pts 為 None 時依真實長寬比產生目標長方形。
    長寬比 (寬/高) 優先序：aspect 參數 > 由焦距估計；焦距優先序：focal 參數 > EXIF > 由角點估計 > 預設值。
    已知紙張規格時可直接給 aspect，例如 A4 橫放 np.sqrt(2)、直放 1/np.sqrt(2)。
    回傳 dict：file, img_rgb, H, warp, src_pts, dst_pts, aspect, focal。
    """
    _check_path(image_path)
    name = os.path.basename(image_path)
    img_bgr = cv2.imread(image_path)  # 會依 EXIF Orientation 自動轉正
    if img_bgr is None:
        raise ValueError(f'OpenCV 無法讀取影像 (路徑含非 ASCII 字元或格式不支援?): {image_path}')
    img_rgb_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    if src_pts is None:
        t0 = time.perf_counter()
        src_pts = detect_paper(img_bgr)
        print(f'{name} 自動偵測紙張四角: {(time.perf_counter() - t0) * 1000:8.3f} ms')
    src_pts = np.float32(src_pts)

    if dst_pts is None:
        if aspect is None:
            if focal is None:
                focal, source = _exif_focal_px(image_path, w, h), 'EXIF'
            else:
                source = '參數'
            if focal is None:
                focal, source = estimate_focal(src_pts, w, h), '角點估計'
            if focal is None:
                focal, source = FALLBACK_F35 / np.hypot(36, 24) * np.hypot(w, h), '預設值'
            aspect = estimate_aspect(src_pts, w, h, focal)
            print(f'{name} 焦距 {focal:.0f} px ({source}) -> 估計寬/高 = {aspect:.3f}')
        dst_pts = paper_dst_points(src_pts, aspect)
    dst_pts = np.float32(dst_pts)
    out_size = (int(round(dst_pts[:, 0].max())), int(round(dst_pts[:, 1].max())))

    t0 = time.perf_counter()
    H_cv = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp_cv = cv2.warpPerspective(img_rgb_cv, H_cv, out_size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    opencv_ms = (time.perf_counter() - t0) * 1000
    log_run('OpenCV', opencv_ms, warp_cv, records, name)

    if show:
        _plot_pair(img_rgb_cv, warp_cv, src_pts, dst_pts, f'{name} OpenCV')
    return {'file': name, 'img_rgb': img_rgb_cv, 'H': H_cv, 'warp': warp_cv,
            'src_pts': src_pts, 'dst_pts': dst_pts, 'aspect': aspect, 'focal': focal}
