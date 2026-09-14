import glob
import os
import re
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 前處理參數
WORK_LONG_SIDE = 1600     # 拼接前把影像縮到長邊 1600 px：手機原圖太大，SIFT 會很慢且雜訊特徵多
CLAHE_CLIP = 2.0          # CLAHE 對比限制，越大對比越強但雜訊也越明顯
CLAHE_TILE = (8, 8)       # CLAHE 分塊數
BLUR_KSIZE = (3, 3)       # 輕微高斯去雜訊

# 特徵與匹配參數
SIFT_FEATURES = 5000      # SIFT 最多保留的特徵點數
RATIO = 0.75              # Lowe ratio test 門檻
RANSAC_THRESH = 4.0       # 重投影誤差門檻 (px)
MIN_INLIERS = 15          # 內點數少於此值視為拼接失敗


def log_run(name, ms, records, file='', **info):
    records.append({'step': name, 'file': file, 'ms': ms, **info})
    extra = ' | '.join(f'{k}: {v}' for k, v in info.items())
    print(f'{name:<10s} 執行時間: {ms:9.3f} ms' + (f' | {extra}' if extra else ''))


def _check_path(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'找不到影像檔: {os.path.abspath(image_path)}')
    return image_path


def find_images(data_dir, prefix='quiz4_', exts=('.jpg', '.jpeg', '.png')):
    """找出 data_dir 下以 prefix 開頭的 jpg/png（排除 *_output 輸出檔），依檔名中的數字排序（10 排在 9 後面）。"""
    paths = [p for p in glob.glob(os.path.join(data_dir, f'{prefix}*'))
             if os.path.splitext(p)[1].lower() in exts
             and '_output' not in os.path.basename(p).lower()]
    natural = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(p))]
    return sorted(paths, key=natural)


def pair_images(paths):
    """依序兩兩分組：(1, 2)、(3, 4)…；張數為奇數時最後一張沒有配對，印出提示並略過。"""
    if len(paths) % 2:
        print(f'影像數為奇數，最後一張 {os.path.basename(paths[-1])} 沒有配對，略過')
    return list(zip(paths[0::2], paths[1::2]))


def load_image(image_path, long_side=WORK_LONG_SIDE):
    """讀取影像（依 EXIF 自動轉正）並等比例縮到長邊 long_side，回傳 BGR。"""
    _check_path(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'OpenCV 無法讀取影像 (路徑含非 ASCII 字元或格式不支援?): {image_path}')
    scale = long_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def preprocess(img_bgr):
    """特徵偵測前的前處理，回傳給 SIFT 用的灰階影像。

    1. 轉灰階：SIFT 只看亮度梯度
    2. 高斯模糊 3x3：壓掉感測器雜訊與 JPEG 塊狀雜訊，減少不穩定的小特徵點
    3. CLAHE（限制對比的自適應直方圖等化）：兩張照片曝光/白平衡不同時，
       局部對比被拉到相近，陰影與亮部也能偵測到特徵，匹配更穩定
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(gray)


def detect_sift(gray, root_sift=True):
    """SIFT 偵測特徵點與描述子。root_sift=True 時轉成 RootSIFT（L1 正規化後開根號），
    等同以 Hellinger 距離比較直方圖，對光照變化更穩健，匹配正確率較高。"""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) < 4:
        raise ValueError('特徵點太少，無法拼接')
    if root_sift:
        des = des / (np.abs(des).sum(axis=1, keepdims=True) + 1e-7)
        des = np.sqrt(des).astype(np.float32)
    return kps, des


def match_features(des1, des2, ratio=RATIO):
    """FLANN KD-tree 做 kNN (k=2) 匹配，再做兩道篩選：
    1. Lowe ratio test：最近鄰距離 < ratio × 次近鄰距離，排除模稜兩可的匹配
    2. 雙向一致 (cross check)：1→2 與 2→1 必須互為最佳匹配
    """
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def ratio_test(a, b):
        good = {}
        for pair in flann.knnMatch(a, b, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                good[pair[0].queryIdx] = pair[0]
        return good

    fwd, bwd = ratio_test(des1, des2), ratio_test(des2, des1)
    return [m for q, m in fwd.items() if m.trainIdx in bwd and bwd[m.trainIdx].trainIdx == q]


def find_homography(kps1, kps2, matches):
    """由匹配點估計把影像 2 映射到影像 1 平面的 H，用 USAC_MAGSAC（穩健版 RANSAC）排除錯誤匹配。"""
    if len(matches) < 4:
        raise ValueError(f'匹配點只有 {len(matches)} 組，至少需要 4 組')
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    H, mask = cv2.findHomography(pts2, pts1, method, RANSAC_THRESH, maxIters=5000, confidence=0.999)
    if H is None:
        raise ValueError('無法估計單應矩陣 H')
    mask = mask.ravel().astype(bool)
    if mask.sum() < MIN_INLIERS:
        raise ValueError(f'內點只有 {mask.sum()} 組 (< {MIN_INLIERS})，兩張影像重疊區可能不足')
    return H, mask


def warp_and_blend(img1, img2, H):
    """把影像 2 以 H 投影到影像 1 平面，並用羽化 (feather) 融合接縫。

    1. 投影影像 2 的四角，和影像 1 一起算出畫布範圍，加上平移 T 避免座標為負
    2. 兩張影像各自 warp 到畫布
    3. 每個像素的權重 = 到自身影像邊界的距離 (distanceTransform)，重疊區依權重加權平均，
       接縫處亮度平滑過渡，不會出現明顯切線
    4. 裁掉畫布四周全黑的部分
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = cv2.perspectiveTransform(np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2), H)
    all_pts = np.vstack([corners1, corners2]).reshape(-1, 2)
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    out_w, out_h = x_max - x_min, y_max - y_min
    if out_w * out_h > 25 * (w1 * h1 + w2 * h2):
        raise ValueError('拼接畫布異常巨大，H 可能估計錯誤')

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    size = (out_w, out_h)
    warp1 = cv2.warpPerspective(img1, T, size)
    warp2 = cv2.warpPerspective(img2, T @ H, size)
    mask1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), T, size, flags=cv2.INTER_NEAREST)
    mask2 = cv2.warpPerspective(np.full((h2, w2), 255, np.uint8), T @ H, size, flags=cv2.INTER_NEAREST)

    wt1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3)[..., None]
    wt2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)[..., None]
    total = wt1 + wt2
    pano = (warp1 * wt1 + warp2 * wt2) / np.where(total == 0, 1, total)
    pano = pano.astype(np.uint8)

    ys, xs = np.nonzero((mask1 > 0) | (mask2 > 0))
    return pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _run_matching(img1, img2, use_preprocess):
    """偵測 + 匹配 + H，回傳統計與中間結果；use_preprocess=False 時直接用灰階 + 原始 SIFT 當對照組。"""
    if use_preprocess:
        g1, g2 = preprocess(img1), preprocess(img2)
    else:
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kps1, des1 = detect_sift(g1, root_sift=use_preprocess)
    kps2, des2 = detect_sift(g2, root_sift=use_preprocess)
    matches = match_features(des1, des2)
    try:
        H, inlier_mask = find_homography(kps1, kps2, matches)
    except ValueError:
        H, inlier_mask = None, np.zeros(len(matches), bool)
    return {'gray1': g1, 'gray2': g2, 'kps1': kps1, 'kps2': kps2, 'matches': matches,
            'H': H, 'inlier_mask': inlier_mask}


def compare_preprocessing(img1, img2):
    """同一組影像比較「無前處理」與「前處理 + RootSIFT」的特徵點數、匹配數、內點數與內點比例。"""
    rows = []
    for label, flag in (('無前處理', False), ('前處理', True)):
        r = _run_matching(img1, img2, flag)
        n_match, n_in = len(r['matches']), int(r['inlier_mask'].sum())
        rows.append((label, len(r['kps1']), len(r['kps2']), n_match, n_in, n_in / n_match if n_match else 0))
    print(f"{'':<8s}{'特徵點1':>8s}{'特徵點2':>8s}{'匹配':>8s}{'內點':>8s}{'內點比例':>10s}")
    for label, k1, k2, m, i, ratio in rows:
        print(f'{label:<8s}{k1:>10d}{k2:>10d}{m:>10d}{i:>10d}{ratio:>12.1%}')
    return rows


def Image_Stitching(image_path1, image_path2, records=None, show=True):
    """兩張影像拼接：讀取縮圖 → 前處理 → SIFT (RootSIFT) → FLANN + ratio test + 雙向一致 →
    USAC_MAGSAC 估計 H → warp + 羽化融合。

    影像 2 會投影到影像 1 的平面上，左右或上下順序不拘。
    回傳 dict：img1, img2, gray1, gray2, kps1, kps2, matches, inlier_mask, H, pano。
    """
    records = [] if records is None else records
    name = f'{os.path.basename(image_path1)} + {os.path.basename(image_path2)}'
    print(f'=== {name} ===')

    t0 = time.perf_counter()
    img1, img2 = load_image(image_path1), load_image(image_path2)
    log_run('讀取縮圖', (time.perf_counter() - t0) * 1000, records, name, 尺寸=f'{img1.shape[:2]} / {img2.shape[:2]}')

    t0 = time.perf_counter()
    gray1, gray2 = preprocess(img1), preprocess(img2)
    log_run('前處理', (time.perf_counter() - t0) * 1000, records, name)

    t0 = time.perf_counter()
    kps1, des1 = detect_sift(gray1)
    kps2, des2 = detect_sift(gray2)
    log_run('SIFT', (time.perf_counter() - t0) * 1000, records, name, 特徵點=f'{len(kps1)} / {len(kps2)}')

    t0 = time.perf_counter()
    matches = match_features(des1, des2)
    log_run('匹配', (time.perf_counter() - t0) * 1000, records, name, 匹配數=len(matches))

    t0 = time.perf_counter()
    H, inlier_mask = find_homography(kps1, kps2, matches)
    n_in = int(inlier_mask.sum())
    log_run('估計H', (time.perf_counter() - t0) * 1000, records, name, 內點=f'{n_in} ({n_in / len(matches):.1%})')

    t0 = time.perf_counter()
    pano = warp_and_blend(img1, img2, H)
    log_run('投影融合', (time.perf_counter() - t0) * 1000, records, name, 輸出尺寸=pano.shape[:2])

    result = {'file': name, 'img1': img1, 'img2': img2, 'gray1': gray1, 'gray2': gray2,
              'kps1': kps1, 'kps2': kps2, 'matches': matches, 'inlier_mask': inlier_mask, 'H': H, 'pano': pano}
    if show:
        show_panorama(result)
    return result


def show_preprocessing(result):
    """原圖 vs 前處理後灰階（CLAHE 後暗部細節會被拉出來）。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, key in enumerate(('1', '2')):
        axes[row, 0].imshow(cv2.cvtColor(result['img' + key], cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f'Image {key} (origin)')
        axes[row, 1].imshow(result['gray' + key], cmap='gray')
        axes[row, 1].set_title(f'Image {key} (blur + CLAHE)')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_keypoints(result):
    """在前處理後的影像上畫出 SIFT 特徵點（圓圈大小 = 尺度，線段 = 主方向）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key in zip(axes, ('1', '2')):
        vis = cv2.drawKeypoints(result['gray' + key], result['kps' + key], None,
                                color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(vis)
        ax.set_title(f"Image {key}: {len(result['kps' + key])} SIFT keypoints")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_matches(result, max_draw=80):
    """畫出 RANSAC 後的內點匹配（綠線）。"""
    inliers = [m for m, ok in zip(result['matches'], result['inlier_mask']) if ok]
    step = max(1, len(inliers) // max_draw)
    vis = cv2.drawMatches(result['img1'], result['kps1'], result['img2'], result['kps2'], inliers[::step], None,
                          matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16, 7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Inlier matches: {len(inliers)} / {len(result['matches'])}")
    plt.axis('off')
    plt.show()


def show_panorama(result):
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(result['pano'], cv2.COLOR_BGR2RGB))
    plt.title(f"Panorama: {result['file']}")
    plt.axis('off')
    plt.show()


def analyze_records(records, results):
    """每一組依序列出各步驟執行時間、H 矩陣、前處理效果比較，並顯示前處理、特徵點、匹配與拼接結果。

    results 可以是 Image_Stitching 回傳的單一 dict，或多組結果的 list。
    """
    if not records:
        print('尚無執行紀錄，請先執行 Image_Stitching。')
        return
    results = [results] if isinstance(results, dict) else results

    np.set_printoptions(precision=6, suppress=True)
    for result in results:
        steps = [r for r in records if r['file'] == result['file']]
        print(f"\n==================== {result['file']} ====================")
        print('=== 執行紀錄 ===')
        for i, r in enumerate(steps, 1):
            print(f"{i:2d}. {r['step']:<10s} {r['ms']:9.3f} ms")
        print(f"總計 {sum(r['ms'] for r in steps):9.3f} ms")

        print('\n=== 單應矩陣 H (影像 2 -> 影像 1) ===\n', result['H'])

        print('\n=== 前處理效果比較 ===')
        compare_preprocessing(result['img1'], result['img2'])

        show_preprocessing(result)
        show_keypoints(result)
        show_matches(result)
        show_panorama(result)
