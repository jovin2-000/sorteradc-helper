"""Operators for ball detection: mask proc, binarize, connect, analyze."""
import numpy as np
import cv2
from framework import Operator, Context, StepResult, _encode, to_gray, overlay_mask


class BallMaskProc(Operator):
    """Ball mask dilation + region extraction."""

    def __init__(self):
        super().__init__("ball_mask_proc", "球: 掩码膨胀+区域", "球检测",
                         param_keys=["ball__mask_usedilate"],
                         input_keys=["mask_ball", "aligned_image"],
                         output_keys=["ball_mask_dilated", "ball_region"])

    def execute(self, ctx: Context) -> StepResult:
        mask = ctx.get("mask_ball")
        if mask is None or np.sum(mask) == 0:
            return self._result(None, {"has_ball_mask": False}, status="skip")
        use_dilate = ctx.p("ball__mask_usedilate", True)
        if use_dilate:
            ks = 5 * (mask.shape[0] + mask.shape[1]) / 1280
            ks = max(1, int(ks) | 1)
            kernel = np.ones((ks, ks), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        gray = to_gray(ctx.get("aligned_image"))
        region = cv2.bitwise_and(gray, gray, mask=mask)
        ctx.set("ball_mask_dilated", mask)
        ctx.set("ball_region", region)
        vis = overlay_mask(ctx.get("aligned_image"), mask, (255, 100, 0), 0.3)
        return self._result(vis, {"mask_px": int(np.sum(mask > 0))})


class BallBinarize(Operator):
    """Ball binarization: adaptive/otsu/blob/fixed + morphology."""

    def __init__(self):
        super().__init__("ball_binarize", "球: 二值化", "球检测",
                         param_keys=["ball_inspect_method", "white_pixel_threshold",
                                     "ball_close_iterations", "ball_min_area"],
                         input_keys=["ball_region", "ball_mask_dilated"],
                         output_keys=["ball_binary"])

    def execute(self, ctx: Context) -> StepResult:
        region = ctx.get("ball_region")
        mask = ctx.get("ball_mask_dilated")
        if region is None:
            return self._result(None, status="skip")
        method = ctx.p("ball_inspect_method", "otsu")
        min_area = ctx.p("ball_min_area", 600)

        if method == "adaptive":
            binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, 10)
        elif method == "otsu":
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "blob":
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = min_area
            params.maxArea = min_area * 10
            params.filterByCircularity = True
            params.minCircularity = 0.6
            det = cv2.SimpleBlobDetector_create(params)
            kps = det.detect(region)
            binary = np.zeros_like(region)
            for kp in kps:
                cv2.circle(binary, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), 255, -1)
        else:
            thresh = ctx.p("white_pixel_threshold", 150)
            _, binary = cv2.threshold(region, thresh, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(binary, binary, mask=mask)
        iters = ctx.p("ball_close_iterations", 1)
        if iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)
        ctx.set("ball_binary", binary)
        vis = overlay_mask(ctx.get("aligned_image"), binary, (255, 0, 0), 0.3)
        return self._result(vis, {"method": method, "white_px": int(np.sum(binary > 0))})


class BallConnect(Operator):
    """Connected component analysis + area filtering."""

    def __init__(self):
        super().__init__("ball_connect", "球: 连通域分析", "球检测",
                         param_keys=["ball_min_area"],
                         input_keys=["ball_binary"],
                         output_keys=["balls", "labels", "stats"])

    def execute(self, ctx: Context) -> StepResult:
        binary = ctx.get("ball_binary")
        if binary is None:
            return self._result(None, status="skip")
        min_area = ctx.p("ball_min_area", 600)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        balls = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            balls.append({"id": len(balls), "label": i,
                           "center": (float(centroids[i][0]), float(centroids[i][1])),
                           "area": area,
                           "x": int(stats[i, cv2.CC_STAT_LEFT]),
                           "y": int(stats[i, cv2.CC_STAT_TOP]),
                           "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                           "h": int(stats[i, cv2.CC_STAT_HEIGHT])})
        ctx.set("balls", balls)
        ctx.set("labels", labels)
        ctx.set("stats", stats)
        vis = ctx.get("aligned_image").copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for b in balls:
            cx, cy = int(b["center"][0]), int(b["center"][1])
            cv2.circle(vis, (cx, cy), max(5, int(np.sqrt(b["area"] / 3.14))), (0, 255, 0), 2)
        return self._result(vis, {"total": n - 1, "valid": len(balls)})


class BallAnalyze(Operator):
    """Per-ball analysis: black/white area, ratio, defect classification."""

    def __init__(self):
        super().__init__("ball_analyze", "球: 逐球分析+判定", "球检测",
                         param_keys=["ball_min_black_area", "white_to_black_ratio_threshold",
                                     "close_single_ball_mask", "ball_expected_count"],
                         input_keys=["balls", "labels", "aligned_image"],
                         output_keys=["ball_result"])

    def execute(self, ctx: Context) -> StepResult:
        balls = ctx.get("balls")
        labels = ctx.get("labels")
        if not balls:
            return self._result(None, status="skip")
        min_black = ctx.p("ball_min_black_area", 500)
        ratio_th = ctx.p("white_to_black_ratio_threshold", 0.3)
        close_mask = ctx.p("close_single_ball_mask", False)
        vis = ctx.get("aligned_image").copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        defective = 0
        for b in balls:
            li = b["label"]
            xi, yi, wi, hi = b["x"], b["y"], b["w"], b["h"]
            roi = labels[yi:yi + hi, xi:xi + wi]
            single = (roi == li).astype(np.uint8) * 255
            if close_mask:
                single = cv2.morphologyEx(single, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, hier = cv2.findContours(single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) <= 1:
                continue
            best_idx = -1
            max_a = 0
            for ic, c in enumerate(contours):
                parent = int(hier[0][ic][3])
                if parent == -1:
                    continue
                if parent < len(hier[0]) and int(hier[0][parent][3]) == -1:
                    a = cv2.contourArea(c)
                    if a > max_a:
                        max_a = a
                        best_idx = ic
            if best_idx == -1:
                continue
            black_area = cv2.contourArea(contours[best_idx])
            ratio = 0 / black_area if black_area > 0 else float('inf')
            is_def = black_area < min_black or ratio > ratio_th
            if is_def:
                defective += 1
            color = (0, 0, 255) if is_def else (0, 255, 0)
            cv2.circle(vis, (int(b["center"][0]), int(b["center"][1])),
                       max(5, int(np.sqrt(b["area"] / 3.14))), color, 2)
        expected = ctx.p("ball_expected_count", 100)
        actual = len(balls)
        is_ng = actual != expected or defective > 0
        if is_ng:
            ctx.set("overall_result", "NG")
            reasons = ctx.get("ng_reasons", [])
            reasons.append(f"球检测: 期望{expected}实际{actual}缺陷{defective}")
            ctx.set("ng_reasons", reasons)
        return self._result(vis, {"expected": expected, "actual": actual,
                                   "defective": defective, "is_ng": is_ng})
