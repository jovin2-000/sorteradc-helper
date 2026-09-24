"""
Operator-level detection pipeline.

Each operator is a self-contained function that:
  1. Reads inputs from a shared PipelineContext
  2. Performs one atomic image-processing step
  3. Writes outputs back to the context
  4. Returns a StepResult (visualization image + metrics + param names)

Operators are chained in a defined ORDER. When a parameter changes,
only the affected operator and its downstream operators re-execute.
"""
import time
import base64
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

# Ensure sorteradc-poc is importable for shape_match fallback
import sys, os
POC_ROOT = os.environ.get("POC_ROOT", "/app")
if POC_ROOT not in sys.path:
    sys.path.insert(0, POC_ROOT)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Output of a single operator."""
    name: str
    title: str
    image: Optional[str] = None   # base64 JPEG
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"             # ok | fail | skip | error
    params: List[str] = field(default_factory=list)  # param names this op uses
    error: Optional[str] = None


class PipelineContext:
    """Carries all intermediate data between operators."""

    def __init__(self, image: np.ndarray, template: np.ndarray, params: dict,
                 mask_border: np.ndarray = None, mask_ball: np.ndarray = None,
                 mask_foreground: np.ndarray = None, mask_trace: np.ndarray = None,
                 mask_expand: np.ndarray = None,
                 tpl_gx=None, tpl_gy=None, tpl_mag=None, tpl_contour=None, tpl_xym=None):
        self.image = image
        self.template = template
        self.params = params
        self.mask_border = mask_border
        self.mask_ball = mask_ball
        self.mask_foreground = mask_foreground
        self.mask_trace = mask_trace
        self.mask_expand = mask_expand
        self.tpl_gx = tpl_gx
        self.tpl_gy = tpl_gy
        self.tpl_mag = tpl_mag
        self.tpl_contour = tpl_contour
        self.tpl_xym = tpl_xym

        # Intermediate results (filled by operators)
        self.search_roi: Optional[np.ndarray] = None
        self.roi_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.aligned_image: Optional[np.ndarray] = None
        self.match_score: float = 0.0
        self.match_angle: float = 0.0
        self.match_scale: float = 1.0
        self.match_method: str = ""
        self.match_corners: Optional[np.ndarray] = None
        self.match_center: Optional[Tuple[float, float]] = None
        self.sift_kp1 = None
        self.sift_kp2 = None
        self.sift_good_matches = None
        self.ransac_M = None
        self.ransac_inliers = 0

        # Border detection per method
        self._border_results: Dict[str, Dict[str, Any]] = {}
        self.border_fused_result = None
        self.border_classified = None

        # Ball detection
        self.ball_binary: Optional[np.ndarray] = None
        self.ball_result = None

        # Surface detection per method
        self._surface_results: Dict[str, Dict[str, Any]] = {}
        self.surface_fused_result = None

        # Final
        self.overall_result = "OK"
        self.ng_reasons: List[str] = []

    def _p(self, name, default=None):
        """Get a param value."""
        return self.params.get(name, default)

    def set_border_intermediate(self, method: str, key: str, value):
        """Store an intermediate result for a border detection method."""
        if method not in self._border_results:
            self._border_results[method] = {}
        self._border_results[method][key] = value

    def get_border_intermediate(self, method: str, key: str):
        return self._border_results.get(method, {}).get(key)

    def set_surface_intermediate(self, method: str, key: str, value):
        if method not in self._surface_results:
            self._surface_results[method] = {}
        self._surface_results[method][key] = value

    def get_surface_intermediate(self, method: str, key: str):
        return self._surface_results.get(method, {}).get(key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(img) -> Optional[str]:
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def _overlay(base, mask, color=(0, 0, 255), alpha=0.4):
    """Overlay a binary mask on base image."""
    vis = base.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    mb = mask > 0
    if mb.any():
        ov = vis.copy()
        ov[mb] = color
        vis = cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0)
    return vis


def _draw_contours(base, contours, color=(0, 0, 255), thickness=2):
    vis = base.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _preprocess_region(template_img, target_img, region_mask):
    """Extract gray template and target regions within the mask."""
    tpl_gray = _to_gray(template_img)
    tgt_gray = _to_gray(target_img)
    if region_mask is not None and region_mask.shape == tpl_gray.shape:
        tpl_region = cv2.bitwise_and(tpl_gray, tpl_gray, mask=region_mask)
        tgt_region = cv2.bitwise_and(tgt_gray, tgt_gray, mask=region_mask)
    else:
        tpl_region = tpl_gray
        tgt_region = tgt_gray
    return tpl_region, tgt_region


def _post_process(binary_defects, region_mask, min_defect_area, diff_image=None, kernelsz=3):
    """Morphology cleanup + contour extraction + area filter. Returns BorderDefectResult-like dict."""
    if kernelsz > 1:
        kernel = np.ones((kernelsz, kernelsz), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary_defects, cv2.MORPH_OPEN, kernel)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel)
    else:
        binary_cleaned = binary_defects

    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_defect_area]

    filtered_binary = np.zeros_like(binary_cleaned)
    cv2.drawContours(filtered_binary, filtered, -1, 255, thickness=cv2.FILLED)
    defect_px = int(np.sum(filtered_binary > 0))
    total_px = int(np.sum(region_mask > 0)) if region_mask is not None else 1
    ratio = defect_px / total_px * 100 if total_px > 0 else 0.0

    return {
        "defect_pixels": defect_px,
        "defect_ratio": ratio,
        "defect_contours": filtered,
        "diff_image": diff_image if diff_image is not None else filtered_binary,
        "binary_defects": filtered_binary,
    }


# ---------------------------------------------------------------------------
# Operator functions
# Each returns StepResult
# ---------------------------------------------------------------------------

def op_roi_extract(ctx: PipelineContext) -> StepResult:
    """ROI裁剪：从输入图中心提取搜索区域。"""
    img = ctx.image
    h, w = img.shape[:2]
    roiw = min(ctx._p("search_roiw", 300), w)
    roih = min(ctx._p("search_roih", 400), h)
    xc, yc = w // 2, h // 2
    y0, y1 = max(0, yc - roih // 2), min(h, yc + roih // 2)
    x0, x1 = max(0, xc - roiw // 2), min(w, xc + roiw // 2)
    ctx.search_roi = img[y0:y1, x0:x1]
    ctx.roi_rect = (x0, y0, x1, y1)

    vis = img.copy()
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(vis, f"ROI {x1-x0}x{y1-y0}", (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return StepResult("roi_extract", "ROI裁剪", _encode(vis),
                       {"roi_w": x1 - x0, "roi_h": y1 - y0},
                       params=["search_roiw", "search_roih"])


def op_sift_detect(ctx: PipelineContext) -> StepResult:
    """SIFT特征检测：在模板和ROI上提取SIFT关键点和描述子。"""
    tmpl_gray = _to_gray(ctx.template)
    scene_gray = _to_gray(ctx.search_roi)
    max_feat = 4000
    sift = cv2.SIFT_create(nfeatures=max_feat)
    kp1, des1 = sift.detectAndCompute(tmpl_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)

    ctx.sift_kp1 = kp1
    ctx.sift_kp2 = kp2
    # Store descriptors on context for next op
    ctx._sift_des1 = des1
    ctx._sift_des2 = des2

    n1 = len(kp1) if kp1 else 0
    n2 = len(kp2) if kp2 else 0
    vis = cv2.drawKeypoints(tmpl_gray, kp1 or [], None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    status = "ok" if n1 >= 4 and n2 >= 4 else "fail"
    return StepResult("sift_detect", "SIFT特征检测", _encode(vis),
                       {"tmpl_kp": n1, "scene_kp": n2, "sufficient": n1 >= 4 and n2 >= 4},
                       status=status)


def op_sift_match(ctx: PipelineContext) -> StepResult:
    """SIFT特征匹配：Lowe ratio test筛选好匹配。"""
    des1, des2 = ctx._sift_des1, ctx._sift_des2
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        ctx.sift_good_matches = []
        return StepResult("sift_match", "SIFT特征匹配", None,
                           {"good_matches": 0}, status="fail",
                           params=["tm_thresh"])

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    ratio = 0.75
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < ratio * n.distance]
    ctx.sift_good_matches = good

    min_good = 10
    status = "ok" if len(good) >= min_good else "fail"
    # Draw matches
    vis = cv2.drawMatches(_to_gray(ctx.template), ctx.sift_kp1,
                          _to_gray(ctx.search_roi), ctx.sift_kp2,
                          good[:50], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return StepResult("sift_match", "SIFT特征匹配", _encode(vis),
                       {"good_matches": len(good), "min_required": min_good},
                       status=status)


def op_ransac_pose(ctx: PipelineContext) -> StepResult:
    """RANSAC变换估计：从好匹配估计相似变换矩阵。"""
    good = ctx.sift_good_matches
    if not good or len(good) < 10:
        return StepResult("ransac_pose", "RANSAC变换估计", None,
                           {"inliers": 0}, status="skip")

    kp1, kp2 = ctx.sift_kp1, ctx.sift_kp2
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=4.0)
    ctx.ransac_M = M
    if M is not None:
        inl = int(inliers.sum())
        ctx.ransac_inliers = inl
        score = inl / max(1, len(good))
        a, b = M[0, 0], M[1, 0]
        scale = float(np.hypot(a, b))
        angle = float(np.degrees(np.arctan2(b, a)))
        return StepResult("ransac_pose", "RANSAC变换估计", None,
                           {"inliers": inl, "total": len(good), "score": round(score, 4),
                            "angle": round(angle, 2), "scale": round(scale, 4)},
                           status="ok" if inl >= 6 else "fail")
    return StepResult("ransac_pose", "RANSAC变换估计", None,
                       {"inliers": 0}, status="fail")


def op_shape_match_fallback(ctx: PipelineContext) -> StepResult:
    """ShapeMatch兜底：SIFT失败时用边缘形状匹配定位die。"""
    from algorithms.ShapeMatch import shape_match

    class M: pass
    m = M()
    m.shape_match_template = ctx.template
    m.mask_border_only = ctx.mask_border
    m.mask_ball_only = ctx.mask_ball
    m.mask_all_foreground = ctx.mask_foreground
    m.shape_match_tpl_gx = ctx.tpl_gx
    m.shape_match_tpl_gy = ctx.tpl_gy
    m.shape_match_tpl_mag = ctx.tpl_mag
    m.shape_match_tpl_contour_points = ctx.tpl_contour
    m.shape_match_tpl_xym = ctx.tpl_xym
    m.shape_match_pyramid_levels = 3
    m.shape_match_min_score = 0.7
    m.canonical_roi_size = ctx.template.shape[:2]

    min_conf = ctx._p("shape_match_min_confidence", 0.85)
    output = shape_match(ctx.search_roi, m, min_conf,
                         step_size=2, enable_pyramid=True,
                         pyramid_scales=[4, 2, 1], refinement_radius=10)
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    if not boxes:
        return StepResult("shape_match_fallback", "ShapeMatch兜底", _encode(ctx.search_roi),
                           {"hit": False}, status="fail",
                           params=["shape_match_min_confidence"])

    best = int(np.argmax(scores)) if isinstance(scores, list) else 0
    die = boxes[best]
    score = max(scores) if isinstance(scores, list) else scores
    ctx._shape_match_box = die
    ctx._shape_match_score = score
    ctx.match_method = "shape_match"
    ctx.match_score = float(score)

    vis = ctx.search_roi.copy()
    x, y, bw, bh = die
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv2.putText(vis, f"score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return StepResult("shape_match_fallback", "ShapeMatch兜底", _encode(vis),
                       {"score": round(float(score), 4), "boxes": len(boxes)},
                       params=["shape_match_min_confidence"])


def op_warp_align(ctx: PipelineContext) -> StepResult:
    """摆正抠图：将匹配到的die区域变换回模板姿态。"""
    aligned = None
    method = ""
    if ctx.ransac_M is not None and ctx.ransac_inliers >= 6:
        # Use SIFT result
        M = ctx.ransac_M
        th_t, tw_t = _to_gray(ctx.template).shape[:2]
        M_inv = cv2.invertAffineTransform(M)
        # M maps template->scene, need scene->template for ROI
        # But M was estimated on search_roi coords, need to adjust for full image offset
        # For simplicity, use the ROI offset
        x0, y0 = ctx.roi_rect[0], ctx.roi_rect[1]
        M_adj = M.copy()
        M_adj[0, 2] -= x0
        M_adj[1, 2] -= y0
        M_inv = cv2.invertAffineTransform(M_adj)
        aligned = cv2.warpAffine(ctx.image, M_inv, (tw_t, th_t),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        method = "SIFT+RANSAC"
        ctx.match_method = method
    elif hasattr(ctx, "_shape_match_box"):
        # Use shape_match + ImageRegistration
        from algorithms.ImageRegistration import ImageRegistration
        die = ctx._shape_match_box
        registrar = ImageRegistration(black_threshold=ctx._p("extract_threshold", 0),
                                       min_contour_area=1000.0)
        reg = registrar.register_by_perspective_transform(
            template=ctx.template, target_image=ctx.search_roi,
            match_row=die[1], match_col=die[0],
            reference_point="topleft", roi_size=(die[2], die[3]),
            enable_refinement=True)
        aligned = reg.aligned_image
        method = "shape_match+registration"
        ctx.match_method = method
        ctx.match_angle = float(reg.rotation_angle)
        ctx.match_scale = float(reg.scale_factor)
        ctx.match_score = float(getattr(ctx, "_shape_match_score", 0))

    if aligned is None:
        return StepResult("warp_align", "摆正抠图", None,
                           {"error": "no match"}, status="fail")

    ctx.aligned_image = aligned
    return StepResult("warp_align", "摆正抠图", _encode(aligned),
                       {"size": f"{aligned.shape[1]}x{aligned.shape[0]}",
                        "method": method, "score": round(ctx.match_score, 4)})


def op_mask_apply(ctx: PipelineContext) -> StepResult:
    """掩码叠加：在aligned图上可视化border/ball/trace掩码。"""
    aligned = ctx.aligned_image
    vis = aligned.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    masks = {"border": (ctx.mask_border, (0, 255, 0)),
             "ball": (ctx.mask_ball, (255, 100, 0)),
             "foreground": (ctx.mask_foreground, (100, 100, 100))}
    metrics = {}
    for name, (mask, color) in masks.items():
        if mask is not None and mask.shape[:2] == vis.shape[:2]:
            mb = mask > 0
            ov = vis.copy()
            ov[mb] = color
            vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)
            metrics[f"{name}_px"] = int(np.sum(mb))
    return StepResult("mask_apply", "掩码叠加", _encode(vis), metrics)


# ---- Border detection: shared sub-operators (parameterized by method) ----

def _make_border_preprocess(method: str, method_label: str):
    """Create a preprocess operator for a border detection method."""
    def op(ctx: PipelineContext) -> StepResult:
        tpl_region, tgt_region = _preprocess_region(
            ctx.template, ctx.aligned_image, ctx.mask_border)
        ctx.set_border_intermediate(method, "tpl_region", tpl_region)
        ctx.set_border_intermediate(method, "tgt_region", tgt_region)
        vis = cv2.hconcat([tpl_region, tgt_region])
        return StepResult(f"border_{method}_pre", f"边框-{method_label}: 预处理",
                          _encode(vis), {"region_w": tpl_region.shape[1], "region_h": tpl_region.shape[0]})
    return op


def _make_border_freq_filter():
    """频域法: FFT + 高通滤波 + 逆变换。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl = ctx.get_border_intermediate("frequency", "tpl_region")
        tgt = ctx.get_border_intermediate("frequency", "tgt_region")
        if tpl is None:
            return StepResult("border_freq_fft", "边框-频域: FFT+高通", None,
                               {"error": "no input"}, status="skip")
        cutoff = ctx._p("border_frequency_cutoff", 30.0)

        f_tpl = np.fft.fft2(tpl.astype(np.float32))
        f_tgt = np.fft.fft2(tgt.astype(np.float32))
        rows, cols = tpl.shape
        crow, ccol = rows // 2, cols // 2
        hp = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        hp[dist <= cutoff] = 0

        f_tpl_hp = f_tpl * hp
        f_tgt_hp = f_tgt * hp
        tpl_hp = np.abs(np.fft.ifft2(f_tpl_hp))
        tgt_hp = np.abs(np.fft.ifft2(f_tgt_hp))
        tpl_hp = cv2.normalize(tpl_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tgt_hp = cv2.normalize(tgt_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        ctx.set_border_intermediate("frequency", "tpl_hp", tpl_hp)
        ctx.set_border_intermediate("frequency", "tgt_hp", tgt_hp)

        vis = cv2.hconcat([tpl_hp, tgt_hp])
        return StepResult("border_freq_fft", "边框-频域: FFT+高通",
                          _encode(vis), {"cutoff": cutoff},
                          params=["border_frequency_cutoff"])
    return op


def _make_border_spatial_clahe():
    """空域法: CLAHE对比度增强。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl = ctx.get_border_intermediate("spatial", "tpl_region")
        tgt = ctx.get_border_intermediate("spatial", "tgt_region")
        if tpl is None:
            return StepResult("border_spatial_clahe", "边框-空域: CLAHE", None,
                               {"error": "no input"}, status="skip")
        clip = ctx._p("border_clahe_clip_limit", 2)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        tpl_enh = clahe.apply(tpl)
        tgt_enh = clahe.apply(tgt)
        ctx.set_border_intermediate("spatial", "tpl_enh", tpl_enh)
        ctx.set_border_intermediate("spatial", "tgt_enh", tgt_enh)
        vis = cv2.hconcat([tpl_enh, tgt_enh])
        return StepResult("border_spatial_clahe", "边框-空域: CLAHE",
                          _encode(vis), {"clip_limit": clip},
                          params=["border_clahe_clip_limit"])
    return op


def _make_border_spatial_gradient():
    """空域法: Sobel梯度计算 + 差分。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl = ctx.get_border_intermediate("spatial", "tpl_enh")
        tgt = ctx.get_border_intermediate("spatial", "tgt_enh")
        if tpl is None:
            return StepResult("border_spatial_grad", "边框-空域: 梯度", None,
                               status="skip")
        tpl_detail = cv2.GaussianBlur(tpl, (3, 3), 0)
        tgt_detail = cv2.GaussianBlur(tgt, (3, 3), 0)
        tgx = cv2.Sobel(tpl_detail, cv2.CV_64F, 1, 0, ksize=3)
        tgy = cv2.Sobel(tpl_detail, cv2.CV_64F, 0, 1, ksize=3)
        tmag = cv2.magnitude(tgx, tgy)
        dgx = cv2.Sobel(tgt_detail, cv2.CV_64F, 1, 0, ksize=3)
        dgy = cv2.Sobel(tgt_detail, cv2.CV_64F, 0, 1, ksize=3)
        dmag = cv2.magnitude(dgx, dgy)
        tmag_n = cv2.normalize(tmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dmag_n = cv2.normalize(dmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_diff = cv2.absdiff(tmag_n, dmag_n)
        ctx.set_border_intermediate("spatial", "grad_diff", grad_diff)
        vis = cv2.hconcat([tmag_n, dmag_n])
        return StepResult("border_spatial_grad", "边框-空域: Sobel梯度",
                          _encode(vis), {})
    return op


def _make_border_tpl_binary():
    """模板法: 二值化(OTSU/固定)。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl = ctx.get_border_intermediate("template", "tpl_region")
        tgt = ctx.get_border_intermediate("template", "tgt_region")
        if tpl is None:
            return StepResult("border_tpl_binary", "边框-模板: 二值化", None, status="skip")
        bt = ctx._p("border_binary_threshold", 0)
        if bt == 0:
            _, tpl_bin = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, tgt_bin = cv2.threshold(tgt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, tpl_bin = cv2.threshold(tpl, bt, 255, cv2.THRESH_BINARY)
            _, tgt_bin = cv2.threshold(tgt, bt, 255, cv2.THRESH_BINARY)
        ctx.set_border_intermediate("template", "tpl_bin", tpl_bin)
        ctx.set_border_intermediate("template", "tgt_bin", tgt_bin)
        vis = cv2.hconcat([tpl_bin, tgt_bin])
        return StepResult("border_tpl_binary", "边框-模板: 二值化",
                          _encode(vis), {"threshold": bt if bt > 0 else "OTSU"},
                          params=["border_binary_threshold"])
    return op


def _make_border_tpl_contour():
    """模板法: 轮廓提取 + 差分。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl_bin = ctx.get_border_intermediate("template", "tpl_bin")
        tgt_bin = ctx.get_border_intermediate("template", "tgt_bin")
        if tpl_bin is None:
            return StepResult("border_tpl_contour", "边框-模板: 轮廓差分", None, status="skip")
        thickness = ctx._p("border_thickness", 1)
        tc, _ = cv2.findContours(tpl_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dc, _ = cv2.findContours(tgt_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tpl_mask = np.zeros_like(tpl_bin)
        tgt_mask = np.zeros_like(tgt_bin)
        cv2.drawContours(tpl_mask, tc, -1, 255, thickness)
        cv2.drawContours(tgt_mask, dc, -1, 255, thickness)
        diff = cv2.absdiff(tpl_mask, tgt_mask)
        ctx.set_border_intermediate("template", "diff", diff)
        vis = cv2.hconcat([tpl_mask, tgt_mask])
        return StepResult("border_tpl_contour", "边框-模板: 轮廓差分",
                          _encode(vis), {"tpl_contours": len(tc), "tgt_contours": len(dc)},
                          params=["border_thickness"])
    return op


def _make_border_diff_absdiff():
    """差分法: 绝对差分。"""
    def op(ctx: PipelineContext) -> StepResult:
        tpl = ctx.get_border_intermediate("difference", "tpl_region")
        tgt = ctx.get_border_intermediate("difference", "tgt_region")
        if tpl is None:
            return StepResult("border_diff_absdiff", "边框-差分: 绝对差分", None, status="skip")
        diff = cv2.absdiff(tpl, tgt)
        ctx.set_border_intermediate("difference", "diff", diff)
        return StepResult("border_diff_absdiff", "边框-差分: 绝对差分",
                          _encode(diff), {})
    return op


def _make_border_method_threshold(method: str, method_label: str,
                                    threshold_param: str, diff_key: str = "diff"):
    """通用: 二值化 + 后处理 (frequency/spatial/template/difference共用)。"""
    def op(ctx: PipelineContext) -> StepResult:
        diff = ctx.get_border_intermediate(method, diff_key)
        if diff is None:
            # frequency method stores tpl_hp/tgt_hp, compute diff here
            if method == "frequency":
                tpl_hp = ctx.get_border_intermediate("frequency", "tpl_hp")
                tgt_hp = ctx.get_border_intermediate("frequency", "tgt_hp")
                if tpl_hp is not None:
                    diff = cv2.absdiff(tpl_hp, tgt_hp)
                    ctx.set_border_intermediate("frequency", "diff", diff)
            if diff is None:
                return StepResult(f"border_{method}_thresh",
                                  f"边框-{method_label}: 二值化", None, status="skip")

        thresh = ctx._p(threshold_param, 30)
        _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        min_area = ctx._p("border_min_defect_area", 10.0)
        kernelsz = ctx._p("border_post_kernelsz", 3)
        result = _post_process(binary, ctx.mask_border, min_area, diff, kernelsz)
        ctx.set_border_intermediate(method, "result", result)

        vis = _overlay(ctx.aligned_image, result["binary_defects"], (0, 0, 255))
        vis = _draw_contours(vis, result["defect_contours"], (0, 0, 255), 2)
        return StepResult(f"border_{method}_thresh",
                          f"边框-{method_label}: 二值化+后处理",
                          _encode(vis),
                          {"defect_px": result["defect_pixels"],
                           "ratio": round(result["defect_ratio"], 4),
                           "thresh": thresh},
                          params=[threshold_param, "border_min_defect_area", "border_post_kernelsz"])
    return op


def _make_border_tpl_postproc():
    """模板法: 形态学 + 轮廓过滤 (special: needs its own diff handling)."""
    def op(ctx: PipelineContext) -> StepResult:
        diff = ctx.get_border_intermediate("template", "diff")
        if diff is None:
            return StepResult("border_tpl_postproc", "边框-模板: 后处理", None, status="skip")
        _, binary = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)
        kernelsz = ctx._p("border_post_kernelsz", 3)
        if kernelsz > 1:
            kernel = np.ones((kernelsz, kernelsz), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        min_area = ctx._p("border_min_defect_area", 10.0)
        result = _post_process(binary, ctx.mask_border, min_area, diff, kernelsz)
        ctx.set_border_intermediate("template", "result", result)
        vis = _overlay(ctx.aligned_image, result["binary_defects"], (0, 0, 255))
        vis = _draw_contours(vis, result["defect_contours"], (0, 0, 255), 2)
        return StepResult("border_tpl_postproc", "边框-模板: 形态学+轮廓",
                          _encode(vis),
                          {"defect_px": result["defect_pixels"],
                           "ratio": round(result["defect_ratio"], 4)},
                          params=["border_min_defect_area", "border_post_kernelsz"])
    return op


def op_border_fusion(ctx: PipelineContext) -> StepResult:
    """边框融合: 加权投票融合多方法结果。"""
    methods = ctx._p("border_methods", ["frequency", "spatial", "template", "difference"])
    weights = ctx._p("border_weights", {"frequency": 0.4, "spatial": 0.3, "template": 0.2, "difference": 0.1})
    vote_thresh = ctx._p("border_vote_threshold", 0.5)
    min_area = ctx._p("border_min_defect_area", 10.0)
    fusion_kernelsz = ctx._p("border_fusion_kernelsz", 3)
    skip_open = ctx._p("border_fusion_skip_open", False)

    binary_masks = []
    weight_arr = []
    for method in methods:
        result = ctx.get_border_intermediate(method, "result")
        if result is not None:
            binary_masks.append(result["binary_defects"].astype(np.float32) / 255.0)
            weight_arr.append(weights.get(method, 0.0))

    if not binary_masks:
        return StepResult("border_fusion", "边框-融合", None, {"error": "no methods"}, status="fail")

    weight_arr = np.array(weight_arr)
    weight_arr = weight_arr / weight_arr.sum()
    fused = np.zeros_like(binary_masks[0])
    for i, mask in enumerate(binary_masks):
        fused += mask * weight_arr[i]

    _, fused_binary = cv2.threshold((fused * 255).astype(np.uint8),
                                     int(vote_thresh * 255), 255, cv2.THRESH_BINARY)
    if fusion_kernelsz > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (fusion_kernelsz, fusion_kernelsz))
        fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)
        if not skip_open:
            fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_OPEN, kernel)

    result = _post_process(fused_binary, ctx.mask_border, min_area, (fused * 255).astype(np.uint8), 1)
    ctx.border_fused_result = result
    vis = _overlay(ctx.aligned_image, result["binary_defects"], (0, 0, 255))
    vis = _draw_contours(vis, result["defect_contours"], (0, 0, 255), 2)
    is_ng = result["defect_pixels"] >= min_area
    if is_ng:
        ctx.overall_result = "NG"
        ctx.ng_reasons.append(f"边框缺陷: {result['defect_pixels']}px")
    return StepResult("border_fusion", "边框-融合结果",
                      _encode(vis),
                      {"defect_px": result["defect_pixels"],
                       "ratio": round(result["defect_ratio"], 4),
                       "is_ng": is_ng, "methods": len(binary_masks)},
                      params=["border_methods", "border_weights", "border_vote_threshold",
                              "border_fusion_kernelsz", "border_fusion_skip_open"])


def op_border_classify(ctx: PipelineContext) -> StepResult:
    """崩角/崩边分类。"""
    if ctx.border_fused_result is None:
        return StepResult("border_classify", "边框-分类", None, status="skip")
    ratio = ctx._p("border_corner_ratio", 0.18)
    h, w = ctx.aligned_image.shape[:2]
    corner_size = max(1, int(min(w, h) * ratio))
    corners = [(0, 0), (w - corner_size, 0), (0, h - corner_size), (w - corner_size, h - corner_size)]
    subtypes = []
    for contour in ctx.border_fused_result["defect_contours"]:
        x, y, cw, ch = cv2.boundingRect(contour)
        is_corner = any(x < cx0 + corner_size and cx0 < x + cw and
                        y < cy0 + corner_size and cy0 < y + ch for cx0, cy0 in corners)
        subtypes.append("崩角" if is_corner else "崩边")

    vis = _overlay(ctx.aligned_image, ctx.border_fused_result["binary_defects"], (0, 0, 255))
    # Draw corner regions
    for cx0, cy0 in corners:
        cv2.rectangle(vis, (cx0, cy0), (cx0 + corner_size, cy0 + corner_size), (0, 255, 255), 1)
    n_corner = subtypes.count("崩角")
    n_edge = subtypes.count("崩边")
    return StepResult("border_classify", "边框-崩角/崩边分类",
                      _encode(vis),
                      {"崩角": n_corner, "崩边": n_edge, "corner_ratio": ratio},
                      params=["border_corner_ratio"])


# ---- Ball detection operators ----

def op_ball_mask_proc(ctx: PipelineContext) -> StepResult:
    """球掩码处理: 膨胀 + 区域提取。"""
    mask = ctx.mask_ball
    if mask is None or np.sum(mask) == 0:
        return StepResult("ball_mask_proc", "球: 掩码处理", None,
                           {"has_ball_mask": False}, status="skip")
    use_dilate = ctx._p("ball__mask_usedilate", True)
    if use_dilate:
        ks = 5 * (mask.shape[0] + mask.shape[1]) / 1280
        ks = max(1, int(ks) | 1)
        kernel = np.ones((ks, ks), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    gray = _to_gray(ctx.aligned_image)
    ball_region = cv2.bitwise_and(gray, gray, mask=mask)
    ctx.set_border_intermediate("__ball__", "mask_dilated", mask)
    ctx.set_border_intermediate("__ball__", "ball_region", ball_region)
    vis = _overlay(ctx.aligned_image, mask, (255, 100, 0), 0.3)
    return StepResult("ball_mask_proc", "球: 掩码膨胀+区域提取",
                      _encode(vis), {"mask_px": int(np.sum(mask > 0)), "dilate": use_dilate},
                      params=["ball__mask_usedilate"])


def op_ball_binarize(ctx: PipelineContext) -> StepResult:
    """球二值化: adaptive/otsu/blob/fixed。"""
    region = ctx.get_border_intermediate("__ball__", "ball_region")
    mask = ctx.get_border_intermediate("__ball__", "mask_dilated")
    if region is None:
        return StepResult("ball_binarize", "球: 二值化", None, status="skip")
    method = ctx._p("ball_inspect_method", "otsu")
    min_area = ctx._p("ball_min_area", 600)

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
    else:  # fixed
        thresh = ctx._p("white_pixel_threshold", 150)
        _, binary = cv2.threshold(region, thresh, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_and(binary, binary, mask=mask)
    # Morphology
    iters = ctx._p("ball_close_iterations", 1)
    if iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)
    ctx.ball_binary = binary

    vis = _overlay(ctx.aligned_image, binary, (255, 0, 0), 0.3)
    return StepResult("ball_binarize", "球: 二值化",
                      _encode(vis), {"method": method, "white_px": int(np.sum(binary > 0))},
                      params=["ball_inspect_method", "white_pixel_threshold", "ball_close_iterations"])


def op_ball_connect(ctx: PipelineContext) -> StepResult:
    """球连通域分析: 标记 + 面积过滤。"""
    if ctx.ball_binary is None:
        return StepResult("ball_connect", "球: 连通域分析", None, status="skip")
    min_area = ctx._p("ball_min_area", 600)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ctx.ball_binary, 8)
    balls = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        balls.append({
            "id": len(balls),
            "center": (float(centroids[i][0]), float(centroids[i][1])),
            "area": area,
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "label": i,
        })
    ctx.set_border_intermediate("__ball__", "balls", balls)
    ctx.set_border_intermediate("__ball__", "labels", labels)
    ctx.set_border_intermediate("__ball__", "stats", stats)

    vis = ctx.aligned_image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for b in balls:
        cx, cy = int(b["center"][0]), int(b["center"][1])
        cv2.circle(vis, (cx, cy), max(5, int(np.sqrt(b["area"] / 3.14))), (0, 255, 0), 2)
    return StepResult("ball_connect", "球: 连通域+面积过滤",
                      _encode(vis), {"total_labels": num_labels - 1, "valid_balls": len(balls)},
                      params=["ball_min_area"])


def op_ball_analyze(ctx: PipelineContext) -> StepResult:
    """逐球分析: 黑面积/白面积/比值 + 缺陷判定。"""
    balls = ctx.get_border_intermediate("__ball__", "balls")
    labels = ctx.get_border_intermediate("__ball__", "labels")
    if not balls:
        return StepResult("ball_analyze", "球: 逐球分析", None, status="skip")
    min_black = ctx._p("ball_min_black_area", 500)
    ratio_thresh = ctx._p("white_to_black_ratio_threshold", 0.3)
    close_mask = ctx._p("close_single_ball_mask", False)

    vis = ctx.aligned_image.copy()
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
        contours, hi2 = cv2.findContours(single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 1:
            continue
        best_idx = -1
        max_area = 0
        for ic, c in enumerate(contours):
            parent = int(hi2[0][ic][3])
            if parent == -1:
                continue
            if parent < len(hi2[0]) and int(hi2[0][parent][3]) == -1:
                a = cv2.contourArea(c)
                if a > max_area:
                    max_area = a
                    best_idx = ic
        if best_idx == -1:
            continue
        black_area = cv2.contourArea(contours[best_idx])
        white_area = 0  # simplified
        ratio = white_area / black_area if black_area > 0 else float('inf')
        is_def = black_area < min_black or ratio > ratio_thresh
        if is_def:
            defective += 1
        color = (0, 0, 255) if is_def else (0, 255, 0)
        cv2.circle(vis, (int(b["center"][0]), int(b["center"][1])),
                   max(5, int(np.sqrt(b["area"] / 3.14))), color, 2)

    expected = ctx._p("ball_expected_count", 100)
    actual = len(balls)
    is_ng = actual != expected or defective > 0
    if is_ng:
        ctx.overall_result = "NG"
        ctx.ng_reasons.append(f"球检测: 期望{expected}实际{actual}缺陷{defective}")
    return StepResult("ball_analyze", "球: 逐球分析+判定",
                      _encode(vis),
                      {"expected": expected, "actual": actual, "defective": defective, "is_ng": is_ng},
                      params=["ball_min_black_area", "white_to_black_ratio_threshold",
                              "close_single_ball_mask", "ball_expected_count"])


# ---- Surface detection (reuses border operators with surface mask) ----
# Surface methods are identical to border methods but with surface masks.
# We create them dynamically using the same operator factories.

def op_surface_fusion(ctx: PipelineContext) -> StepResult:
    """表面融合: 同边框融合逻辑，用表面掩码。"""
    methods = ctx._p("surface_methods", ["frequency", "spatial", "template"])
    weights = ctx._p("surface_weights", {"frequency": 0.4, "spatial": 0.3, "template": 0.3})
    vote_thresh = ctx._p("surface_vote_threshold", 0.5)
    min_area = ctx._p("surface_min_defect_area", 10.0)

    # Surface mask = everything except border and ball
    sm = np.ones_like(ctx.mask_border) * 255 if ctx.mask_border is not None else None
    if sm is not None:
        if ctx.mask_border is not None:
            sm[ctx.mask_border > 0] = 0
        if ctx.mask_ball is not None:
            sm[ctx.mask_ball > 0] = 0

    binary_masks = []
    weight_arr = []
    for method in methods:
        result = ctx.get_surface_intermediate(method, "result")
        if result is not None:
            binary_masks.append(result["binary_defects"].astype(np.float32) / 255.0)
            weight_arr.append(weights.get(method, 0.0))

    if not binary_masks:
        return StepResult("surface_fusion", "表面-融合", None, {"error": "no methods"}, status="skip")

    weight_arr = np.array(weight_arr)
    weight_arr = weight_arr / weight_arr.sum()
    fused = np.zeros_like(binary_masks[0])
    for i, mask in enumerate(binary_masks):
        fused += mask * weight_arr[i]
    _, fused_binary = cv2.threshold((fused * 255).astype(np.uint8),
                                     int(vote_thresh * 255), 255, cv2.THRESH_BINARY)
    result = _post_process(fused_binary, sm, min_area, (fused * 255).astype(np.uint8), 1)
    ctx.surface_fused_result = result
    vis = _overlay(ctx.aligned_image, result["binary_defects"], (0, 255, 0))
    vis = _draw_contours(vis, result["defect_contours"], (0, 255, 0), 2)
    is_ng = result["defect_pixels"] >= min_area
    if is_ng:
        ctx.overall_result = "NG"
        ctx.ng_reasons.append(f"表面缺陷: {result['defect_pixels']}px")
    return StepResult("surface_fusion", "表面-融合结果",
                      _encode(vis),
                      {"defect_px": result["defect_pixels"],
                       "ratio": round(result["defect_ratio"], 4),
                       "is_ng": is_ng},
                      params=["surface_methods", "surface_weights", "surface_vote_threshold"])


def op_result_aggregate(ctx: PipelineContext) -> StepResult:
    """最终结果汇总。"""
    from algorithms.MaskBasedDefectDetector import MaskBasedDefectDetector
    try:
        detector = MaskBasedDefectDetector()
        vis_results = {}
        if ctx.border_fused_result:
            vis_results["border_defects"] = type("R", (), ctx.border_fused_result)()
        if ctx.surface_fused_result:
            vis_results["surface_defects"] = type("R", (), ctx.surface_fused_result)()
        final_vis = detector.visualize_all_results(
            target_image=ctx.aligned_image, results=vis_results,
            overall_result=ctx.overall_result, save_path=None)
    except Exception:
        final_vis = ctx.aligned_image.copy()
        cv2.putText(final_vis, ctx.overall_result, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if ctx.overall_result == "NG" else (0, 255, 0), 2)
    return StepResult("result_aggregate", "最终结果",
                      _encode(final_vis),
                      {"overall": ctx.overall_result, "ng_reasons": ctx.ng_reasons,
                       "match_method": ctx.match_method, "match_score": round(ctx.match_score, 4)})


# ---------------------------------------------------------------------------
# Operator registry: defines the execution order
# ---------------------------------------------------------------------------

def build_operator_chain():
    """Build the ordered list of (operator_func, group) tuples."""
    chain = []

    # Group: 定位摆正
    chain.append(("定位摆正", op_roi_extract))
    chain.append(("定位摆正", op_sift_detect))
    chain.append(("定位摆正", op_sift_match))
    chain.append(("定位摆正", op_ransac_pose))
    chain.append(("定位摆正", op_shape_match_fallback))
    chain.append(("定位摆正", op_warp_align))
    chain.append(("定位摆正", op_mask_apply))

    # Group: 边框-频域法
    chain.append(("边框-频域法", _make_border_preprocess("frequency", "频域")))
    chain.append(("边框-频域法", _make_border_freq_filter()))
    chain.append(("边框-频域法", _make_border_method_threshold("frequency", "频域", "border_frequency_diff_threshold")))

    # Group: 边框-空域法
    chain.append(("边框-空域法", _make_border_preprocess("spatial", "空域")))
    chain.append(("边框-空域法", _make_border_spatial_clahe()))
    chain.append(("边框-空域法", _make_border_spatial_gradient()))
    chain.append(("边框-空域法", _make_border_method_threshold("spatial", "空域", "border_gradient_threshold", "grad_diff")))

    # Group: 边框-模板法
    chain.append(("边框-模板法", _make_border_preprocess("template", "模板")))
    chain.append(("边框-模板法", _make_border_tpl_binary()))
    chain.append(("边框-模板法", _make_border_tpl_contour()))
    chain.append(("边框-模板法", _make_border_tpl_postproc()))

    # Group: 边框-差分法
    chain.append(("边框-差分法", _make_border_preprocess("difference", "差分")))
    chain.append(("边框-差分法", _make_border_diff_absdiff()))
    chain.append(("边框-差分法", _make_border_method_threshold("difference", "差分", "border_diff_diff_threshold")))

    # Group: 边框融合
    chain.append(("边框融合", op_border_fusion))
    chain.append(("边框融合", op_border_classify))

    # Group: 球检测
    chain.append(("球检测", op_ball_mask_proc))
    chain.append(("球检测", op_ball_binarize))
    chain.append(("球检测", op_ball_connect))
    chain.append(("球检测", op_ball_analyze))

    # Group: 表面检测 (reuse border operators with surface context)
    for method, label, thresh_param in [
        ("frequency", "频域", "surface_frequency_diff_threshold"),
        ("spatial", "空域", "surface_gradient_threshold"),
        ("template", "模板", "surface_binary_threshold"),
        ("difference", "差分", "surface_diff_diff_threshold"),
    ]:
        chain.append(("表面检测", _make_surface_preprocess(method, label)))
        # For frequency/spatial, add intermediate steps
        if method == "frequency":
            chain.append(("表面检测", _make_surface_freq_filter()))
        elif method == "spatial":
            chain.append(("表面检测", _make_surface_clahe()))
            chain.append(("表面检测", _make_surface_gradient()))
        elif method == "template":
            chain.append(("表面检测", _make_surface_tpl_binary()))
            chain.append(("表面检测", _make_surface_tpl_contour()))
        elif method == "difference":
            chain.append(("表面检测", _make_surface_diff_absdiff()))
        chain.append(("表面检测", _make_surface_method_threshold(method, label, thresh_param)))

    chain.append(("表面检测", op_surface_fusion))

    # Group: 结果汇总
    chain.append(("结果汇总", op_result_aggregate))

    return chain


# Surface-specific operator factories (mirror border but store in surface context)
def _make_surface_preprocess(method, label):
    def op(ctx):
        # Use surface mask (everything except border+ball)
        sm = np.ones_like(ctx.mask_border) * 255 if ctx.mask_border is not None else None
        if sm is not None:
            if ctx.mask_border is not None:
                sm[ctx.mask_border > 0] = 0
            if ctx.mask_ball is not None:
                sm[ctx.mask_ball > 0] = 0
        tpl_region, tgt_region = _preprocess_region(ctx.template, ctx.aligned_image, sm)
        ctx.set_surface_intermediate(method, "tpl_region", tpl_region)
        ctx.set_surface_intermediate(method, "tgt_region", tgt_region)
        vis = cv2.hconcat([tpl_region, tgt_region])
        return StepResult(f"surface_{method}_pre", f"表面-{label}: 预处理", _encode(vis),
                          {"region_w": tpl_region.shape[1]})
    return op

def _make_surface_freq_filter():
    def op(ctx):
        tpl = ctx.get_surface_intermediate("frequency", "tpl_region")
        tgt = ctx.get_surface_intermediate("frequency", "tgt_region")
        if tpl is None: return StepResult("surface_freq_fft", "表面-频域: FFT", None, status="skip")
        cutoff = ctx._p("surface_frequency_cutoff", 30.0)
        f_t = np.fft.fft2(tpl.astype(np.float32))
        f_d = np.fft.fft2(tgt.astype(np.float32))
        rows, cols = tpl.shape
        crow, ccol = rows // 2, cols // 2
        hp = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
        hp[dist <= cutoff] = 0
        tpl_hp = cv2.normalize(np.abs(np.fft.ifft2(f_t * hp)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tgt_hp = cv2.normalize(np.abs(np.fft.ifft2(f_d * hp)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ctx.set_surface_intermediate("frequency", "tpl_hp", tpl_hp)
        ctx.set_surface_intermediate("frequency", "tgt_hp", tgt_hp)
        return StepResult("surface_freq_fft", "表面-频域: FFT+高通", _encode(cv2.hconcat([tpl_hp, tgt_hp])),
                          {"cutoff": cutoff}, params=["surface_frequency_cutoff"])
    return op

def _make_surface_clahe():
    def op(ctx):
        tpl = ctx.get_surface_intermediate("spatial", "tpl_region")
        tgt = ctx.get_surface_intermediate("spatial", "tgt_region")
        if tpl is None: return StepResult("surface_spatial_clahe", "表面-空域: CLAHE", None, status="skip")
        clip = ctx._p("surface_clahe_clip_limit", 2)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        te = clahe.apply(tpl); de = clahe.apply(tgt)
        ctx.set_surface_intermediate("spatial", "tpl_enh", te)
        ctx.set_surface_intermediate("spatial", "tgt_enh", de)
        return StepResult("surface_spatial_clahe", "表面-空域: CLAHE", _encode(cv2.hconcat([te, de])),
                          {"clip_limit": clip}, params=["surface_clahe_clip_limit"])
    return op

def _make_surface_gradient():
    def op(ctx):
        tpl = ctx.get_surface_intermediate("spatial", "tpl_enh")
        tgt = ctx.get_surface_intermediate("spatial", "tgt_enh")
        if tpl is None: return StepResult("surface_spatial_grad", "表面-空域: 梯度", None, status="skip")
        td = cv2.GaussianBlur(tpl, (3, 3), 0); dd = cv2.GaussianBlur(tgt, (3, 3), 0)
        tm = cv2.normalize(cv2.magnitude(cv2.Sobel(td, cv2.CV_64F, 1, 0, 3), cv2.Sobel(td, cv2.CV_64F, 0, 1, 3)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dm = cv2.normalize(cv2.magnitude(cv2.Sobel(dd, cv2.CV_64F, 1, 0, 3), cv2.Sobel(dd, cv2.CV_64F, 0, 1, 3)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ctx.set_surface_intermediate("spatial", "grad_diff", cv2.absdiff(tm, dm))
        return StepResult("surface_spatial_grad", "表面-空域: Sobel梯度", _encode(cv2.hconcat([tm, dm])), {})
    return op

def _make_surface_tpl_binary():
    def op(ctx):
        tpl = ctx.get_surface_intermediate("template", "tpl_region")
        tgt = ctx.get_surface_intermediate("template", "tgt_region")
        if tpl is None: return StepResult("surface_tpl_binary", "表面-模板: 二值化", None, status="skip")
        bt = ctx._p("surface_binary_threshold", 0)
        if bt == 0:
            _, tb = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, db = cv2.threshold(tgt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, tb = cv2.threshold(tpl, bt, 255, cv2.THRESH_BINARY)
            _, db = cv2.threshold(tgt, bt, 255, cv2.THRESH_BINARY)
        ctx.set_surface_intermediate("template", "tpl_bin", tb)
        ctx.set_surface_intermediate("template", "tgt_bin", db)
        return StepResult("surface_tpl_binary", "表面-模板: 二值化", _encode(cv2.hconcat([tb, db])),
                          {"threshold": bt if bt > 0 else "OTSU"}, params=["surface_binary_threshold"])
    return op

def _make_surface_tpl_contour():
    def op(ctx):
        tb = ctx.get_surface_intermediate("template", "tpl_bin")
        db = ctx.get_surface_intermediate("template", "tgt_bin")
        if tb is None: return StepResult("surface_tpl_contour", "表面-模板: 轮廓差分", None, status="skip")
        thickness = ctx._p("surface_thickness", 3)
        tc, _ = cv2.findContours(tb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dc, _ = cv2.findContours(db, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tm = np.zeros_like(tb); dm = np.zeros_like(db)
        cv2.drawContours(tm, tc, -1, 255, thickness)
        cv2.drawContours(dm, dc, -1, 255, thickness)
        diff = cv2.absdiff(tm, dm)
        ctx.set_surface_intermediate("template", "diff", diff)
        return StepResult("surface_tpl_contour", "表面-模板: 轮廓差分", _encode(cv2.hconcat([tm, dm])),
                          {"thickness": thickness}, params=["surface_thickness"])
    return op

def _make_surface_diff_absdiff():
    def op(ctx):
        tpl = ctx.get_surface_intermediate("difference", "tpl_region")
        tgt = ctx.get_surface_intermediate("difference", "tgt_region")
        if tpl is None: return StepResult("surface_diff_absdiff", "表面-差分: 绝对差分", None, status="skip")
        diff = cv2.absdiff(tpl, tgt)
        ctx.set_surface_intermediate("difference", "diff", diff)
        return StepResult("surface_diff_absdiff", "表面-差分: 绝对差分", _encode(diff), {})
    return op

def _make_surface_method_threshold(method, label, thresh_param):
    def op(ctx):
        diff = ctx.get_surface_intermediate(method, "diff")
        if diff is None and method == "frequency":
            th = ctx.get_surface_intermediate("frequency", "tpl_hp")
            tg = ctx.get_surface_intermediate("frequency", "tgt_hp")
            if th is not None:
                diff = cv2.absdiff(th, tg)
                ctx.set_surface_intermediate("frequency", "diff", diff)
        if diff is None and method == "spatial":
            diff = ctx.get_surface_intermediate("spatial", "grad_diff")
        if diff is None and method == "template":
            diff = ctx.get_surface_intermediate("template", "diff")
            if diff is not None:
                _, binary = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)
            else:
                return StepResult(f"surface_{method}_thresh", f"表面-{label}: 二值化", None, status="skip")
        else:
            thresh = ctx._p(thresh_param, 30)
            _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        if method != "template":
            thresh = ctx._p(thresh_param, 30)
            _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        min_area = ctx._p("surface_min_defect_area", 10.0)
        kernelsz = ctx._p("surface_post_kernelsz", 3)
        sm = np.ones_like(ctx.mask_border) * 255 if ctx.mask_border is not None else None
        if sm is not None:
            if ctx.mask_border is not None: sm[ctx.mask_border > 0] = 0
            if ctx.mask_ball is not None: sm[ctx.mask_ball > 0] = 0
        result = _post_process(binary, sm, min_area, diff, kernelsz)
        ctx.set_surface_intermediate(method, "result", result)
        vis = _overlay(ctx.aligned_image, result["binary_defects"], (0, 255, 0))
        vis = _draw_contours(vis, result["defect_contours"], (0, 255, 0), 2)
        return StepResult(f"surface_{method}_thresh", f"表面-{label}: 二值化+后处理",
                          _encode(vis),
                          {"defect_px": result["defect_pixels"], "ratio": round(result["defect_ratio"], 4)},
                          params=[thresh_param, "surface_min_defect_area", "surface_post_kernelsz"])
    return op


def run_pipeline(ctx: PipelineContext, start_from: int = 0) -> List[StepResult]:
    """Execute the operator chain. Returns list of StepResults."""
    chain = build_operator_chain()
    results = []
    for i, (group, op_func) in enumerate(chain):
        if i < start_from:
            results.append(StepResult(op_func.__name__ if hasattr(op_func, '__name__') else f"op_{i}",
                                       group, status="skip"))
            continue
        try:
            t0 = time.time()
            sr = op_func(ctx)
            sr.metrics["time_ms"] = round((time.time() - t0) * 1000, 1)
            results.append(sr)
        except Exception as e:
            import traceback
            results.append(StepResult(f"op_{i}", group, status="error", error=str(e)))
    return results, chain
