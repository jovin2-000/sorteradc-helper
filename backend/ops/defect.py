"""Operators for defect detection: frequency, spatial, template, difference.

Each method is decomposed into sub-step operators. A generic factory
creates parameterized operators for both border and surface detection
by varying the mask and context key prefix.
"""
import numpy as np
import cv2
from framework import (Operator, Context, StepResult, _encode, to_gray, overlay_mask,
                        draw_contours, post_process_defects, preprocess_region,
                        compute_surface_mask, hconcat_images)


# ---------------------------------------------------------------------------
# Preprocess operator (shared by all methods)
# ---------------------------------------------------------------------------

class RegionPreprocess(Operator):
    """Extract gray template/target regions within a mask."""

    def __init__(self, region: str, label: str, use_surface_mask=False):
        super().__init__(
            f"{region}_preprocess", f"{label}: 预处理", region,
            input_keys=["template", "aligned_image", "mask_border", "mask_ball"],
            output_keys=[f"{region}_tpl_region", f"{region}_tgt_region"])
        self.region = region
        self.label = label
        self.use_surface = use_surface_mask

    def execute(self, ctx: Context) -> StepResult:
        if self.use_surface:
            mask = compute_surface_mask(ctx.get("mask_border"), ctx.get("mask_ball"))
        else:
            mask = ctx.get("mask_border")
        tpl_r, tgt_r = preprocess_region(ctx.get("template"), ctx.get("aligned_image"), mask)
        ctx.set(f"{self.region}_tpl_region", tpl_r)
        ctx.set(f"{self.region}_tgt_region", tgt_r)
        vis = hconcat_images([tpl_r, tgt_r])
        return self._result(vis, {"w": tpl_r.shape[1], "h": tpl_r.shape[0]})


# ---------------------------------------------------------------------------
# Frequency domain: FFT + highpass + inverse
# ---------------------------------------------------------------------------

class FreqFilter(Operator):
    """FFT + highpass filter + inverse FFT."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_freq_fft", f"{label}: FFT+高通滤波", region,
            param_keys=["border_frequency_cutoff"] if "border" in region else ["surface_frequency_cutoff"],
            input_keys=[f"{region}_tpl_region", f"{region}_tgt_region"],
            output_keys=[f"{region}_tpl_hp", f"{region}_tgt_hp"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tpl = ctx.get(f"{self.region}_tpl_region")
        tgt = ctx.get(f"{self.region}_tgt_region")
        if tpl is None:
            return self._result(None, status="skip")
        cutoff_key = "surface_frequency_cutoff" if "surface" in self.region else "border_frequency_cutoff"
        cutoff = ctx.p(cutoff_key, 30.0)

        f_t = np.fft.fft2(tpl.astype(np.float32))
        f_d = np.fft.fft2(tgt.astype(np.float32))
        rows, cols = tpl.shape
        crow, ccol = rows // 2, cols // 2
        hp = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        hp[dist <= cutoff] = 0

        tpl_hp = cv2.normalize(np.abs(np.fft.ifft2(f_t * hp)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tgt_hp = cv2.normalize(np.abs(np.fft.ifft2(f_d * hp)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ctx.set(f"{self.region}_tpl_hp", tpl_hp)
        ctx.set(f"{self.region}_tgt_hp", tgt_hp)
        return self._result(hconcat_images([tpl_hp, tgt_hp]), {"cutoff": cutoff})


# ---------------------------------------------------------------------------
# Spatial: CLAHE + Sobel gradient
# ---------------------------------------------------------------------------

class SpatialCLAHE(Operator):
    """CLAHE contrast enhancement."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_clahe", f"{label}: CLAHE增强", region,
            param_keys=["border_clahe_clip_limit"] if "border" in region else ["surface_clahe_clip_limit"],
            input_keys=[f"{region}_tpl_region", f"{region}_tgt_region"],
            output_keys=[f"{region}_tpl_enh", f"{region}_tgt_enh"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tpl = ctx.get(f"{self.region}_tpl_region")
        tgt = ctx.get(f"{self.region}_tgt_region")
        if tpl is None:
            return self._result(None, status="skip")
        key = "surface_clahe_clip_limit" if "surface" in self.region else "border_clahe_clip_limit"
        clip = ctx.p(key, 2)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        te = clahe.apply(tpl)
        de = clahe.apply(tgt)
        ctx.set(f"{self.region}_tpl_enh", te)
        ctx.set(f"{self.region}_tgt_enh", de)
        return self._result(hconcat_images([te, de]), {"clip": clip})


class SpatialGradient(Operator):
    """Sobel gradient computation + magnitude diff."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_gradient", f"{label}: Sobel梯度", region,
            input_keys=[f"{region}_tpl_enh", f"{region}_tgt_enh"],
            output_keys=[f"{region}_grad_diff"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tpl = ctx.get(f"{self.region}_tpl_enh")
        tgt = ctx.get(f"{self.region}_tgt_enh")
        if tpl is None:
            return self._result(None, status="skip")
        td = cv2.GaussianBlur(tpl, (3, 3), 0)
        dd = cv2.GaussianBlur(tgt, (3, 3), 0)
        tm = cv2.normalize(cv2.magnitude(cv2.Sobel(td, cv2.CV_64F, 1, 0, 3),
                                          cv2.Sobel(td, cv2.CV_64F, 0, 1, 3)),
                           None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dm = cv2.normalize(cv2.magnitude(cv2.Sobel(dd, cv2.CV_64F, 1, 0, 3),
                                          cv2.Sobel(dd, cv2.CV_64F, 0, 1, 3)),
                           None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_diff = cv2.absdiff(tm, dm)
        ctx.set(f"{self.region}_grad_diff", grad_diff)
        return self._result(hconcat_images([tm, dm]), {})


# ---------------------------------------------------------------------------
# Template comparison: binary threshold + contour diff
# ---------------------------------------------------------------------------

class TplBinary(Operator):
    """Binary threshold (OTSU or fixed)."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_binary", f"{label}: 二值化", region,
            param_keys=["border_binary_threshold"] if "border" in region else ["surface_binary_threshold"],
            input_keys=[f"{region}_tpl_region", f"{region}_tgt_region"],
            output_keys=[f"{region}_tpl_bin", f"{region}_tgt_bin"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tpl = ctx.get(f"{self.region}_tpl_region")
        tgt = ctx.get(f"{self.region}_tgt_region")
        if tpl is None:
            return self._result(None, status="skip")
        key = "surface_binary_threshold" if "surface" in self.region else "border_binary_threshold"
        bt = ctx.p(key, 0)
        if bt == 0:
            _, tb = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, db = cv2.threshold(tgt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, tb = cv2.threshold(tpl, bt, 255, cv2.THRESH_BINARY)
            _, db = cv2.threshold(tgt, bt, 255, cv2.THRESH_BINARY)
        ctx.set(f"{self.region}_tpl_bin", tb)
        ctx.set(f"{self.region}_tgt_bin", db)
        return self._result(hconcat_images([tb, db]), {"threshold": bt if bt > 0 else "OTSU"})


class TplContourDiff(Operator):
    """Contour extraction + contour difference."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_contour_diff", f"{label}: 轮廓差分", region,
            param_keys=["border_thickness"] if "border" in region else ["surface_thickness"],
            input_keys=[f"{region}_tpl_bin", f"{region}_tgt_bin"],
            output_keys=[f"{region}_diff"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tb = ctx.get(f"{self.region}_tpl_bin")
        db = ctx.get(f"{self.region}_tgt_bin")
        if tb is None:
            return self._result(None, status="skip")
        key = "surface_thickness" if "surface" in self.region else "border_thickness"
        thickness = ctx.p(key, 1)
        tc, _ = cv2.findContours(tb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dc, _ = cv2.findContours(db, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tm = np.zeros_like(tb)
        dm = np.zeros_like(db)
        cv2.drawContours(tm, tc, -1, 255, thickness)
        cv2.drawContours(dm, dc, -1, 255, thickness)
        diff = cv2.absdiff(tm, dm)
        ctx.set(f"{self.region}_diff", diff)
        return self._result(hconcat_images([tm, dm]), {"tpl_contours": len(tc), "tgt_contours": len(dc)})


# ---------------------------------------------------------------------------
# Difference method: abs diff
# ---------------------------------------------------------------------------

class AbsDiff(Operator):
    """Absolute difference between template and target."""

    def __init__(self, region: str, label: str):
        super().__init__(
            f"{region}_absdiff", f"{label}: 绝对差分", region,
            input_keys=[f"{region}_tpl_region", f"{region}_tgt_region"],
            output_keys=[f"{region}_diff"])
        self.region = region

    def execute(self, ctx: Context) -> StepResult:
        tpl = ctx.get(f"{self.region}_tpl_region")
        tgt = ctx.get(f"{self.region}_tgt_region")
        if tpl is None:
            return self._result(None, status="skip")
        diff = cv2.absdiff(tpl, tgt)
        ctx.set(f"{self.region}_diff", diff)
        return self._result(diff, {})


# ---------------------------------------------------------------------------
# Threshold + post-process (shared by all methods)
# ---------------------------------------------------------------------------

class ThresholdPostproc(Operator):
    """Binary threshold + morphology + contour extraction + area filter."""

    def __init__(self, region: str, label: str, method: str,
                 thresh_param: str, diff_key: str = "diff"):
        super().__init__(
            f"{region}_threshold", f"{label}: 二值化+后处理", region,
            param_keys=[thresh_param, "border_min_defect_area", "border_post_kernelsz"]
            if "border" in region else
            [thresh_param, "surface_min_defect_area", "surface_post_kernelsz"],
            input_keys=[f"{region}_{diff_key}", "mask_border", "mask_ball", "aligned_image"],
            output_keys=[f"{region}_result"])
        self.region = region
        self.method = method
        self.thresh_param = thresh_param
        self.diff_key = diff_key

    def execute(self, ctx: Context) -> StepResult:
        # Get the diff image (try diff_key, then fallback to computed hp diff)
        diff = ctx.get(f"{self.region}_{self.diff_key}")
        if diff is None and self.method == "frequency":
            th = ctx.get(f"{self.region}_tpl_hp")
            tg = ctx.get(f"{self.region}_tgt_hp")
            if th is not None:
                diff = cv2.absdiff(th, tg)
                ctx.set(f"{self.region}_diff", diff)
        if diff is None and self.method == "spatial":
            diff = ctx.get(f"{self.region}_grad_diff")
        if diff is None:
            return self._result(None, status="skip")

        # Threshold
        if self.method == "template":
            _, binary = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)
        else:
            thresh = ctx.p(self.thresh_param, 30)
            _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        # Region mask
        if "surface" in self.region:
            mask = compute_surface_mask(ctx.get("mask_border"), ctx.get("mask_ball"))
        else:
            mask = ctx.get("mask_border")

        min_area = ctx.p("surface_min_defect_area" if "surface" in self.region else "border_min_defect_area", 10.0)
        ks = ctx.p("surface_post_kernelsz" if "surface" in self.region else "border_post_kernelsz", 3)

        # For template method, do its own morphology
        if self.method == "template":
            if ks > 1:
                kernel = np.ones((ks, ks), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            result = post_process_defects(binary, mask, min_area, diff, 1)
        else:
            result = post_process_defects(binary, mask, min_area, diff, ks)

        ctx.set(f"{self.region}_result", result)
        aligned = ctx.get("aligned_image")
        color = (0, 255, 0) if "surface" in self.region else (0, 0, 255)
        vis = overlay_mask(aligned, result["binary_defects"], color)
        vis = draw_contours(vis, result["defect_contours"], color, 2)
        return self._result(vis, {"defect_px": result["defect_pixels"],
                                   "ratio": round(result["defect_ratio"], 4)})
