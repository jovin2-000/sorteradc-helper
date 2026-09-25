"""Operators for template creation: mask generation from a reference image.

These mirror the logic in sorteradc-poc/getTemplate.py but decomposed into
individual debuggable operators.
"""
import json
import os
import numpy as np
import cv2
from datetime import datetime
from framework import Operator, Context, StepResult, _encode, to_gray, overlay_mask


def _get_roi(ctx):
    """Get ROI image from context, checking multiple keys."""
    roi = ctx.get("roi_optimized")
    if roi is None:
        roi = ctx.get("roi_image")
    if roi is None:
        roi = ctx.get("image")
    return roi


class ROISelect(Operator):
    """Interactive ROI selection (user draws a rectangle on the image)."""

    def __init__(self):
        super().__init__("roi_select", "ROI选择", "模板创建",
                         output_keys=["roi_rect", "roi_image"])

    def execute(self, ctx: Context) -> StepResult:
        img = ctx.get("image")
        roi = ctx.get("user_roi")
        if roi:
            x, y, w, h = roi
            ctx.set("roi_rect", (x, y, x + w, y + h))
            ctx.set("roi_image", img[y:y + h, x:x + w])
            vis = img.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return self._result(vis, {"x": x, "y": y, "w": w, "h": h})
        return self._result(img, {"needs_input": True}, status="interactive",
                            interactive_data={"type": "rect_select", "image": _encode(img)})


class BoundaryOptimize(Operator):
    """Auto-optimize ROI boundary via OTSU/threshold + largest contour."""

    def __init__(self):
        super().__init__("boundary_optimize", "边界优化", "模板创建",
                         param_keys=["binarization_threshold", "border_x", "border_y"],
                         input_keys=["roi_image"],
                         output_keys=["roi_optimized"])

    def execute(self, ctx: Context) -> StepResult:
        roi = ctx.get("roi_image")
        if roi is None:
            roi = ctx.get("image")
        if roi is None:
            return self._result(None, status="skip")
        gray = to_gray(roi)
        bt = ctx.p("binarization_threshold", 0)
        if bt == 0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, bt, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(best)
            bx = ctx.p("border_x", 5)
            by = ctx.p("border_y", 5)
            h_img, w_img = roi.shape[:2]
            y0 = max(0, y - by)
            y1 = min(h_img, y + h + by)
            x0 = max(0, x - bx)
            x1 = min(w_img, x + w + bx)
            optimized = roi[y0:y1, x0:x1]
            ctx.set("roi_optimized", optimized)
            vis = roi.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return self._result(vis, {"contours": len(contours),
                                       "area": int(cv2.contourArea(best)),
                                       "bbox": f"{w}x{h}"})
        ctx.set("roi_optimized", roi)
        return self._result(roi, {"contours": 0}, status="fail")


class FeatureExtract(Operator):
    """Extract template gradient features (gx, gy, mag, contour, xym)."""

    def __init__(self):
        super().__init__("feature_extract", "特征提取", "模板创建",
                         input_keys=["roi_optimized"],
                         output_keys=["tpl_gx", "tpl_gy", "tpl_mag", "tpl_contour", "tpl_xym"])

    def execute(self, ctx: Context) -> StepResult:
        roi = _get_roi(ctx)
        if roi is None:
            return self._result(None, status="skip")
        gray = to_gray(roi)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_pts = np.vstack(contours) if contours else np.array([[0, 0]])
        ctx.set("tpl_gx", gx)
        ctx.set("tpl_gy", gy)
        ctx.set("tpl_mag", mag)
        ctx.set("tpl_contour", contour_pts)
        ctx.set("tpl_xym", np.array([]))
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
        return self._result(vis, {"contours": len(contours),
                                   "feature_pts": len(contour_pts)})


class ForegroundMask(Operator):
    """Generate foreground mask via OTSU/threshold + morphology."""

    def __init__(self):
        super().__init__("foreground_mask", "前景掩码", "模板创建",
                         param_keys=["binarization_threshold", "border_kernel_size"],
                         input_keys=["roi_optimized"],
                         output_keys=["mask_foreground"])

    def execute(self, ctx: Context) -> StepResult:
        roi = _get_roi(ctx)
        if roi is None:
            return self._result(None, status="skip")
        gray = to_gray(roi)
        bt = ctx.p("binarization_threshold", 0)
        if bt == 0:
            _, mask_all = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask_all = cv2.threshold(gray, bt, 255, cv2.THRESH_BINARY)
        ks = ctx.p("border_kernel_size", 3)
        kernel = np.ones((ks, ks), np.uint8)
        mask_fg = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
        ctx.set("mask_foreground", mask_fg)
        return self._result(mask_fg, {"fg_px": int(np.sum(mask_fg > 0))})


class BorderMask(Operator):
    """Extract border mask from foreground mask."""

    def __init__(self):
        super().__init__("border_mask", "边框掩码", "模板创建",
                         param_keys=["border_width", "border_method"],
                         input_keys=["mask_foreground"],
                         output_keys=["mask_border"])

    def execute(self, ctx: Context) -> StepResult:
        fg = ctx.get("mask_foreground")
        if fg is None:
            return self._result(None, status="skip")
        bw = ctx.p("border_width", 8)
        method = ctx.p("border_method", "direct")
        _, binary = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        if method == "distance":
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outer = np.zeros_like(binary)
            cv2.drawContours(outer, contours, -1, 255, -1)
            dist = cv2.distanceTransform(outer, cv2.DIST_L2, 5)
            border = np.zeros_like(binary)
            border[(dist <= bw) & (outer > 0)] = 255
        else:
            h, w = binary.shape
            border = np.zeros_like(binary)
            cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cont:
                r = cv2.boundingRect(cont[0])
                cv2.rectangle(binary, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), 255, -1)
            border[0:bw, :] = binary[0:bw, :]
            border[h-bw:h, :] = binary[h-bw:h, :]
            border[:, 0:bw] = binary[:, 0:bw]
            border[:, w-bw:w] = binary[:, w-bw:w]
            border = cv2.bitwise_and(border, binary)
        ctx.set("mask_border", border)
        roi = _get_roi(ctx)
        vis = overlay_mask(roi if roi is not None else np.zeros_like(border), border, (0, 255, 0))
        return self._result(vis, {"border_px": int(np.sum(border > 0)), "method": method})


class BallMask(Operator):
    """Extract ball mask via morphology + contour area/circularity filtering."""

    def __init__(self):
        super().__init__("ball_mask", "球掩码", "模板创建",
                         param_keys=["tc_ball_min_area", "tc_ball_max_area",
                                     "tc_ball_circularity", "tc_ball_kernel_size"],
                         input_keys=["mask_foreground"],
                         output_keys=["mask_ball"])

    def execute(self, ctx: Context) -> StepResult:
        fg = ctx.get("mask_foreground")
        if fg is None:
            return self._result(None, status="skip")
        ks = ctx.p("tc_ball_kernel_size", 3)
        min_area = ctx.p("tc_ball_min_area", 500)
        max_area = ctx.p("tc_ball_max_area", 2500)
        circ_th = ctx.p("tc_ball_circularity", 0.6)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        opened = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ball_mask = np.zeros_like(fg)
        ball_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                perim = cv2.arcLength(c, True)
                if perim > 0:
                    circ = 4 * np.pi * area / (perim * perim)
                    if circ > circ_th:
                        ball_count += 1
                        cv2.fillPoly(ball_mask, [c], 255)
        ctx.set("mask_ball", ball_mask)
        roi = _get_roi(ctx)
        vis = overlay_mask(roi if roi is not None else np.zeros_like(fg), ball_mask, (255, 100, 0))
        return self._result(vis, {"ball_count": ball_count,
                                   "mask_px": int(np.sum(ball_mask > 0))})


class TraceMask(Operator):
    """Extract trace mask (optional, subtracts border and ball)."""

    def __init__(self):
        super().__init__("trace_mask", "轨迹掩码", "模板创建",
                         param_keys=["trace_threshold", "trace_kernel", "trace_iterations"],
                         input_keys=["roi_optimized", "mask_border", "mask_ball"],
                         output_keys=["mask_trace", "mask_expand"])

    def execute(self, ctx: Context) -> StepResult:
        roi = _get_roi(ctx)
        if roi is None:
            return self._result(None, status="skip")
        gray = to_gray(roi)
        border = ctx.get("mask_border")
        ball = ctx.get("mask_ball")
        th = ctx.p("trace_threshold", 100)
        ks = ctx.p("trace_kernel", 3)
        iters = ctx.p("trace_iterations", 1)
        _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
        if ks != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iters)
        if border is not None:
            cont, _ = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            expand = np.zeros_like(border)
            cv2.drawContours(expand, cont, -1, 255, -1)
            binary = cv2.bitwise_and(binary, expand)
            expand = cv2.bitwise_not(expand)
            not_border = cv2.bitwise_not(border)
            binary = cv2.bitwise_and(binary, not_border)
        else:
            expand = np.ones_like(gray) * 255
        if ball is not None:
            binary = np.where(ball != 0, 0, binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        trace = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iters)
        ctx.set("mask_trace", trace)
        ctx.set("mask_expand", expand if border is not None else None)
        vis = overlay_mask(roi, trace, (255, 255, 0), 0.3)
        return self._result(vis, {"trace_px": int(np.sum(trace > 0))})


class SaveTemplate(Operator):
    """Save the created template to disk in sorteradc-poc format."""

    def __init__(self):
        super().__init__("save_template", "保存模板", "模板创建",
                         input_keys=["roi_optimized", "mask_foreground", "mask_border",
                                     "mask_ball", "mask_trace", "mask_expand",
                                     "tpl_gx", "tpl_gy", "tpl_mag", "tpl_contour", "tpl_xym"],
                         output_keys=["template_saved"])

    def execute(self, ctx: Context) -> StepResult:
        product_id = ctx.get("new_product_id", "")
        roi = _get_roi(ctx)

        if not product_id or roi is None:
            ctx.set("template_ready", True)
            ctx.set("template_image", roi)
            vis = roi.copy() if roi is not None else None
            if vis is not None and len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            for mask, color in [(ctx.get("mask_border"), (0, 255, 0)),
                                (ctx.get("mask_ball"), (255, 100, 0)),
                                (ctx.get("mask_trace"), (255, 255, 0))]:
                if mask is not None and vis is not None and mask.shape[:2] == vis.shape[:2]:
                    mb = mask > 0
                    ov = vis.copy()
                    ov[mb] = color
                    vis = cv2.addWeighted(ov, 0.3, vis, 0.7, 0)
            return self._result(vis, {"ready": True, "product_id": product_id or "(not set)"})

        # Save to disk
        poc_root = os.environ.get("POC_ROOT", "/app")
        tpl_dir = os.path.join(poc_root, "data", "templates", product_id, "version1", "main")
        os.makedirs(tpl_dir, exist_ok=True)

        np.save(os.path.join(tpl_dir, "roi.npy"), roi)
        cv2.imwrite(os.path.join(tpl_dir, "roi.png"), roi)
        for name in ["tpl_gx", "tpl_gy", "tpl_mag", "tpl_contour", "tpl_xym"]:
            arr = ctx.get(name)
            if arr is not None:
                np.save(os.path.join(tpl_dir, f"{name}.npy"), arr)

        mask_map = {"mask_all_foreground": ctx.get("mask_foreground"),
                    "mask_ball_only": ctx.get("mask_ball"),
                    "mask_border_only": ctx.get("mask_border"),
                    "mask_trace_only": ctx.get("mask_trace"),
                    "mask_expand_only": ctx.get("mask_expand")}
        for name, mask in mask_map.items():
            if mask is not None:
                np.save(os.path.join(tpl_dir, f"{name}.npy"), mask)
                cv2.imwrite(os.path.join(tpl_dir, f"{name}.png"), mask)

        h, w = roi.shape[:2]
        meta = {"product_id": product_id, "version": 1, "state": "published",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": "helper", "width": w, "height": h,
                "canonical_roi_size": [w, h], "shape_match_pyramid_levels": 3,
                "shape_match_min_score": 0.7,
                "detection_params": ctx.get("params", {}),
                "sample_count": 1, "avg_match_score": 0.85,
                "align_error": 2.0, "mask_boundary_stability": 0.95,
                "_ndarray_files": ["shape_match_template", "shape_match_tpl_gx",
                                   "shape_match_tpl_gy", "shape_match_tpl_mag",
                                   "shape_match_tpl_contour_points", "shape_match_tpl_xym",
                                   "mask_all_foreground", "mask_ball_only",
                                   "mask_border_only", "mask_trace_only", "mask_expand_only"]}
        with open(os.path.join(tpl_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)

        ctx.set("template_saved", True)
        vis = roi.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        colors = {"mask_border_only": (0, 255, 0), "mask_ball_only": (255, 100, 0),
                  "mask_trace_only": (255, 255, 0)}
        for name, mask in mask_map.items():
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                mb = mask > 0
                c = colors.get(name, (100, 100, 100))
                ov = vis.copy()
                ov[mb] = c
                vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)
        return self._result(vis, {"saved": True, "product_id": product_id,
                                  "path": tpl_dir, "files": len(mask_map) + 6})
