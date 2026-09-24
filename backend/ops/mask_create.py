"""Operators for template creation: mask generation from a reference image.

These mirror the logic in sorteradc-poc/getTemplate.py but decomposed into
individual debuggable operators.
"""
import numpy as np
import cv2
from framework import Operator, Context, StepResult, _encode, to_gray, overlay_mask, hconcat_images


class ROISelect(Operator):
    """Interactive ROI selection (user draws a rectangle on the image)."""

    def __init__(self):
        super().__init__("roi_select", "ROI选择", "模板创建",
                         output_keys=["roi_rect", "roi_image"])

    def execute(self, ctx: Context) -> StepResult:
        img = ctx.get("image")
        # Check if ROI was provided externally (from frontend canvas)
        roi = ctx.get("user_roi")
        if roi:
            x, y, w, h = roi
            ctx.set("roi_rect", (x, y, x + w, y + h))
            ctx.set("roi_image", img[y:y + h, x:x + w])
            vis = img.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return self._result(vis, {"x": x, "y": y, "w": w, "h": h})
        # No ROI provided: return interactive status to prompt frontend
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
        roi = ctx.get("roi_image") or ctx.get("roi_optimized") or ctx.get("image")
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
            optimized = roi[max(0, y - by):y + h + by, max(0, x - bx):x + w + bx]
            ctx.set("roi_optimized", optimized)
            vis = roi.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return self._result(vis, {"contours": len(contours), "area": cv2.contourArea(best),
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
        roi = ctx.get("roi_optimized") or ctx.get("roi_image")
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
        ctx.set("tpl_xym", [])  # simplified

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
        roi = ctx.get("roi_optimized") or ctx.get("roi_image")
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
        else:  # direct
            h, w = binary.shape
            border = np.zeros_like(binary)
            cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cont:
                r = cv2.boundingRect(cont[0])
                cv2.rectangle(binary, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), 255, -1)
            border[0:bw, :] = binary[0:bw, :]
            border[h - bw:h, :] = binary[h - bw:h, :]
            border[:, 0:bw] = binary[:, 0:bw]
            border[:, w - bw:w] = binary[:, w - bw:w]
            border = cv2.bitwise_and(border, binary)

        ctx.set("mask_border", border)
        vis = overlay_mask(ctx.get("roi_optimized") or ctx.get("roi_image", np.zeros_like(border)), border, (0, 255, 0))
        return self._result(vis, {"border_px": int(np.sum(border > 0)), "method": method})


class BallMask(Operator):
    """Extract ball mask via morphology + contour area/circularity filtering."""

    def __init__(self):
        super().__init__("ball_mask", "球掩码", "模板创建",
                         param_keys=["ball_min_area", "ball_max_area", "ball_circularity_min", "ball_kernel_size"],
                         input_keys=["mask_foreground"],
                         output_keys=["mask_ball"])

    def execute(self, ctx: Context) -> StepResult:
        fg = ctx.get("mask_foreground")
        if fg is None:
            return self._result(None, status="skip")
        ks = ctx.p("ball_kernel_size", 3)
        min_area = ctx.p("ball_min_area", 500)
        max_area = ctx.p("ball_max_area", 2500)
        circ_th = ctx.p("ball_circularity_min", 0.6)

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
        vis = overlay_mask(ctx.get("roi_optimized") or ctx.get("roi_image", np.zeros_like(fg)), ball_mask, (255, 100, 0))
        return self._result(vis, {"ball_count": ball_count, "mask_px": int(np.sum(ball_mask > 0))})


class TraceMask(Operator):
    """Extract trace mask (optional, subtracts border and ball)."""

    def __init__(self):
        super().__init__("trace_mask", "轨迹掩码", "模板创建",
                         param_keys=["trace_threshold", "trace_kernel", "trace_iterations"],
                         input_keys=["roi_optimized", "mask_border", "mask_ball"],
                         output_keys=["mask_trace", "mask_expand"])

    def execute(self, ctx: Context) -> StepResult:
        roi = ctx.get("roi_optimized") or ctx.get("roi_image")
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

        # Expand border mask
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
    """Save the created template to disk."""

    def __init__(self):
        super().__init__("save_template", "保存模板", "模板创建",
                         input_keys=["roi_optimized", "mask_foreground", "mask_border",
                                     "mask_ball", "mask_trace", "tpl_gx", "tpl_gy"])

    def execute(self, ctx: Context) -> StepResult:
        # This operator triggers saving logic via the API layer
        # For now, just visualize all masks together
        roi = ctx.get("roi_optimized") or ctx.get("roi_image")
        vis = roi.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        masks = [(ctx.get("mask_border"), (0, 255, 0)),
                 (ctx.get("mask_ball"), (255, 100, 0)),
                 (ctx.get("mask_trace"), (255, 255, 0))]
        for mask, color in masks:
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                mb = mask > 0
                ov = vis.copy()
                ov[mb] = color
                vis = cv2.addWeighted(ov, 0.3, vis, 0.7, 0)

        ctx.set("template_ready", True)
        ctx.set("template_image", roi)
        return self._result(vis, {"ready": True, "masks": sum(1 for m, _ in masks if m is not None)})
