"""Operators for result fusion, classification, and final aggregation."""
import numpy as np
import cv2
from framework import (Operator, Context, StepResult, _encode, overlay_mask, draw_contours,
                        post_process_defects, compute_surface_mask)


class BorderFusion(Operator):
    """Weighted voting fusion of multiple border defect methods."""

    def __init__(self):
        super().__init__("border_fusion", "边框-融合结果", "边框融合",
                         param_keys=["border_methods", "border_weights", "border_vote_threshold",
                                     "border_fusion_kernelsz", "border_fusion_skip_open",
                                     "border_min_defect_area"],
                         input_keys=["border_result", "mask_border", "aligned_image"],
                         output_keys=["border_fused_result"])

    def execute(self, ctx: Context) -> StepResult:
        methods = ctx.p("border_methods", ["frequency", "spatial", "template", "difference"])
        weights = ctx.p("border_weights", {})
        vote_th = ctx.p("border_vote_threshold", 0.5)
        min_area = ctx.p("border_min_defect_area", 10.0)
        ks = ctx.p("border_fusion_kernelsz", 3)
        skip_open = ctx.p("border_fusion_skip_open", False)

        masks = []
        w_arr = []
        for m in methods:
            r = ctx.get(f"border_{m}_result") or ctx.get(f"border_result")
            if r is None:
                # Try to get from per-method context key
                r = ctx.get(f"border_{m}_threshold_result")
            if r is not None and isinstance(r, dict) and "binary_defects" in r:
                masks.append(r["binary_defects"].astype(np.float32) / 255.0)
                w_arr.append(weights.get(m, 0.0))

        if not masks:
            return self._result(None, {"error": "no method results"}, status="skip")

        w_arr = np.array(w_arr)
        w_arr = w_arr / w_arr.sum()
        fused = np.zeros_like(masks[0])
        for i, m in enumerate(masks):
            fused += m * w_arr[i]
        _, fb = cv2.threshold((fused * 255).astype(np.uint8), int(vote_th * 255), 255, cv2.THRESH_BINARY)
        if ks > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ks, ks))
            fb = cv2.morphologyEx(fb, cv2.MORPH_CLOSE, kernel)
            if not skip_open:
                fb = cv2.morphologyEx(fb, cv2.MORPH_OPEN, kernel)

        mask = ctx.get("mask_border")
        result = post_process_defects(fb, mask, min_area, (fused * 255).astype(np.uint8), 1)
        ctx.set("border_fused_result", result)
        aligned = ctx.get("aligned_image")
        vis = overlay_mask(aligned, result["binary_defects"], (0, 0, 255))
        vis = draw_contours(vis, result["defect_contours"], (0, 0, 255), 2)
        is_ng = result["defect_pixels"] >= min_area
        if is_ng:
            ctx.set("overall_result", "NG")
            reasons = ctx.get("ng_reasons", [])
            reasons.append(f"边框缺陷: {result['defect_pixels']}px")
            ctx.set("ng_reasons", reasons)
        return self._result(vis, {"defect_px": result["defect_pixels"],
                                   "ratio": round(result["defect_ratio"], 4),
                                   "is_ng": is_ng, "methods": len(masks)})


class BorderClassify(Operator):
    """Classify border defects as corner (崩角) or edge (崩边)."""

    def __init__(self):
        super().__init__("border_classify", "边框-崩角/崩边分类", "边框融合",
                         param_keys=["border_corner_ratio"],
                         input_keys=["border_fused_result", "aligned_image"])

    def execute(self, ctx: Context) -> StepResult:
        result = ctx.get("border_fused_result")
        if result is None:
            return self._result(None, status="skip")
        ratio = ctx.p("border_corner_ratio", 0.18)
        aligned = ctx.get("aligned_image")
        h, w = aligned.shape[:2]
        cs = max(1, int(min(w, h) * ratio))
        corners = [(0, 0), (w - cs, 0), (0, h - cs), (w - cs, h - cs)]
        subtypes = []
        for c in result["defect_contours"]:
            x, y, cw, ch = cv2.boundingRect(c)
            is_corner = any(x < cx0 + cs and cx0 < x + cw and
                            y < cy0 + cs and cy0 < y + ch for cx0, cy0 in corners)
            subtypes.append("崩角" if is_corner else "崩边")
        vis = overlay_mask(aligned, result["binary_defects"], (0, 0, 255))
        for cx0, cy0 in corners:
            cv2.rectangle(vis, (cx0, cy0), (cx0 + cs, cy0 + cs), (0, 255, 255), 1)
        return self._result(vis, {"崩角": subtypes.count("崩角"), "崩边": subtypes.count("崩边")})


class SurfaceFusion(Operator):
    """Weighted voting fusion of surface defect methods."""

    def __init__(self):
        super().__init__("surface_fusion", "表面-融合结果", "表面融合",
                         param_keys=["surface_methods", "surface_weights", "surface_vote_threshold",
                                     "surface_min_defect_area"],
                         input_keys=["mask_border", "mask_ball", "aligned_image"],
                         output_keys=["surface_fused_result"])

    def execute(self, ctx: Context) -> StepResult:
        methods = ctx.p("surface_methods", ["frequency", "spatial", "template"])
        weights = ctx.p("surface_weights", {})
        vote_th = ctx.p("surface_vote_threshold", 0.5)
        min_area = ctx.p("surface_min_defect_area", 10.0)

        sm = compute_surface_mask(ctx.get("mask_border"), ctx.get("mask_ball"))
        masks = []
        w_arr = []
        for m in methods:
            r = ctx.get(f"surface_{m}_result")
            if r is not None and "binary_defects" in r:
                masks.append(r["binary_defects"].astype(np.float32) / 255.0)
                w_arr.append(weights.get(m, 0.0))
        if not masks:
            return self._result(None, {"error": "no methods"}, status="skip")
        w_arr = np.array(w_arr)
        w_arr = w_arr / w_arr.sum()
        fused = np.zeros_like(masks[0])
        for i, mk in enumerate(masks):
            fused += mk * w_arr[i]
        _, fb = cv2.threshold((fused * 255).astype(np.uint8), int(vote_th * 255), 255, cv2.THRESH_BINARY)
        result = post_process_defects(fb, sm, min_area, (fused * 255).astype(np.uint8), 1)
        ctx.set("surface_fused_result", result)
        aligned = ctx.get("aligned_image")
        vis = overlay_mask(aligned, result["binary_defects"], (0, 255, 0))
        vis = draw_contours(vis, result["defect_contours"], (0, 255, 0), 2)
        is_ng = result["defect_pixels"] >= min_area
        if is_ng:
            ctx.set("overall_result", "NG")
            reasons = ctx.get("ng_reasons", [])
            reasons.append(f"表面缺陷: {result['defect_pixels']}px")
            ctx.set("ng_reasons", reasons)
        return self._result(vis, {"defect_px": result["defect_pixels"],
                                   "ratio": round(result["defect_ratio"], 4),
                                   "is_ng": is_ng})


class ResultAggregate(Operator):
    """Final result aggregation and visualization."""

    def __init__(self):
        super().__init__("result_aggregate", "最终结果", "结果汇总",
                         input_keys=["aligned_image", "border_fused_result",
                                     "surface_fused_result", "overall_result", "ng_reasons"])

    def execute(self, ctx: Context) -> StepResult:
        aligned = ctx.get("aligned_image")
        overall = ctx.get("overall_result", "OK")
        reasons = ctx.get("ng_reasons", [])

        # Build a simple visualization
        vis = aligned.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Overlay all defect contours
        for result_key, color in [("border_fused_result", (0, 0, 255)),
                                   ("surface_fused_result", (0, 255, 0))]:
            r = ctx.get(result_key)
            if r and "binary_defects" in r:
                vis = overlay_mask(vis, r["binary_defects"], color, 0.3)
                vis = draw_contours(vis, r["defect_contours"], color, 2)

        # Draw result text
        color = (0, 0, 255) if overall == "NG" else (0, 255, 0)
        cv2.putText(vis, overall, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if reasons:
            for i, r in enumerate(reasons[:3]):
                cv2.putText(vis, r, (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return self._result(vis, {"overall": overall, "ng_reasons": reasons,
                                   "match_method": ctx.get("match_method", ""),
                                   "match_score": round(ctx.get("match_score", 0), 4)})
