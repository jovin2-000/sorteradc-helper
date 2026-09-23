"""Step-by-step detection pipeline with intermediate result capture.

Each step produces a visualization image + metrics dict, mimicking
Halcon's operator-by-operator debugging style.
"""
import sys
import os
import time
import base64
import traceback
import numpy as np
import cv2

POC_ROOT = os.environ.get("POC_ROOT", "/app")
if POC_ROOT not in sys.path:
    sys.path.insert(0, POC_ROOT)

from algorithms.template_match import match_template as tm_match
from algorithms.ShapeMatch import shape_match, shape_match2
from algorithms.ImageRegistration import ImageRegistration
from algorithms.MaskBasedDefectDetector import MaskBasedDefectDetector


def _encode(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def _overlay_defects(aligned, binary_mask, color=(0, 0, 255), alpha=0.4):
    vis = aligned.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    mask_bool = binary_mask > 0
    if mask_bool.any():
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def _draw_contours(img, contours, color=(0, 0, 255), thickness=2):
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def _make_step(name, title, img, metrics=None, status="ok"):
    return {
        "name": name,
        "title": title,
        "image": _encode(img),
        "metrics": metrics or {},
        "status": status,
    }


class _ParamsObj:
    """Simple attribute-access wrapper for a params dict."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def _extract_roi(image, roiw, roih):
    """Extract center ROI from image."""
    h, w = image.shape[:2]
    xc, yc = w // 2, h // 2
    roiw = min(roiw, w)
    roih = min(roih, h)
    y0 = max(0, yc - roih // 2)
    y1 = min(h, yc + roih // 2)
    x0 = max(0, xc - roiw // 2)
    x1 = min(w, xc + roiw // 2)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def _draw_roi_rect(image, roi_rect):
    x0, y0, x1, y1 = roi_rect
    vis = image.copy()
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(vis, f"ROI {x1-x0}x{y1-y0}", (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


# ---- Shape match helpers ----
def _make_model_proxy(td):
    """Create a lightweight proxy that mimics ProductModel for shape_match."""
    class M: pass
    m = M()
    m.shape_match_template = td.template_image
    m.mask_border_only = td.mask_border
    m.mask_ball_only = td.mask_ball
    m.mask_all_foreground = td.mask_foreground
    m.mask_trace_only = td.mask_trace
    m.mask_expand_only = td.mask_expand
    m.shape_match_tpl_gx = td.tpl_gx
    m.shape_match_tpl_gy = td.tpl_gy
    m.shape_match_tpl_mag = td.tpl_mag
    m.shape_match_tpl_contour_points = td.tpl_contour
    m.shape_match_tpl_xym = td.tpl_xym
    m.shape_match_pyramid_levels = 3
    m.shape_match_min_score = 0.7
    m.canonical_roi_size = td.template_image.shape[:2]
    return m


def _make_side_proxy(st):
    class M: pass
    m = M()
    m.shape_match_template = st.tpl_image
    m.mask_border_only = st.mask_border
    m.mask_ball_only = st.mask_ball
    m.mask_all_foreground = st.mask_foreground
    m.mask_trace_only = st.mask_trace
    m.mask_expand_only = st.mask_expand
    m.shape_match_tpl_gx = st.tpl_gx
    m.shape_match_tpl_gy = st.tpl_gy
    m.shape_match_tpl_mag = st.tpl_mag
    m.shape_match_tpl_contour_points = st.tpl_contour
    m.shape_match_tpl_xym = st.tpl_xym
    m.shape_match_pyramid_levels = 3
    m.shape_match_min_score = 0.7
    m.canonical_roi_size = st.tpl_image.shape[:2]
    return m


# ---- Main detection pipeline ----
def run_main_detection(template_data, image, params_dict):
    steps = []
    t_total = time.time()
    align_template = template_data.template_image
    mask_border = template_data.mask_border
    mask_ball = template_data.mask_ball
    mask_trace = template_data.mask_trace
    mask_expand = template_data.mask_expand
    p = _ParamsObj(params_dict)

    # Step 1: Input
    h, w = image.shape[:2]
    steps.append(_make_step("input", "1. 输入图像", image, {"width": w, "height": h}))

    # Step 2: ROI
    roiw = getattr(p, "search_roiw", min(w, 800))
    roih = getattr(p, "search_roih", min(h, 800))
    search_roi, roi_rect = _extract_roi(image, roiw, roih)
    steps.append(_make_step("roi", "2. ROI提取", _draw_roi_rect(image, roi_rect),
                            {"roi_w": roi_rect[2]-roi_rect[0], "roi_h": roi_rect[3]-roi_rect[1]}))

    # Step 3: Template matching
    aligned_image = None
    use_tm = getattr(p, "use_template_match", False)

    if use_tm:
        tm_kwargs = dict(
            angle_range=tuple(getattr(p, "tm_angle_range", (0.0, 360.0))),
            scale_range=tuple(getattr(p, "tm_scale_range", (0.9, 1.1))),
            thresh=getattr(p, "tm_thresh", 0.6),
            method=getattr(p, "tm_method", "auto"),
            blur_thresh=getattr(p, "tm_blur_thresh", 0.0),
            center_margin=getattr(p, "tm_center_margin", 0.1),
            require_die_in_roi=getattr(p, "tm_require_die_in_roi", True),
            roi_w=roiw, roi_h=roih, verbose=False,
        )
        t0 = time.time()
        try:
            tm_result = tm_match(template=align_template, scene=image, **tm_kwargs)
        except Exception as e:
            tm_result = {"hit": False, "reason": str(e)}
        t_match = time.time() - t0

        if tm_result.get("hit"):
            score = tm_result.get("score", 0)
            angle = tm_result.get("angle", 0)
            scale = tm_result.get("scale", 1.0)
            center = tm_result.get("center", (0, 0))
            corners = tm_result.get("corners")
            aligned_image = tm_result.get("aligned")
            mm = {"method": "template_match(SIFT+NCC)", "score": round(float(score), 4),
                  "angle": round(float(angle), 2), "scale": round(float(scale), 4),
                  "center": [round(float(center[0]), 1), round(float(center[1]), 1)],
                  "time_ms": round(t_match * 1000, 1)}
            vis = image.copy()
            if corners is not None:
                pts = np.int32(corners).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
            cv2.drawMarker(vis, (int(round(center[0])), int(round(center[1]))),
                           (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
            vis = _draw_roi_rect(vis, roi_rect)
            cv2.putText(vis, f"score={score:.3f} angle={angle:.1f} scale={scale:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            steps.append(_make_step("match", "3. 模板匹配", vis, mm))
        else:
            # Fallback to shape_match
            reason = tm_result.get("reason", "")
            fb_msg = f"template_match失败({reason}), 回退到shape_match"
            aligned_image = _run_shape_match(steps, template_data, params_dict,
                                             search_roi, roi_rect, t_match, fb_msg)
    else:
        aligned_image = _run_shape_match(steps, template_data, params_dict,
                                         search_roi, roi_rect, 0, "")

    if aligned_image is None:
        return {"steps": steps, "result": {"overall": "NG", "reason": "匹配失败"}}

    # Step 4: Aligned
    steps.append(_make_step("aligned", "4. 摆正抠图(aligned)", aligned_image,
                            {"size": f"{aligned_image.shape[1]}x{aligned_image.shape[0]}"}))

    # Step 5: Mask overlay
    mask_vis = aligned_image.copy()
    if len(mask_vis.shape) == 2:
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    if mask_border is not None and mask_border.shape[:2] == mask_vis.shape[:2]:
        mask_vis[mask_border > 0] = [0, 255, 0]
    if mask_ball is not None and mask_ball.shape[:2] == mask_vis.shape[:2]:
        mb = mask_ball > 0
        ov = mask_vis.copy()
        ov[mb] = [255, 100, 0]
        mask_vis = cv2.addWeighted(ov, 0.3, mask_vis, 0.7, 0)
    steps.append(_make_step("masks", "5. 掩码叠加", mask_vis,
                            {"border_px": int(np.sum(mask_border > 0)) if mask_border is not None else 0,
                             "ball_px": int(np.sum(mask_ball > 0)) if mask_ball is not None else 0}))

    # Defect detection
    detector = MaskBasedDefectDetector()
    result_summary = {"overall": "OK", "ng_reasons": []}
    result_summary.update(_run_defect_detection(
        steps, detector, align_template, aligned_image,
        mask_border, mask_ball, mask_trace, mask_expand, p, 6))
    result_summary["time_total_ms"] = round((time.time() - t_total) * 1000, 1)
    return {"steps": steps, "result": result_summary}


def _run_shape_match(steps, td, params_dict, search_roi, roi_rect, t_prev, fallback_msg):
    """Run shape_match + registration, add step to steps list, return aligned_image."""
    p = _ParamsObj(params_dict)
    min_conf = getattr(p, "shape_match_min_confidence", 0.85)
    model_proxy = _make_model_proxy(td)

    t0 = time.time()
    try:
        output = shape_match(search_roi, model_proxy, min_conf,
                             step_size=2, enable_pyramid=True,
                             pyramid_scales=[4, 2, 1], refinement_radius=10)
    except Exception as e:
        output = {"boxes": [], "scores": [], "error": str(e)}
    t_match = time.time() - t0

    det_boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    if not det_boxes:
        steps.append(_make_step("match", "3. 模板匹配", _draw_roi_rect(search_roi.copy() if search_roi is not None else np.zeros((100,100,3), np.uint8), (0,0,0,0)),
                                {"hit": False, "time_ms": round(t_match * 1000, 1),
                                 **({"fallback": fallback_msg} if fallback_msg else {})}, status="fail"))
        return None

    best_idx = int(np.argmax(scores)) if isinstance(scores, list) else 0
    die_direct = det_boxes[best_idx]
    max_score = max(scores) if isinstance(scores, list) else scores

    registrar = ImageRegistration(black_threshold=getattr(p, "extract_threshold", 0), min_contour_area=1000.0)
    t0 = time.time()
    reg_result = registrar.register_by_perspective_transform(
        template=td.template_image, target_image=search_roi,
        match_row=die_direct[1], match_col=die_direct[0],
        reference_point="topleft", roi_size=(die_direct[2], die_direct[3]),
        enable_refinement=True)
    t_reg = time.time() - t0
    aligned = reg_result.aligned_image

    mm = {"method": "shape_match+registration", "score": round(float(max_score), 4),
          "rotation": round(float(reg_result.rotation_angle), 2),
          "scale": round(float(reg_result.scale_factor), 4),
          "reg_score": round(float(reg_result.registration_score), 4),
          "boxes": len(det_boxes), "time_ms": round((t_match + t_reg) * 1000, 1)}
    if fallback_msg:
        mm["fallback"] = fallback_msg

    vis = search_roi.copy()
    x, y, bw, bh = die_direct
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv2.putText(vis, f"score={max_score:.3f} rot={reg_result.rotation_angle:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    title = "3. 模板匹配(ShapeMatch)" if not fallback_msg else "3. 模板匹配(回退ShapeMatch)"
    steps.append(_make_step("match", title, vis, mm))
    return aligned


def _run_defect_detection(steps, detector, align_template, aligned_image,
                          mask_border, mask_ball, mask_trace, mask_expand, p, start_step):
    """Run border, ball, and surface defect detection. Returns summary dict."""
    result = {"overall": "OK", "ng_reasons": []}
    step_num = start_step
    mlabels = {"frequency": "频域法", "spatial": "空域法", "template": "模板法", "difference": "差分法"}

    # Border defect detection
    border_methods = getattr(p, "border_methods", ["frequency", "spatial", "template", "difference"])
    border_weights = getattr(p, "border_weights", {"frequency": 0.4, "spatial": 0.3, "template": 0.2, "difference": 0.1})
    border_min_area = getattr(p, "border_min_defect_area", 10.0)

    t0 = time.time()
    try:
        ensemble = detector.detect_defects_ensemble(
            template_image=align_template, target_image=aligned_image, border_mask=mask_border,
            methods=border_methods, weights=border_weights, min_defect_area=border_min_area,
            cutoff_frequency=getattr(p, "border_frequency_cutoff", 30.0),
            diff_threshold=getattr(p, "border_diff_diff_threshold", 30.0),
            freq_threshold=getattr(p, "border_frequency_diff_threshold", 30.0),
            clahe_clip_limit=getattr(p, "border_clahe_clip_limit", 2),
            gradient_threshold=getattr(p, "border_gradient_threshold", 50),
            binary_threshold=getattr(p, "border_binary_threshold", 0),
            vote_threshold=getattr(p, "border_vote_threshold", 0.5),
            thickness=getattr(p, "border_thickness", 1),
            post_kernelsz=getattr(p, "border_post_kernelsz", 3),
            fusion_kernelsz=getattr(p, "border_fusion_kernelsz", 3),
            fusion_skip_open=getattr(p, "border_fusion_skip_open", False))
    except Exception as e:
        ensemble = {}
        steps.append(_make_step("border_error", f"{step_num}. 边框检测(异常)", aligned_image,
                                {"error": str(e)}, status="error"))
        step_num += 1
    t_border = time.time() - t0

    for method in border_methods:
        if method in ensemble:
            res = ensemble[method]
            if res.diff_image is not None and res.diff_image.size > 0:
                dv = cv2.applyColorMap(res.diff_image, cv2.COLORMAP_JET)
                steps.append(_make_step(f"border_{method}", f"{step_num}. 边框-{mlabels.get(method, method)}", dv,
                                        {"defect_px": int(res.defect_pixels), "ratio": round(float(res.defect_ratio), 4)}))
                step_num += 1

    if "fused" in ensemble:
        fused = ensemble["fused"]
        fv = _overlay_defects(aligned_image, fused.binary_defects)
        fv = _draw_contours(fv, fused.defect_contours, (0, 0, 255), 2)
        is_ng = fused.defect_pixels >= border_min_area
        if is_ng:
            result["overall"] = "NG"
            result["ng_reasons"].append(f"边框缺陷: {fused.defect_pixels}px")
        steps.append(_make_step("border_fused", f"{step_num}. 边框-融合", fv,
                                {"defect_px": int(fused.defect_pixels), "ratio": round(float(fused.defect_ratio), 4),
                                 "is_ng": is_ng, "time_ms": round(t_border * 1000, 1)}))
        step_num += 1

    # Ball detection
    ball_expected = getattr(p, "ball_expected_count", 100)
    t0 = time.time()
    try:
        ball_result = detector.inspect_balls(
            target_image=aligned_image, ball_mask=mask_ball,
            expected_ball_count=ball_expected,
            white_to_black_ratio_threshold=getattr(p, "white_to_black_ratio_threshold", 0.3),
            white_pixel_threshold=getattr(p, "white_pixel_threshold", 150),
            ball_detection_method=getattr(p, "ball_inspect_method", "otsu"),
            min_ball_area=getattr(p, "ball_min_area", 600),
            min_black_area=getattr(p, "ball_min_black_area", 500),
            template_ball_positions=None,
            close_single_ball_mask=getattr(p, "close_single_ball_mask", False),
            use_dilate=getattr(p, "ball__mask_usedilate", True),
            ball_close_iters=getattr(p, "ball_close_iterations", 1))
        t_ball = time.time() - t0
        bv = aligned_image.copy()
        if len(bv.shape) == 2:
            bv = cv2.cvtColor(bv, cv2.COLOR_GRAY2BGR)
        for ball in ball_result.all_balls:
            cx, cy = int(ball.center[0]), int(ball.center[1])
            color = (0, 0, 255) if ball.is_defective else (0, 255, 0)
            cv2.circle(bv, (cx, cy), max(5, int(np.sqrt(ball.area / 3.14))), color, 2)
            if ball.is_defective:
                cv2.putText(bv, f"#{ball.ball_id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        is_ng = ball_result.overall_result == "NG"
        if is_ng:
            result["overall"] = "NG"
            if ball_result.missing_positions:
                result["ng_reasons"].append(f"缺球: 期望{ball_expected}实际{ball_result.actual_count}")
            for db in ball_result.defective_balls:
                result["ng_reasons"].append(f"球#{db.ball_id}: {db.defect_reason}")
        steps.append(_make_step("ball", f"{step_num}. 球检测", bv,
                                {"expected": ball_expected, "actual": ball_result.actual_count,
                                 "defective": len(ball_result.defective_balls), "is_ng": is_ng,
                                 "time_ms": round(t_ball * 1000, 1)}))
        step_num += 1
    except Exception as e:
        steps.append(_make_step("ball_error", f"{step_num}. 球检测(异常)", aligned_image,
                                {"error": str(e)}, status="error"))
        step_num += 1

    # Surface defect detection
    surface_methods = getattr(p, "surface_methods", ["frequency", "spatial", "template"])
    surface_weights = getattr(p, "surface_weights", {"frequency": 0.4, "spatial": 0.3, "template": 0.3})
    surface_min_area = getattr(p, "surface_min_defect_area", 10.0)
    t0 = time.time()
    try:
        surface = detector.detect_surface_defects_ensemble(
            template_image=align_template, target_image=aligned_image,
            border_mask=mask_border, ball_mask=mask_ball,
            trace_mask=mask_trace, expand_mask=mask_expand,
            methods=surface_methods, weights=surface_weights,
            min_defect_area=surface_min_area,
            cutoff_frequency=getattr(p, "surface_frequency_cutoff", 30.0),
            diff_threshold=getattr(p, "surface_diff_diff_threshold", 30.0),
            freq_threshold=getattr(p, "surface_frequency_diff_threshold", 30.0),
            clahe_clip_limit=getattr(p, "surface_clahe_clip_limit", 2),
            gradient_threshold=getattr(p, "surface_gradient_threshold", 50),
            binary_threshold=getattr(p, "surface_binary_threshold", 0),
            vote_threshold=getattr(p, "surface_vote_threshold", 0.5),
            thickness=getattr(p, "surface_thickness", 3),
            post_kernelsz=getattr(p, "surface_post_kernelsz", 3),
            fusion_kernelsz=getattr(p, "surface_fusion_kernelsz", 3),
            fusion_skip_open=getattr(p, "surface_fusion_skip_open", False))
    except Exception as e:
        surface = {}
        steps.append(_make_step("surface_error", f"{step_num}. 表面检测(异常)", aligned_image,
                                {"error": str(e)}, status="error"))
    t_surf = time.time() - t0

    for method in surface_methods:
        if method in surface:
            res = surface[method]
            if res.diff_image is not None and res.diff_image.size > 0:
                dv = cv2.applyColorMap(res.diff_image, cv2.COLORMAP_JET)
                steps.append(_make_step(f"surface_{method}", f"{step_num}. 表面-{mlabels.get(method, method)}", dv,
                                        {"defect_px": int(res.defect_pixels), "ratio": round(float(res.defect_ratio), 4)}))
                step_num += 1

    if "fused" in surface:
        fused = surface["fused"]
        fv = _overlay_defects(aligned_image, fused.binary_defects, (0, 255, 0))
        fv = _draw_contours(fv, fused.defect_contours, (0, 255, 0), 2)
        is_ng = fused.defect_pixels >= surface_min_area
        if is_ng:
            result["overall"] = "NG"
            result["ng_reasons"].append(f"表面缺陷: {fused.defect_pixels}px")
        steps.append(_make_step("surface_fused", f"{step_num}. 表面-融合", fv,
                                {"defect_px": int(fused.defect_pixels), "ratio": round(float(fused.defect_ratio), 4),
                                 "is_ng": is_ng, "time_ms": round(t_surf * 1000, 1)}))
        step_num += 1

    # Final visualization
    try:
        vis_results = {}
        if "fused" in ensemble:
            vis_results["border_defects"] = ensemble["fused"]
        if "fused" in surface:
            vis_results["surface_defects"] = surface["fused"]
        try:
            vis_results["ball_inspection"] = ball_result
        except NameError:
            pass
        ng_cats = []
        for r in result["ng_reasons"]:
            if "边框" in r: ng_cats.append("崩边")
            if "球" in r: ng_cats.append("球NG")
            if "表面" in r: ng_cats.append("表面缺陷")
        final_vis = detector.visualize_all_results(
            target_image=aligned_image, results=vis_results,
            overall_result=result["overall"], ng_categories=ng_cats or None, save_path=None)
    except Exception:
        final_vis = aligned_image.copy()
        cv2.putText(final_vis, result["overall"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if result["overall"] == "NG" else (0, 255, 0), 2)
    steps.append(_make_step("final", f"{step_num}. 最终结果", final_vis, result))
    return result


# ---- Side detection pipeline ----
def run_side_detection(template_data, side_name, image, params_dict):
    steps = []
    t_total = time.time()

    if side_name not in (template_data.side_templates or {}):
        steps.append(_make_step("error", "错误", image, {"error": f"侧面 {side_name} 模板不存在"}, status="error"))
        return {"steps": steps, "result": {"overall": "NG", "reason": f"no side template: {side_name}"}}

    st = template_data.side_templates[side_name]
    align_template = st.tpl_image
    mask_border = st.mask_border
    mask_ball = st.mask_ball
    mask_trace = st.mask_trace
    mask_expand = st.mask_expand
    p = _ParamsObj(params_dict)

    h, w = image.shape[:2]
    steps.append(_make_step("input", "1. 输入图像", image, {"width": w, "height": h}))

    roiw = getattr(p, "search_roiw", min(w, 800))
    roih = getattr(p, "search_roih", min(h, 800))
    search_roi, roi_rect = _extract_roi(image, roiw, roih)
    steps.append(_make_step("roi", "2. ROI提取", _draw_roi_rect(image, roi_rect),
                            {"roi_w": roi_rect[2]-roi_rect[0], "roi_h": roi_rect[3]-roi_rect[1]}))

    # Shape match
    min_conf = getattr(p, "shape_match_min_confidence", 0.85)
    t0 = time.time()
    try:
        output = shape_match(search_roi, _make_side_proxy(st), min_conf,
                             step_size=2, enable_pyramid=True, pyramid_scales=[4, 2, 1], refinement_radius=10)
    except Exception as e:
        output = {"boxes": [], "scores": []}
    t_match = time.time() - t0

    det_boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    if not det_boxes:
        steps.append(_make_step("match", "3. 模板匹配", search_roi,
                                {"hit": False, "time_ms": round(t_match * 1000, 1)}, status="fail"))
        return {"steps": steps, "result": {"overall": "NG", "reason": "shape_match未找到目标"}}

    best_idx = int(np.argmax(scores)) if isinstance(scores, list) else 0
    die_direct = det_boxes[best_idx]
    max_score = max(scores) if isinstance(scores, list) else scores

    registrar = ImageRegistration(black_threshold=getattr(p, "extract_threshold", 0), min_contour_area=1000.0)
    t0 = time.time()
    reg_result = registrar.register_by_perspective_transform(
        template=align_template, target_image=search_roi,
        match_row=die_direct[1], match_col=die_direct[0],
        reference_point="topleft", roi_size=(die_direct[2], die_direct[3]),
        enable_refinement=True)
    t_reg = time.time() - t0
    aligned_image = reg_result.aligned_image

    vis = search_roi.copy()
    x, y, bw, bh = die_direct
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv2.putText(vis, f"score={max_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(_make_step("match", "3. 模板匹配", vis,
                            {"score": round(float(max_score), 4), "rotation": round(float(reg_result.rotation_angle), 2),
                             "reg_score": round(float(reg_result.registration_score), 4)}))

    steps.append(_make_step("aligned", "4. 摆正抠图", aligned_image,
                            {"size": f"{aligned_image.shape[1]}x{aligned_image.shape[0]}"}))

    detector = MaskBasedDefectDetector()
    result_summary = {"overall": "OK", "ng_reasons": []}
    result_summary.update(_run_defect_detection(
        steps, detector, align_template, aligned_image,
        mask_border, mask_ball, mask_trace, mask_expand, p, 5))
    result_summary["time_total_ms"] = round((time.time() - t_total) * 1000, 1)
    return {"steps": steps, "result": result_summary}
