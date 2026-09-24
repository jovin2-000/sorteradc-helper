"""Operators for template matching and image alignment."""
import sys, os
import numpy as np
import cv2
from framework import Operator, Context, StepResult, _encode, to_gray, overlay_mask, hconcat_images

POC_ROOT = os.environ.get("POC_ROOT", "/app")
if POC_ROOT not in sys.path:
    sys.path.insert(0, POC_ROOT)


class ROIExtract(Operator):
    """Extract center ROI from input image."""

    def __init__(self):
        super().__init__("roi_extract", "ROI裁剪", "定位摆正",
                         param_keys=["search_roiw", "search_roih"],
                         output_keys=["search_roi", "roi_rect"])

    def execute(self, ctx: Context) -> StepResult:
        img = ctx.get("image")
        h, w = img.shape[:2]
        roiw = min(ctx.p("search_roiw", 300), w)
        roih = min(ctx.p("search_roih", 400), h)
        xc, yc = w // 2, h // 2
        y0, y1 = max(0, yc - roih // 2), min(h, yc + roih // 2)
        x0, x1 = max(0, xc - roiw // 2), min(w, xc + roiw // 2)
        roi = img[y0:y1, x0:x1]
        ctx.set("search_roi", roi)
        ctx.set("roi_rect", (x0, y0, x1, y1))

        vis = img.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"ROI {x1-x0}x{y1-y0}", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return self._result(vis, {"roi_w": x1 - x0, "roi_h": y1 - y0})


class SIFTDetect(Operator):
    """SIFT feature detection on template and ROI."""

    def __init__(self):
        super().__init__("sift_detect", "SIFT特征检测", "定位摆正",
                         output_keys=["sift_kp1", "sift_kp2", "sift_des1", "sift_des2"])

    def execute(self, ctx: Context) -> StepResult:
        tmpl_gray = to_gray(ctx.get("template"))
        scene_gray = to_gray(ctx.get("search_roi"))
        sift = cv2.SIFT_create(nfeatures=4000)
        kp1, des1 = sift.detectAndCompute(tmpl_gray, None)
        kp2, des2 = sift.detectAndCompute(scene_gray, None)
        ctx.set("sift_kp1", kp1)
        ctx.set("sift_kp2", kp2)
        ctx.set("sift_des1", des1)
        ctx.set("sift_des2", des2)

        n1 = len(kp1) if kp1 else 0
        n2 = len(kp2) if kp2 else 0
        vis = cv2.drawKeypoints(tmpl_gray, kp1 or [], None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self._result(vis, {"tmpl_kp": n1, "scene_kp": n2},
                            status="ok" if n1 >= 4 and n2 >= 4 else "fail")


class SIFTMatch(Operator):
    """SIFT feature matching with Lowe ratio test."""

    def __init__(self):
        super().__init__("sift_match", "SIFT特征匹配", "定位摆正",
                         input_keys=["sift_kp1", "sift_kp2", "sift_des1", "sift_des2"],
                         output_keys=["sift_good_matches"])

    def execute(self, ctx: Context) -> StepResult:
        des1, des2 = ctx.get("sift_des1"), ctx.get("sift_des2")
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            ctx.set("sift_good_matches", [])
            return self._result(None, {"good_matches": 0}, status="fail")

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        ctx.set("sift_good_matches", good)

        vis = cv2.drawMatches(to_gray(ctx.get("template")), ctx.get("sift_kp1"),
                              to_gray(ctx.get("search_roi")), ctx.get("sift_kp2"),
                              good[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        min_good = 10
        return self._result(vis, {"good_matches": len(good), "min_required": min_good},
                            status="ok" if len(good) >= min_good else "fail")


class RANSACPose(Operator):
    """RANSAC transform estimation from SIFT matches."""

    def __init__(self):
        super().__init__("ransac_pose", "RANSAC变换估计", "定位摆正",
                         input_keys=["sift_good_matches", "sift_kp1", "sift_kp2"],
                         output_keys=["ransac_M", "ransac_inliers"])

    def execute(self, ctx: Context) -> StepResult:
        good = ctx.get("sift_good_matches")
        if not good or len(good) < 10:
            return self._result(None, {"inliers": 0}, status="skip")

        kp1, kp2 = ctx.get("sift_kp1"), ctx.get("sift_kp2")
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                   ransacReprojThreshold=4.0)
        ctx.set("ransac_M", M)
        if M is not None:
            inl = int(inliers.sum())
            ctx.set("ransac_inliers", inl)
            score = inl / max(1, len(good))
            a, b = M[0, 0], M[1, 0]
            scale = float(np.hypot(a, b))
            angle = float(np.degrees(np.arctan2(b, a)))
            return self._result(None, {"inliers": inl, "total": len(good),
                                        "score": round(score, 4),
                                        "angle": round(angle, 2),
                                        "scale": round(scale, 4)},
                                status="ok" if inl >= 6 else "fail")
        return self._result(None, {"inliers": 0}, status="fail")


class ShapeMatchFallback(Operator):
    """Shape match fallback when SIFT fails."""

    def __init__(self):
        super().__init__("shape_match_fallback", "ShapeMatch兜底", "定位摆正",
                         param_keys=["shape_match_min_confidence"],
                         input_keys=["search_roi", "template"],
                         output_keys=["shape_match_box", "shape_match_score", "match_method"])

    def execute(self, ctx: Context) -> StepResult:
        from algorithms.ShapeMatch import shape_match

        class M: pass
        m = M()
        m.shape_match_template = ctx.get("template")
        m.mask_border_only = ctx.get("mask_border")
        m.mask_ball_only = ctx.get("mask_ball")
        m.mask_all_foreground = ctx.get("mask_foreground")
        m.shape_match_tpl_gx = ctx.get("tpl_gx")
        m.shape_match_tpl_gy = ctx.get("tpl_gy")
        m.shape_match_tpl_mag = ctx.get("tpl_mag")
        m.shape_match_tpl_contour_points = ctx.get("tpl_contour")
        m.shape_match_tpl_xym = ctx.get("tpl_xym")
        m.shape_match_pyramid_levels = 3
        m.shape_match_min_score = 0.7
        m.canonical_roi_size = ctx.get("template").shape[:2]

        min_conf = ctx.p("shape_match_min_confidence", 0.85)
        output = shape_match(ctx.get("search_roi"), m, min_conf,
                             step_size=2, enable_pyramid=True,
                             pyramid_scales=[4, 2, 1], refinement_radius=10)
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        if not boxes:
            return self._result(ctx.get("search_roi"), {"hit": False}, status="fail")

        best = int(np.argmax(scores)) if isinstance(scores, list) else 0
        die = boxes[best]
        score = max(scores) if isinstance(scores, list) else scores
        ctx.set("shape_match_box", die)
        ctx.set("shape_match_score", score)
        ctx.set("match_method", "shape_match")
        ctx.set("match_score", float(score))

        vis = ctx.get("search_roi").copy()
        x, y, bw, bh = die
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(vis, f"score={score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return self._result(vis, {"score": round(float(score), 4), "boxes": len(boxes)})


class WarpAlign(Operator):
    """Warp/align the matched die back to template pose."""

    def __init__(self):
        super().__init__("warp_align", "摆正抠图", "定位摆正",
                         input_keys=["ransac_M", "shape_match_box", "template", "image"],
                         output_keys=["aligned_image", "match_method"])

    def execute(self, ctx: Context) -> StepResult:
        aligned = None
        method = ""

        # Try SIFT path first
        M = ctx.get("ransac_M")
        if M is not None and ctx.get("ransac_inliers", 0) >= 6:
            th_t, tw_t = to_gray(ctx.get("template")).shape[:2]
            x0 = ctx.get("roi_rect", (0, 0))[0]
            y0 = ctx.get("roi_rect", (0, 0, 0, 0))[1]
            M_adj = M.copy()
            M_adj[0, 2] -= x0
            M_adj[1, 2] -= y0
            M_inv = cv2.invertAffineTransform(M_adj)
            aligned = cv2.warpAffine(ctx.get("image"), M_inv, (tw_t, th_t),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
            method = "SIFT+RANSAC"
        elif ctx.has("shape_match_box"):
            from algorithms.ImageRegistration import ImageRegistration
            die = ctx.get("shape_match_box")
            registrar = ImageRegistration(
                black_threshold=ctx.p("extract_threshold", 0), min_contour_area=1000.0)
            reg = registrar.register_by_perspective_transform(
                template=ctx.get("template"), target_image=ctx.get("search_roi"),
                match_row=die[1], match_col=die[0],
                reference_point="topleft", roi_size=(die[2], die[3]),
                enable_refinement=True)
            aligned = reg.aligned_image
            method = "shape_match+registration"
            ctx.set("match_angle", float(reg.rotation_angle))
            ctx.set("match_scale", float(reg.scale_factor))
            ctx.set("match_score", float(ctx.get("shape_match_score", 0)))

        ctx.set("match_method", method)
        if aligned is None:
            return self._result(None, {"error": "no match"}, status="fail")

        ctx.set("aligned_image", aligned)
        return self._result(aligned, {"size": f"{aligned.shape[1]}x{aligned.shape[0]}",
                                      "method": method,
                                      "score": round(ctx.get("match_score", 0), 4)})


class MaskApply(Operator):
    """Visualize mask overlays on the aligned image."""

    def __init__(self):
        super().__init__("mask_apply", "掩码叠加", "定位摆正",
                         input_keys=["aligned_image", "mask_border", "mask_ball"])

    def execute(self, ctx: Context) -> StepResult:
        aligned = ctx.get("aligned_image")
        vis = aligned.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        masks = {"border": (ctx.get("mask_border"), (0, 255, 0)),
                 "ball": (ctx.get("mask_ball"), (255, 100, 0))}
        metrics = {}
        for name, (mask, color) in masks.items():
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                mb = mask > 0
                ov = vis.copy()
                ov[mb] = color
                vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)
                metrics[f"{name}_px"] = int(np.sum(mb))
        return self._result(vis, metrics)

