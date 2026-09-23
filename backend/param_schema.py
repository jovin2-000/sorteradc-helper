"""Parameter schema for detection pipeline UI."""

CATEGORIES = [
    {"id": "template_match", "title": "模板匹配"},
    {"id": "shape_match", "title": "ShapeMatch(旧路径)"},
    {"id": "border_defect", "title": "边框缺陷检测"},
    {"id": "ball_detection", "title": "球检测"},
    {"id": "surface_defect", "title": "表面缺陷检测"},
    {"id": "side_detection", "title": "侧面检测"},
]

METHOD_LABELS = {
    "frequency": "频域法",
    "spatial": "空域法",
    "template": "模板法",
    "difference": "差分法",
}

PARAMS = [
    {"name": "use_template_match", "type": "bool", "default": False, "label": "使用新路径模板匹配", "category": "template_match"},
    {"name": "tm_thresh", "type": "float", "min": 0.1, "max": 0.99, "step": 0.01, "default": 0.6, "label": "NCC匹配得分下限", "category": "template_match"},
    {"name": "tm_blur_thresh", "type": "float", "min": 0.0, "max": 0.2, "step": 0.005, "default": 0.0, "label": "中心带清晰度门限(0=关)", "category": "template_match"},
    {"name": "tm_center_margin", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.1, "label": "中心命中余量(占短边比例)", "category": "template_match"},
    {"name": "tm_require_die_in_roi", "type": "bool", "default": True, "label": "die四角须全在ROI内", "category": "template_match"},
    {"name": "tm_method", "type": "select", "options": ["auto", "feature", "edge"], "default": "auto", "label": "姿态估计方法", "category": "template_match"},
    {"name": "tm_angle_range", "type": "tuple_float", "default": [0.0, 360.0], "label": "旋转搜索范围(度)", "category": "template_match"},
    {"name": "tm_scale_range", "type": "tuple_float", "default": [0.9, 1.1], "label": "缩放搜索范围", "category": "template_match"},
    {"name": "search_roiw", "type": "int", "min": 50, "max": 2000, "step": 10, "default": 300, "label": "中心ROI宽度(px)", "category": "template_match"},
    {"name": "search_roih", "type": "int", "min": 50, "max": 2000, "step": 10, "default": 300, "label": "中心ROI高度(px)", "category": "template_match"},
    {"name": "shape_match_min_confidence", "type": "float", "min": 0.3, "max": 0.99, "step": 0.01, "default": 0.85, "label": "ShapeMatch最低置信度", "category": "shape_match"},
    {"name": "shape_match_fail_stepsz", "type": "int", "min": 1, "max": 5, "step": 1, "default": 2, "label": "匹配失败重试步长", "category": "shape_match"},
    {"name": "use_shapematch2", "type": "bool", "default": False, "label": "启用shape_match2兜底", "category": "shape_match"},
    {"name": "shape2_threshold", "type": "int", "min": 0, "max": 255, "step": 1, "default": 100, "label": "shape2阈值", "category": "shape_match"},
    {"name": "shape2_use_otsu", "type": "bool", "default": False, "label": "shape2使用OTSU", "category": "shape_match"},
    {"name": "fine_align_max_error", "type": "float", "min": 1.0, "max": 20.0, "step": 0.5, "default": 5.0, "label": "配准最大允许误差(px)", "category": "shape_match"},
    {"name": "extract_threshold", "type": "int", "min": 0, "max": 255, "step": 1, "default": 0, "label": "二值化提取阈值(0=OTSU)", "category": "shape_match"},
    {"name": "border_min_defect_area", "type": "float", "min": 0.5, "max": 50, "step": 0.5, "default": 10.0, "label": "最小缺陷面积(px)", "category": "border_defect"},
    {"name": "border_vote_threshold", "type": "float", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.5, "label": "投票阈值(多方法融合)", "category": "border_defect"},
    {"name": "border_frequency_diff_threshold", "type": "float", "min": 10, "max": 300, "step": 5, "default": 30.0, "label": "频域法阈值", "category": "border_defect"},
    {"name": "border_gradient_threshold", "type": "int", "min": 10, "max": 300, "step": 5, "default": 50, "label": "空域法梯度阈值", "category": "border_defect"},
    {"name": "border_diff_diff_threshold", "type": "float", "min": 10, "max": 300, "step": 5, "default": 30.0, "label": "差分法阈值", "category": "border_defect"},
    {"name": "border_binary_threshold", "type": "int", "min": 0, "max": 255, "step": 5, "default": 0, "label": "二值化阈值(0=OTSU)", "category": "border_defect"},
    {"name": "border_clahe_clip_limit", "type": "int", "min": 1, "max": 10, "step": 1, "default": 2, "label": "CLAHE对比度增强限制", "category": "border_defect"},
    {"name": "border_thickness", "type": "int", "min": 1, "max": 10, "step": 1, "default": 1, "label": "template法轮廓厚度(px)", "category": "border_defect"},
    {"name": "border_post_kernelsz", "type": "int", "min": 1, "max": 9, "step": 2, "default": 3, "label": "后处理形态学核大小(px)", "category": "border_defect"},
    {"name": "border_fusion_kernelsz", "type": "int", "min": 1, "max": 9, "step": 2, "default": 3, "label": "融合后形态学核大小", "category": "border_defect"},
    {"name": "border_fusion_skip_open", "type": "bool", "default": False, "label": "融合后跳过MORPH_OPEN", "category": "border_defect"},
    {"name": "border_corner_ratio", "type": "float", "min": 0.05, "max": 0.5, "step": 0.01, "default": 0.18, "label": "崩角/崩边分类比例", "category": "border_defect"},
    {"name": "border_frequency_cutoff", "type": "float", "min": 5, "max": 100, "step": 5, "default": 30.0, "label": "频域截止频率", "category": "border_defect"},
    {"name": "border_methods", "type": "multiselect", "options": ["frequency", "spatial", "template", "difference"], "default": ["frequency", "spatial", "template", "difference"], "label": "启用检测方法", "category": "border_defect"},
    {"name": "border_weights", "type": "weights", "options": ["frequency", "spatial", "template", "difference"], "default": {"frequency": 0.4, "spatial": 0.3, "template": 0.2, "difference": 0.1}, "label": "各方法投票权重", "category": "border_defect"},
    {"name": "ball_expected_count", "type": "int", "min": 0, "max": 500, "step": 1, "default": 100, "label": "期望球数量", "category": "ball_detection"},
    {"name": "ball_count_tolerance", "type": "int", "min": 0, "max": 50, "step": 1, "default": 10, "label": "球数量容差", "category": "ball_detection"},
    {"name": "ball_min_area", "type": "float", "min": 1, "max": 5000, "step": 10, "default": 600, "label": "球最小面积(px)", "category": "ball_detection"},
    {"name": "ball_max_area", "type": "float", "min": 100, "max": 10000, "step": 100, "default": 2500, "label": "球最大面积(px)", "category": "ball_detection"},
    {"name": "ball_min_black_area", "type": "int", "min": 0, "max": 5000, "step": 50, "default": 500, "label": "判缺陷:黑球最小面积", "category": "ball_detection"},
    {"name": "ball_hollow_ratio_max", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "default": 0.3, "label": "球空心比例最大值", "category": "ball_detection"},
    {"name": "ball_circularity_min", "type": "float", "min": 0.3, "max": 1.0, "step": 0.05, "default": 0.7, "label": "球圆度最小值", "category": "ball_detection"},
    {"name": "white_to_black_ratio_threshold", "type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "default": 0.3, "label": "白黑像素比阈值", "category": "ball_detection"},
    {"name": "white_pixel_threshold", "type": "int", "min": 0, "max": 255, "step": 5, "default": 150, "label": "白点亮度阈值", "category": "ball_detection"},
    {"name": "ball_inspect_method", "type": "select", "options": ["otsu", "adaptive", "blob", "fixed"], "default": "otsu", "label": "球检测方法", "category": "ball_detection"},
    {"name": "ball__mask_usedilate", "type": "bool", "default": True, "label": "对球掩码做膨胀", "category": "ball_detection"},
    {"name": "close_single_ball_mask", "type": "bool", "default": False, "label": "对单球掩码做闭运算", "category": "ball_detection"},
    {"name": "ball_close_iterations", "type": "int", "min": 0, "max": 10, "step": 1, "default": 1, "label": "球掩码闭运算迭代次数", "category": "ball_detection"},
    {"name": "surface_min_defect_area", "type": "float", "min": 0.5, "max": 50, "step": 0.5, "default": 10.0, "label": "最小缺陷面积(px)", "category": "surface_defect"},
    {"name": "surface_vote_threshold", "type": "float", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.5, "label": "投票阈值", "category": "surface_defect"},
    {"name": "surface_frequency_diff_threshold", "type": "float", "min": 10, "max": 300, "step": 5, "default": 30.0, "label": "频域法阈值", "category": "surface_defect"},
    {"name": "surface_gradient_threshold", "type": "int", "min": 10, "max": 300, "step": 5, "default": 50, "label": "空域法梯度阈值", "category": "surface_defect"},
    {"name": "surface_diff_diff_threshold", "type": "float", "min": 10, "max": 300, "step": 5, "default": 30.0, "label": "差分法阈值", "category": "surface_defect"},
    {"name": "surface_binary_threshold", "type": "int", "min": 0, "max": 255, "step": 5, "default": 0, "label": "二值化阈值(0=OTSU)", "category": "surface_defect"},
    {"name": "surface_clahe_clip_limit", "type": "int", "min": 1, "max": 10, "step": 1, "default": 2, "label": "CLAHE对比度限制", "category": "surface_defect"},
    {"name": "surface_thickness", "type": "int", "min": 1, "max": 10, "step": 1, "default": 3, "label": "template法轮廓厚度", "category": "surface_defect"},
    {"name": "surface_post_kernelsz", "type": "int", "min": 1, "max": 9, "step": 2, "default": 3, "label": "后处理核大小", "category": "surface_defect"},
    {"name": "surface_fusion_kernelsz", "type": "int", "min": 1, "max": 9, "step": 2, "default": 3, "label": "融合后形态学核大小", "category": "surface_defect"},
    {"name": "surface_fusion_skip_open", "type": "bool", "default": False, "label": "融合后跳过MORPH_OPEN", "category": "surface_defect"},
    {"name": "surface_frequency_cutoff", "type": "float", "min": 5, "max": 100, "step": 5, "default": 30.0, "label": "频域截止频率", "category": "surface_defect"},
    {"name": "surface_methods", "type": "multiselect", "options": ["frequency", "spatial", "template", "difference"], "default": ["frequency", "spatial", "template"], "label": "启用检测方法", "category": "surface_defect"},
    {"name": "surface_weights", "type": "weights", "options": ["frequency", "spatial", "template", "difference"], "default": {"frequency": 0.4, "spatial": 0.3, "template": 0.3}, "label": "各方法权重", "category": "surface_defect"},
    {"name": "background_min_defect_area", "type": "float", "min": 1, "max": 100, "step": 1, "default": 5, "label": "背景最小缺陷面积", "category": "surface_defect"},
    {"name": "background_max_defect_area", "type": "float", "min": 50, "max": 5000, "step": 50, "default": 500, "label": "背景最大缺陷面积", "category": "surface_defect"},
    {"name": "background_brightness_threshold", "type": "float", "min": 0, "max": 255, "step": 5, "default": 50, "label": "背景亮度阈值", "category": "surface_defect"},
    {"name": "side_ball_expected_count", "type": "int", "min": 0, "max": 200, "step": 1, "default": 50, "label": "侧面期望球数量", "category": "side_detection"},
    {"name": "side_ball_min_area", "type": "float", "min": 50, "max": 5000, "step": 50, "default": 300, "label": "侧面球最小面积", "category": "side_detection"},
    {"name": "side_ball_max_area", "type": "float", "min": 200, "max": 10000, "step": 100, "default": 2000, "label": "侧面球最大面积", "category": "side_detection"},
    {"name": "side_border_width", "type": "int", "min": 1, "max": 30, "step": 1, "default": 8, "label": "侧面边框宽度", "category": "side_detection"},
    {"name": "side_detection_method", "type": "select", "options": ["frequency", "spatial", "template", "ensemble"], "default": "frequency", "label": "侧面检测方法", "category": "side_detection"},
]


def get_schema():
    return {"categories": CATEGORIES, "params": PARAMS, "method_labels": METHOD_LABELS}


def get_defaults():
    defaults = {}
    for p in PARAMS:
        defaults[p["name"]] = p["default"]
    return defaults
