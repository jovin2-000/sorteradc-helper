"""
Flow definitions: compose operators from the ops/ library into named flows.

Each flow is an ordered list of operators. Adding a new flow (e.g. a custom
inspection pipeline) means defining a new function that returns a Flow object
-- no framework changes needed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework import Flow, Context
from ops.matching import (ROIExtract, SIFTDetect, SIFTMatch, RANSACPose,
                           ShapeMatchFallback, WarpAlign, MaskApply)
from ops.mask_create import (ROISelect, BoundaryOptimize, FeatureExtract,
                              ForegroundMask, BorderMask, BallMask, TraceMask, SaveTemplate)
from ops.defect import (RegionPreprocess, FreqFilter, SpatialCLAHE, SpatialGradient,
                         TplBinary, TplContourDiff, AbsDiff, ThresholdPostproc)
from ops.ball import BallMaskProc, BallBinarize, BallConnect, BallAnalyze
from ops.result import BorderFusion, BorderClassify, SurfaceFusion, ResultAggregate


# ---------------------------------------------------------------------------
# Flow 1: Template Creation (参考 getTemplate.py)
# ---------------------------------------------------------------------------

def build_template_creation_flow() -> Flow:
    """模板创建流程：从一张参考图创建检测模板（掩码+特征）。

    流程步骤：
    1. ROI选择 (交互式画框)
    2. 边界优化 (OTSU+轮廓)
    3. 特征提取 (梯度+轮廓)
    4. 前景掩码
    5. 边框掩码
    6. 球掩码
    7. 轨迹掩码 (可选)
    8. 保存模板
    """
    flow = Flow(
        id="template_create",
        name="模板创建",
        description="从参考图交互式创建检测模板：ROI选择、边界优化、掩码生成",
    )
    flow.add(ROISelect())
    flow.add(BoundaryOptimize())
    flow.add(FeatureExtract())
    flow.add(ForegroundMask())
    flow.add(BorderMask())
    flow.add(BallMask())
    flow.add(TraceMask())
    flow.add(SaveTemplate())
    return flow


# ---------------------------------------------------------------------------
# Flow 2: Front (Main) Detection Tuning
# ---------------------------------------------------------------------------

def _build_defect_method_chain(region: str, label: str, method: str,
                                use_surface: bool = False) -> list:
    """Build a chain of operators for one defect detection method.

    Args:
        region: context key prefix, e.g. "border_frequency", "surface_spatial"
        label: display label, e.g. "边框-频域法", "表面-空域法"
        method: "frequency" | "spatial" | "template" | "difference"
        use_surface: whether to use surface mask (vs border mask)

    Returns a list of operators.
    """
    ops = []
    full_region = f"{region}_{method}"

    # Step 1: Preprocess (gray + mask)
    ops.append(RegionPreprocess(full_region, label, use_surface_mask=use_surface))

    # Step 2-N: Method-specific intermediate steps
    if method == "frequency":
        ops.append(FreqFilter(full_region, label))
    elif method == "spatial":
        ops.append(SpatialCLAHE(full_region, label))
        ops.append(SpatialGradient(full_region, label))
    elif method == "template":
        ops.append(TplBinary(full_region, label))
        ops.append(TplContourDiff(full_region, label))
    elif method == "difference":
        ops.append(AbsDiff(full_region, label))

    # Final step: threshold + post-process
    thresh_param = f"{'surface' if use_surface else 'border'}_{method}_diff_threshold"
    if method == "spatial":
        thresh_param = f"{'surface' if use_surface else 'border'}_gradient_threshold"
    elif method == "difference":
        thresh_param = f"{'surface' if use_surface else 'border'}_diff_diff_threshold"
    elif method == "frequency":
        thresh_param = f"{'surface' if use_surface else 'border'}_frequency_diff_threshold"

    diff_key = "diff"
    if method == "spatial":
        diff_key = "grad_diff"
    elif method == "template":
        diff_key = "diff"

    ops.append(ThresholdPostproc(full_region, label, method, thresh_param, diff_key))
    return ops


def build_front_detection_flow() -> Flow:
    """正面检测调参流程：定位+边框+球+表面，每步独立可视化。

    流程步骤：
    A. 定位摆正 (ROI→SIFT→RANSAC→兜底→摆正→掩码叠加)
    B. 边框缺陷 (频域/空域/模板/差分 各自独立子步骤 → 融合 → 分类)
    C. 球检测 (掩码→二值化→连通域→逐球分析)
    D. 表面缺陷 (同边框结构，用表面掩码)
    E. 结果汇总
    """
    flow = Flow(
        id="front_detect",
        name="正面检测调参",
        description="正面检测全流程：定位摆正→边框缺陷→球检测→表面缺陷→结果汇总",
    )

    # A. 定位摆正
    flow.add(ROIExtract())
    flow.add(SIFTDetect())
    flow.add(SIFTMatch())
    flow.add(RANSACPose())
    flow.add(ShapeMatchFallback())
    flow.add(WarpAlign())
    flow.add(MaskApply())

    # B. 边框缺陷检测 - 四种方法各自独立子步骤链
    for method, label, thresh_param in [
        ("frequency", "边框-频域法", "border_frequency_diff_threshold"),
        ("spatial", "边框-空域法", "border_gradient_threshold"),
        ("template", "边框-模板法", "border_binary_threshold"),
        ("difference", "边框-差分法", "border_diff_diff_threshold"),
    ]:
        for op in _build_defect_method_chain("border", label, method, use_surface=False):
            flow.add(op)

    # 边框融合 + 分类
    flow.add(BorderFusion())
    flow.add(BorderClassify())

    # C. 球检测
    flow.add(BallMaskProc())
    flow.add(BallBinarize())
    flow.add(BallConnect())
    flow.add(BallAnalyze())

    # D. 表面缺陷检测
    for method, label, thresh_param in [
        ("frequency", "表面-频域法", "surface_frequency_diff_threshold"),
        ("spatial", "表面-空域法", "surface_gradient_threshold"),
        ("template", "表面-模板法", "surface_binary_threshold"),
        ("difference", "表面-差分法", "surface_diff_diff_threshold"),
    ]:
        for op in _build_defect_method_chain("surface", label, method, use_surface=True):
            flow.add(op)

    flow.add(SurfaceFusion())

    # E. 结果汇总
    flow.add(ResultAggregate())

    return flow


# ---------------------------------------------------------------------------
# Flow 3: Side Detection Tuning
# ---------------------------------------------------------------------------

def build_side_detection_flow() -> Flow:
    """侧面检测调参流程：使用侧面模板进行检测。

    与正面检测类似，但使用侧面模板和侧面参数。
    可以在后续根据具体侧面需求调整算子组成。
    """
    flow = Flow(
        id="side_detect",
        name="侧面检测调参",
        description="侧面检测流程：定位→边框→球(如有)→表面→结果",
    )

    # A. 定位摆正 (使用侧面模板)
    flow.add(ROIExtract())
    flow.add(ShapeMatchFallback())  # 侧面通常用 shape_match
    flow.add(WarpAlign())
    flow.add(MaskApply())

    # B. 边框缺陷
    for method, label in [("frequency", "边框-频域"), ("spatial", "边框-空域"),
                           ("template", "边框-模板"), ("difference", "边框-差分")]:
        for op in _build_defect_method_chain("border", label, method, use_surface=False):
            flow.add(op)
    flow.add(BorderFusion())
    flow.add(BorderClassify())

    # C. 球检测 (如果侧面有球)
    flow.add(BallMaskProc())
    flow.add(BallBinarize())
    flow.add(BallConnect())
    flow.add(BallAnalyze())

    # D. 表面缺陷
    for method, label in [("frequency", "表面-频域"), ("spatial", "表面-空域"),
                           ("template", "表面-模板")]:
        for op in _build_defect_method_chain("surface", label, method, use_surface=True):
            flow.add(op)
    flow.add(SurfaceFusion())

    flow.add(ResultAggregate())
    return flow


# ---------------------------------------------------------------------------
# Flow registry
# ---------------------------------------------------------------------------

_FLOWS: dict = {}
_FLOW_BUILDERS = {
    "template_create": build_template_creation_flow,
    "front_detect": build_front_detection_flow,
    "side_detect": build_side_detection_flow,
}


def get_flow(flow_id: str) -> Flow:
    """Get a flow by ID, building it on first access."""
    if flow_id in _FLOW_BUILDERS:
        if flow_id not in _FLOWS:
            _FLOWS[flow_id] = _FLOW_BUILDERS[flow_id]()
        return _FLOWS[flow_id]
    # Custom flow
    if flow_id not in _FLOWS:
        from registry import load_custom_flow, build_custom_flow
        data = load_custom_flow(flow_id)
        if data is None:
            raise ValueError(f"Unknown flow: {flow_id}")
        _FLOWS[flow_id] = build_custom_flow(
            data["id"], data["name"], data.get("description", ""), data["operators"])
    return _FLOWS[flow_id]


def list_flows() -> list:
    """List all available flows (built-in + custom)."""
    result = [{"id": fid, "name": b().name, "description": b().description,
               "operator_count": len(b().operators), "custom": False}
              for fid, b in _FLOW_BUILDERS.items()]
    try:
        from registry import list_custom_flows
        for cf in list_custom_flows():
            result.append({"id": cf["id"], "name": cf["name"],
                           "description": cf.get("description", ""),
                           "operator_count": len(cf.get("operators", [])),
                           "custom": True})
    except Exception:
        pass
    return result


def reset_flow(flow_id: str):
    """Force rebuild of a flow (e.g. after adding new operators)."""
    _FLOWS.pop(flow_id, None)
