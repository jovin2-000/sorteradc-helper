"""
Operator registry: all available operators pre-instantiated with metadata.

This is the source of truth for the flow builder UI. Each entry has:
  name:        unique identifier
  title:       display label
  group:       UI grouping
  class_name:  Python class name
  instance:    pre-instantiated Operator (or None if parameterized)

Parameterized operators (e.g. RegionPreprocess with region/label args)
are registered as "operator specs" with their constructor arguments filled.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework import Operator
from ops.matching import (ROIExtract, SIFTDetect, SIFTMatch, RANSACPose,
                           ShapeMatchFallback, WarpAlign, MaskApply)
from ops.mask_create import (ROISelect, BoundaryOptimize, FeatureExtract,
                              ForegroundMask, BorderMask, BallMask, TraceMask, SaveTemplate)
from ops.defect import (RegionPreprocess, FreqFilter, SpatialCLAHE, SpatialGradient,
                         TplBinary, TplContourDiff, AbsDiff, ThresholdPostproc)
from ops.ball import BallMaskProc, BallBinarize, BallConnect, BallAnalyze
from ops.result import BorderFusion, BorderClassify, SurfaceFusion, ResultAggregate
import json

# All operators, instantiated with their arguments
_ALL_OPERATORS = []


def _add(op):
    _ALL_OPERATORS.append(op)


# --- 定位摆正 ---
_add(ROIExtract())
_add(SIFTDetect())
_add(SIFTMatch())
_add(RANSACPose())
_add(ShapeMatchFallback())
_add(WarpAlign())
_add(MaskApply())

# --- 模板创建 ---
_add(ROISelect())
_add(BoundaryOptimize())
_add(FeatureExtract())
_add(ForegroundMask())
_add(BorderMask())
_add(BallMask())
_add(TraceMask())
_add(SaveTemplate())

# --- 边框-频域法 ---
_add(RegionPreprocess("border_frequency", "边框-频域法", use_surface_mask=False))
_add(FreqFilter("border_frequency", "边框-频域法"))
_add(ThresholdPostproc("border_frequency", "边框-频域法", "frequency", "border_frequency_diff_threshold"))

# --- 边框-空域法 ---
_add(RegionPreprocess("border_spatial", "边框-空域法", use_surface_mask=False))
_add(SpatialCLAHE("border_spatial", "边框-空域法"))
_add(SpatialGradient("border_spatial", "边框-空域法"))
_add(ThresholdPostproc("border_spatial", "边框-空域法", "spatial", "border_gradient_threshold", "grad_diff"))

# --- 边框-模板法 ---
_add(RegionPreprocess("border_template", "边框-模板法", use_surface_mask=False))
_add(TplBinary("border_template", "边框-模板法"))
_add(TplContourDiff("border_template", "边框-模板法"))
_add(ThresholdPostproc("border_template", "边框-模板法", "template", "border_binary_threshold"))

# --- 边框-差分法 ---
_add(RegionPreprocess("border_difference", "边框-差分法", use_surface_mask=False))
_add(AbsDiff("border_difference", "边框-差分法"))
_add(ThresholdPostproc("border_difference", "边框-差分法", "difference", "border_diff_diff_threshold"))

# --- 边框融合 ---
_add(BorderFusion())
_add(BorderClassify())

# --- 球检测 ---
_add(BallMaskProc())
_add(BallBinarize())
_add(BallConnect())
_add(BallAnalyze())

# --- 表面-频域法 ---
_add(RegionPreprocess("surface_frequency", "表面-频域法", use_surface_mask=True))
_add(FreqFilter("surface_frequency", "表面-频域法"))
_add(ThresholdPostproc("surface_frequency", "表面-频域法", "frequency", "surface_frequency_diff_threshold"))

# --- 表面-空域法 ---
_add(RegionPreprocess("surface_spatial", "表面-空域法", use_surface_mask=True))
_add(SpatialCLAHE("surface_spatial", "表面-空域法"))
_add(SpatialGradient("surface_spatial", "表面-空域法"))
_add(ThresholdPostproc("surface_spatial", "表面-空域法", "spatial", "surface_gradient_threshold", "grad_diff"))

# --- 表面-模板法 ---
_add(RegionPreprocess("surface_template", "表面-模板法", use_surface_mask=True))
_add(TplBinary("surface_template", "表面-模板法"))
_add(TplContourDiff("surface_template", "表面-模板法"))
_add(ThresholdPostproc("surface_template", "表面-模板法", "template", "surface_binary_threshold"))

# --- 表面-差分法 ---
_add(RegionPreprocess("surface_difference", "表面-差分法", use_surface_mask=True))
_add(AbsDiff("surface_difference", "表面-差分法"))
_add(ThresholdPostproc("surface_difference", "表面-差分法", "difference", "surface_diff_diff_threshold"))

# --- 表面融合 ---
_add(SurfaceFusion())

# --- 结果汇总 ---
_add(ResultAggregate())


# Build lookup by name
_BY_NAME = {op.name: op for op in _ALL_OPERATORS}


def list_operators():
    """Return metadata for all registered operators."""
    return [{
        "name": op.name,
        "title": op.title,
        "group": op.group,
        "param_keys": op.param_keys,
        "input_keys": op.input_keys,
        "output_keys": op.output_keys,
    } for op in _ALL_OPERATORS]


def get_operator(name: str):
    """Get an operator instance by name."""
    return _BY_NAME.get(name)


def build_custom_flow(flow_id: str, name: str, description: str,
                      op_names: list):
    """Build a Flow from a list of operator names."""
    from framework import Flow
    flow = Flow(id=flow_id, name=name, description=description)
    for op_name in op_names:
        op = get_operator(op_name)
        if op:
            flow.add(op)
        else:
            raise ValueError(f"Unknown operator: {op_name}")
    return flow


# --- Custom flow persistence ---
_CUSTOM_FLOWS_DIR = os.path.join(os.path.dirname(__file__), "..", "custom_flows")


def _ensure_dir():
    os.makedirs(_CUSTOM_FLOWS_DIR, exist_ok=True)


def list_custom_flows():
    """List all saved custom flows."""
    _ensure_dir()
    flows = []
    for f in sorted(os.listdir(_CUSTOM_FLOWS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(_CUSTOM_FLOWS_DIR, f), "r", encoding="utf-8") as fh:
                data = json.load(fh)
                flows.append(data)
    return flows


def save_custom_flow(flow_id: str, name: str, description: str, op_names: list):
    """Save a custom flow definition to disk."""
    _ensure_dir()
    data = {"id": flow_id, "name": name, "description": description,
            "operators": op_names}
    path = os.path.join(_CUSTOM_FLOWS_DIR, f"{flow_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def delete_custom_flow(flow_id: str):
    """Delete a custom flow."""
    path = os.path.join(_CUSTOM_FLOWS_DIR, f"{flow_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def load_custom_flow(flow_id: str):
    """Load a custom flow definition."""
    path = os.path.join(_CUSTOM_FLOWS_DIR, f"{flow_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
