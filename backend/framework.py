"""
Generic operator framework for building debuggable image-processing flows.

Core concepts:
  - Operator: one atomic step (reads ctx -> computes -> writes ctx -> returns viz+metrics)
  - Flow: an ordered list of operators + a context factory
  - FlowRunner: executes operators, supports partial re-run (from op N downstream)
  - Context: generic dict-like carrier for intermediate data

A flow is just a composition of operators. The framework is flow-agnostic:
template creation, front detection, side detection, and future custom flows
all use the same engine and the same debug UI.
"""
import time
import base64
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

# ---------------------------------------------------------------------------
# StepResult: what each operator produces for the UI
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str                       # unique operator name
    title: str                      # human-readable label
    group: str                      # UI grouping label
    image: Optional[str] = None     # base64 JPEG for visualization
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"              # ok | fail | skip | error | interactive
    error: Optional[str] = None
    param_keys: List[str] = field(default_factory=list)     # params this op reads
    input_keys: List[str] = field(default_factory=list)     # ctx keys this op reads
    output_keys: List[str] = field(default_factory=list)    # ctx keys this op writes
    interactive_data: Optional[Dict] = None                 # for interactive ops (e.g. ROI drawing)


# ---------------------------------------------------------------------------
# Context: generic carrier for intermediate data
# ---------------------------------------------------------------------------

class Context:
    """
    A generic dict-with-attributes carrier.
    Operators read/write keys freely. No fixed schema.
    """

    def __init__(self, **kwargs):
        self._data: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self._data[k] = v

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str):
        self._data.pop(key, None)

    def p(self, key: str, default=None):
        """Shorthand for getting a param from the 'params' dict."""
        params = self._data.get("params", {})
        return params.get(key, default)

    def to_serializable(self) -> Dict:
        """Return a JSON-serializable summary (no ndarrays)."""
        out = {}
        for k, v in self._data.items():
            if isinstance(v, np.ndarray):
                out[k] = f"<ndarray {v.shape}>"
            elif isinstance(v, (str, int, float, bool, list, dict, tuple)):
                try:
                    import json
                    json.dumps(v)
                    out[k] = v
                except Exception:
                    out[k] = str(v)
            else:
                out[k] = str(v)
        return out


# ---------------------------------------------------------------------------
# Operator: base class for all pipeline steps
# ---------------------------------------------------------------------------

class Operator:
    """
    Base class for an operator.

    Subclasses (or factory-created instances) must implement execute().
    Metadata (input_keys, output_keys, param_keys) is used by the runner
    for dependency analysis and by the UI for rendering controls.
    """

    def __init__(self, name: str, title: str, group: str,
                 param_keys: List[str] = None,
                 input_keys: List[str] = None,
                 output_keys: List[str] = None):
        self.name = name
        self.title = title
        self.group = group
        self.param_keys = param_keys or []
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []

    def execute(self, ctx: Context) -> StepResult:
        raise NotImplementedError

    def _result(self, image=None, metrics=None, status="ok", error=None,
                interactive_data=None) -> StepResult:
        return StepResult(
            name=self.name, title=self.title, group=self.group,
            image=_encode(image), metrics=metrics or {},
            status=status, error=error, param_keys=self.param_keys,
            input_keys=self.input_keys, output_keys=self.output_keys,
            interactive_data=interactive_data)


# ---------------------------------------------------------------------------
# Flow: an ordered composition of operators
# ---------------------------------------------------------------------------

@dataclass
class Flow:
    """A named sequence of operators."""
    id: str                         # e.g. "template_create", "front_detect"
    name: str                       # human-readable
    description: str
    operators: List[Operator] = field(default_factory=list)
    context_factory: Optional[Callable] = None  # creates initial Context for this flow

    def add(self, op: Operator) -> "Flow":
        self.operators.append(op)
        return self


# ---------------------------------------------------------------------------
# FlowRunner: executes operators with partial re-run support
# ---------------------------------------------------------------------------

class FlowRunner:
    """
    Executes a flow's operators in order.

    Supports:
      - run_all: execute every operator from scratch
      - run_from: execute from operator index N to the end (downstream re-run)
      - run_single: execute one operator (for debugging a single step)
    """

    @staticmethod
    def run_all(flow: Flow, ctx: Context) -> Tuple[List[StepResult], List[Dict]]:
        """Execute all operators. Returns (results, operator_meta)."""
        results = []
        meta = []
        for i, op in enumerate(flow.operators):
            t0 = time.time()
            try:
                sr = op.execute(ctx)
            except Exception as e:
                import traceback
                sr = StepResult(
                    name=op.name, title=op.title, group=op.group,
                    status="error", error=str(e),
                    param_keys=op.param_keys, input_keys=op.input_keys,
                    output_keys=op.output_keys)
            sr.metrics["time_ms"] = round((time.time() - t0) * 1000, 1)
            sr.metrics["op_index"] = i
            results.append(sr)
            meta.append({
                "index": i, "name": op.name, "title": op.title,
                "group": op.group, "param_keys": op.param_keys,
                "input_keys": op.input_keys, "output_keys": op.output_keys,
            })
            # If an operator fails critically, downstream ops will likely skip
            if sr.status == "fail" and op.name in ("roi_extract", "warp_align"):
                for j in range(i + 1, len(flow.operators)):
                    nop = flow.operators[j]
                    results.append(StepResult(
                        name=nop.name, title=nop.title, group=nop.group,
                        status="skip", param_keys=nop.param_keys))
                    meta.append({
                        "index": j, "name": nop.name, "title": nop.title,
                        "group": nop.group, "param_keys": nop.param_keys,
                        "input_keys": nop.input_keys, "output_keys": nop.output_keys,
                    })
                break
        return results, meta

    @staticmethod
    def run_from(flow: Flow, ctx: Context, start_index: int) -> List[StepResult]:
        """Execute from operator at start_index to the end."""
        results = []
        for i in range(start_index, len(flow.operators)):
            op = flow.operators[i]
            t0 = time.time()
            try:
                sr = op.execute(ctx)
            except Exception as e:
                sr = StepResult(
                    name=op.name, title=op.title, group=op.group,
                    status="error", error=str(e),
                    param_keys=op.param_keys)
            sr.metrics["time_ms"] = round((time.time() - t0) * 1000, 1)
            sr.metrics["op_index"] = i
            results.append(sr)
            if sr.status == "fail" and op.name in ("roi_extract", "warp_align"):
                for j in range(i + 1, len(flow.operators)):
                    nop = flow.operators[j]
                    results.append(StepResult(
                        name=nop.name, title=nop.title, group=nop.group,
                        status="skip"))
                break
        return results

    @staticmethod
    def run_single(flow: Flow, ctx: Context, op_index: int) -> StepResult:
        """Execute a single operator (for step-by-step debugging)."""
        op = flow.operators[op_index]
        t0 = time.time()
        try:
            sr = op.execute(ctx)
        except Exception as e:
            sr = StepResult(
                name=op.name, title=op.title, group=op.group,
                status="error", error=str(e),
                param_keys=op.param_keys)
        sr.metrics["time_ms"] = round((time.time() - t0) * 1000, 1)
        sr.metrics["op_index"] = op_index
        return sr

    @staticmethod
    def get_flow_meta(flow: Flow) -> List[Dict]:
        """Return operator metadata for UI rendering (without executing)."""
        return [{
            "index": i, "name": op.name, "title": op.title, "group": op.group,
            "param_keys": op.param_keys, "input_keys": op.input_keys,
            "output_keys": op.output_keys,
        } for i, op in enumerate(flow.operators)]


# ---------------------------------------------------------------------------
# Utility functions shared by all operators
# ---------------------------------------------------------------------------

def _encode(img) -> Optional[str]:
    """Encode image to base64 JPEG data URI."""
    if img is None:
        return None
    if not isinstance(img, np.ndarray):
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def overlay_mask(base: np.ndarray, mask: np.ndarray,
                 color=(0, 0, 255), alpha=0.4) -> np.ndarray:
    vis = base.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    mb = mask > 0
    if mb.any():
        ov = vis.copy()
        ov[mb] = color
        vis = cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0)
    return vis


def draw_contours(base: np.ndarray, contours, color=(0, 0, 255), thickness=2) -> np.ndarray:
    vis = base.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def hconcat_images(imgs: List[np.ndarray]) -> np.ndarray:
    """Concatenate images horizontally (for side-by-side display)."""
    normalized = []
    for img in imgs:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        normalized.append(img)
    h = min(img.shape[0] for img in normalized)
    resized = []
    for img in normalized:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h))
        resized.append(img)
    return cv2.hconcat(resized)


def post_process_defects(binary_defects: np.ndarray, region_mask: np.ndarray,
                         min_defect_area: float, diff_image=None,
                         kernelsz: int = 3) -> Dict:
    """
    Shared post-processing: morphology + contour extraction + area filter.
    Returns a dict mimicking BorderDefectResult fields.
    """
    if kernelsz > 1:
        kernel = np.ones((kernelsz, kernelsz), np.uint8)
        binary = cv2.morphologyEx(binary_defects, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        binary = binary_defects

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_defect_area]

    filtered_binary = np.zeros_like(binary)
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


def preprocess_region(template_img: np.ndarray, target_img: np.ndarray,
                      region_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract gray template and target regions within the mask."""
    tpl_gray = to_gray(template_img)
    tgt_gray = to_gray(target_img)
    if region_mask is not None and region_mask.shape == tpl_gray.shape:
        tpl_region = cv2.bitwise_and(tpl_gray, tpl_gray, mask=region_mask)
        tgt_region = cv2.bitwise_and(tgt_gray, tgt_gray, mask=region_mask)
    else:
        tpl_region = tpl_gray
        tgt_region = tgt_gray
    return tpl_region, tgt_region


def compute_surface_mask(border_mask: np.ndarray,
                         ball_mask: np.ndarray = None) -> np.ndarray:
    """Surface = everything inside foreground minus border minus ball."""
    if border_mask is None:
        return None
    sm = np.ones_like(border_mask) * 255
    sm[border_mask > 0] = 0
    if ball_mask is not None and ball_mask.shape == sm.shape:
        sm[ball_mask > 0] = 0
    return sm
