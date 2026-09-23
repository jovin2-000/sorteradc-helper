"""Load product templates from the sorteradc-poc data directory.

Knows the actual on-disk file naming convention which differs from
the ProductModel dataclass field names:
  roi.npy        -> shape_match_template (template image)
  tpl_gx.npy     -> shape_match_tpl_gx
  tpl_gy.npy     -> shape_match_tpl_gy
  tpl_mag.npy    -> shape_match_tpl_mag
  tpl_contour.npy-> shape_match_tpl_contour_points
  tpl_xym.npy    -> shape_match_tpl_xym
  mask_*.npy     -> same name
"""
import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

POC_ROOT = os.environ.get("POC_ROOT", "/app")

# Map on-disk filename (without .npy) -> logical field name
MAIN_FILE_MAP = {
    "roi": "shape_match_template",
    "tpl_gx": "shape_match_tpl_gx",
    "tpl_gy": "shape_match_tpl_gy",
    "tpl_mag": "shape_match_tpl_mag",
    "tpl_contour": "shape_match_tpl_contour_points",
    "tpl_xym": "shape_match_tpl_xym",
    "mask_all_foreground": "mask_all_foreground",
    "mask_ball_only": "mask_ball_only",
    "mask_border_only": "mask_border_only",
    "mask_trace_only": "mask_trace_only",
    "mask_expand_only": "mask_expand_only",
}

SIDE_FILE_MAP = {
    "tpl": "tpl_image",
    "tpl_gx": "tpl_gx",
    "tpl_gy": "tpl_gy",
    "tpl_mag": "tpl_mag",
    "tpl_contour": "tpl_contour_points",
    "tpl_xym": "tpl_xym",
    "mask_all_foreground": "mask_all_foreground",
    "mask_ball_only": "mask_ball_only",
    "mask_border_only": "mask_border_only",
    "mask_trace_only": "mask_trace_only",
    "mask_expand_only": "mask_expand_only",
}

VALID_SIDES = [
    "left_l00", "left_l01", "right_l00", "right_l01",
    "back_l00", "back_l01", "front_l00", "front_l01",
]


@dataclass
class TemplateData:
    """Loaded template data ready for pipeline use."""
    product_id: str
    meta: Dict[str, Any]
    params: Dict[str, Any]          # flat detection_params dict
    template_image: np.ndarray      # shape_match_template (BGR or gray)
    mask_border: np.ndarray
    mask_ball: np.ndarray
    mask_foreground: np.ndarray
    mask_trace: Optional[np.ndarray] = None
    mask_expand: Optional[np.ndarray] = None
    tpl_gx: Optional[np.ndarray] = None
    tpl_gy: Optional[np.ndarray] = None
    tpl_mag: Optional[np.ndarray] = None
    tpl_contour: Optional[np.ndarray] = None
    tpl_xym: Optional[Any] = None
    side_templates: Dict[str, "SideTemplateData"] = None


@dataclass
class SideTemplateData:
    side_name: str
    width: int
    height: int
    tpl_image: np.ndarray
    mask_border: np.ndarray
    mask_ball: np.ndarray
    mask_foreground: np.ndarray
    mask_trace: Optional[np.ndarray] = None
    mask_expand: Optional[np.ndarray] = None
    tpl_gx: Optional[np.ndarray] = None
    tpl_gy: Optional[np.ndarray] = None
    tpl_mag: Optional[np.ndarray] = None
    tpl_contour: Optional[np.ndarray] = None
    tpl_xym: Optional[Any] = None


def _templates_root() -> Path:
    return Path(POC_ROOT) / "data" / "templates"


def list_products() -> List[Dict[str, Any]]:
    """List all product IDs that have templates."""
    root = _templates_root()
    if not root.exists():
        return []
    products = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        # Find version directories
        version_dirs = [v for v in d.iterdir() if v.is_dir() and v.name.startswith("version")]
        if not version_dirs:
            continue
        # Use the latest version
        version_dirs.sort(key=lambda v: v.name)
        vdir = version_dirs[-1]
        meta_path = vdir / "main" / "model_meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Count available sides
            sides_dir = vdir / "sides"
            sides = []
            if sides_dir.exists():
                sides = [s.name for s in sides_dir.iterdir() if s.is_dir()]
            products.append({
                "product_id": d.name,
                "version": vdir.name,
                "state": meta.get("state", "unknown"),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "sides": sides,
            })
        except Exception as e:
            products.append({"product_id": d.name, "version": vdir.name, "error": str(e)})
    return products


def _load_npy(dir_path: Path, filename: str) -> Optional[np.ndarray]:
    """Load a .npy file, return None if not found."""
    p = dir_path / f"{filename}.npy"
    if p.exists():
        return np.load(str(p), allow_pickle=True)
    return None


def _load_params(meta: Dict) -> Dict[str, Any]:
    """Extract detection_params from meta, handle both flat and nested structures."""
    dp = meta.get("detection_params", {})
    if not dp:
        return {}
    # If detection_params is a DetectionParams dataclass dict (flat), return as-is
    # If it's nested with "main" and "sides", return the "main" part
    if "main" in dp and isinstance(dp["main"], dict):
        return dp["main"]
    if "sides" in dp and isinstance(dp.get("main"), dict):
        return dp["main"]
    return dp


def load_main_template(product_id: str) -> Optional[TemplateData]:
    """Load the main (front) template for a product."""
    root = _templates_root()
    product_dir = root / product_id
    if not product_dir.exists():
        return None

    # Find latest version
    version_dirs = [v for v in product_dir.iterdir() if v.is_dir() and v.name.startswith("version")]
    if not version_dirs:
        return None
    version_dirs.sort(key=lambda v: v.name)
    vdir = version_dirs[-1]

    main_dir = vdir / "main"
    meta_path = main_dir / "model_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    params = _load_params(meta)

    # Load arrays using the file map
    arrays = {}
    for disk_name, logical_name in MAIN_FILE_MAP.items():
        arr = _load_npy(main_dir, disk_name)
        if arr is not None:
            arrays[logical_name] = arr

    template_image = arrays.get("shape_match_template")
    if template_image is None:
        # Fallback: try roi.png
        roi_png = main_dir / "roi.png"
        if roi_png.exists():
            template_image = cv2_imread(str(roi_png))

    if template_image is None:
        return None

    # Load side templates
    side_templates = {}
    sides_dir = vdir / "sides"
    if sides_dir.exists():
        for side_dir in sides_dir.iterdir():
            if side_dir.is_dir() and side_dir.name in VALID_SIDES:
                st = _load_side_template(side_dir)
                if st:
                    side_templates[side_dir.name] = st

    return TemplateData(
        product_id=product_id,
        meta=meta,
        params=params,
        template_image=template_image,
        mask_border=arrays.get("mask_border_only", np.zeros_like(template_image.shape[:2], dtype=np.uint8)),
        mask_ball=arrays.get("mask_ball_only", np.zeros(template_image.shape[:2], dtype=np.uint8)),
        mask_foreground=arrays.get("mask_all_foreground", np.ones(template_image.shape[:2], dtype=np.uint8)),
        mask_trace=arrays.get("mask_trace_only"),
        mask_expand=arrays.get("mask_expand_only"),
        tpl_gx=arrays.get("shape_match_tpl_gx"),
        tpl_gy=arrays.get("shape_match_tpl_gy"),
        tpl_mag=arrays.get("shape_match_tpl_mag"),
        tpl_contour=arrays.get("shape_match_tpl_contour_points"),
        tpl_xym=arrays.get("shape_match_tpl_xym"),
        side_templates=side_templates,
    )


def _load_side_template(side_dir: Path) -> Optional[SideTemplateData]:
    """Load a single side template."""
    meta_path = side_dir / "side_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    arrays = {}
    for disk_name, logical_name in SIDE_FILE_MAP.items():
        arr = _load_npy(side_dir, disk_name)
        if arr is not None:
            arrays[logical_name] = arr

    tpl_image = arrays.get("tpl_image")
    if tpl_image is None:
        tpl_png = side_dir / "tpl.png"
        if tpl_png.exists():
            tpl_image = cv2_imread(str(tpl_png))
    if tpl_image is None:
        return None

    h, w = tpl_image.shape[:2]
    return SideTemplateData(
        side_name=meta.get("side_name", side_dir.name),
        width=meta.get("width", w),
        height=meta.get("height", h),
        tpl_image=tpl_image,
        mask_border=arrays.get("mask_border_only", np.zeros((h, w), dtype=np.uint8)),
        mask_ball=arrays.get("mask_ball_only", np.zeros((h, w), dtype=np.uint8)),
        mask_foreground=arrays.get("mask_all_foreground", np.ones((h, w), dtype=np.uint8)),
        mask_trace=arrays.get("mask_trace_only"),
        mask_expand=arrays.get("mask_expand_only"),
        tpl_gx=arrays.get("tpl_gx"),
        tpl_gy=arrays.get("tpl_gy"),
        tpl_mag=arrays.get("tpl_mag"),
        tpl_contour=arrays.get("tpl_contour"),
        tpl_xym=arrays.get("tpl_xym"),
    )


def cv2_imread(path: str) -> Optional[np.ndarray]:
    """Read image with cv2, handling unicode paths."""
    import cv2
    try:
        return cv2.imread(path)
    except Exception:
        return None


def save_params(product_id: str, params: Dict[str, Any]) -> bool:
    """Save detection params back to model_meta.json."""
    root = _templates_root()
    product_dir = root / product_id
    if not product_dir.exists():
        return False

    version_dirs = [v for v in product_dir.iterdir() if v.is_dir() and v.name.startswith("version")]
    if not version_dirs:
        return False
    version_dirs.sort(key=lambda v: v.name)
    vdir = version_dirs[-1]

    meta_path = vdir / "main" / "model_meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Update detection_params
    dp = meta.get("detection_params", {})
    if "main" in dp and isinstance(dp["main"], dict):
        dp["main"].update(params)
    else:
        # Flat structure - update directly
        dp.update(params)
    meta["detection_params"] = dp

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    return True


def list_test_images() -> List[Dict[str, str]]:
    """List available test images from the testdata directory."""
    testdata_dir = Path(os.environ.get("TESTDATA_DIR", "/testdata"))
    images = []
    if not testdata_dir.exists():
        return images

    # Front images
    front_dir = testdata_dir / "正面"
    if front_dir.exists():
        for f in sorted(front_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".png", ".bmp"):
                images.append({
                    "path": str(f),
                    "name": f.name,
                    "type": "main",
                    "label": f"正面/{f.name}",
                })

    # Side images
    side_dir = testdata_dir / "侧面"
    if side_dir.exists():
        for subdir in sorted(side_dir.iterdir()):
            if subdir.is_dir():
                for f in sorted(subdir.iterdir()):
                    if f.suffix.lower() in (".jpg", ".png", ".bmp"):
                        side_name = subdir.name.lower()  # L00, L01
                        images.append({
                            "path": str(f),
                            "name": f.name,
                            "type": "side",
                            "side": side_name,
                            "label": f"侧面/{subdir.name}/{f.name}",
                        })

    return images
