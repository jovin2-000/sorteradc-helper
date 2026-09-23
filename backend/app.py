"""Flask server for sorteradc-helper: template tuning and visualization tool.

Serves both the REST API and the static frontend.
"""
import os
import sys
import json
import base64
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure backend modules are importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from param_schema import get_schema, get_defaults
import template_loader
import pipeline

app = Flask(__name__, static_folder=os.path.join(HERE, "..", "frontend"))
CORS(app)

# Template cache: product_id -> TemplateData
_template_cache = {}


def _get_template(product_id):
    """Load template with caching."""
    if product_id not in _template_cache:
        td = template_loader.load_main_template(product_id)
        if td is not None:
            _template_cache[product_id] = td
    return _template_cache.get(product_id)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ---- API endpoints ----

@app.route("/api/param-schema")
def api_schema():
    return jsonify(get_schema())


@app.route("/api/products")
def api_products():
    return jsonify(template_loader.list_products())


@app.route("/api/products/<product_id>/params")
def api_params(product_id):
    td = _get_template(product_id)
    if td is None:
        return jsonify({"error": f"Product {product_id} not found"}), 404
    # Merge stored params with defaults
    defaults = get_defaults()
    defaults.update(td.params)
    return jsonify({
        "product_id": product_id,
        "params": defaults,
        "meta": {
            "width": td.meta.get("width", 0),
            "height": td.meta.get("height", 0),
            "state": td.meta.get("state", "unknown"),
        },
        "sides": list(td.side_templates.keys()) if td.side_templates else [],
    })


@app.route("/api/products/<product_id>/template-images")
def api_template_images(product_id):
    """Return base64 preview images of template and masks."""
    import cv2
    import numpy as np
    td = _get_template(product_id)
    if td is None:
        return jsonify({"error": f"Product {product_id} not found"}), 404

    def enc(img):
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    result = {
        "template": enc(td.template_image),
        "mask_border": enc(td.mask_border),
        "mask_ball": enc(td.mask_ball),
        "mask_foreground": enc(td.mask_foreground),
        "mask_trace": enc(td.mask_trace),
        "mask_expand": enc(td.mask_expand),
    }
    return jsonify(result)


@app.route("/api/products/<product_id>/params", methods=["POST"])
def api_save_params(product_id):
    """Save detection params to model_meta.json."""
    data = request.get_json()
    params = data.get("params", {})
    # Clear cache so params reload on next access
    if product_id in _template_cache:
        del _template_cache[product_id]
    ok = template_loader.save_params(product_id, params)
    return jsonify({"success": ok})


@app.route("/api/test-images")
def api_test_images():
    return jsonify(template_loader.list_test_images())


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Run detection pipeline step-by-step.

    Request body:
    {
        "product_id": "M34256...",
        "image": "base64... or data:image/...;base64,...",
        "params": { ... },          # optional, overrides stored params
        "detection_type": "main",    # "main" or "side"
        "side_name": "left_l00"     # required if detection_type == "side"
    }
    """
    import cv2
    import numpy as np

    data = request.get_json()
    product_id = data.get("product_id")
    image_b64 = data.get("image", "")
    detection_type = data.get("detection_type", "main")
    side_name = data.get("side_name")
    custom_params = data.get("params")

    if not product_id or not image_b64:
        return jsonify({"error": "product_id and image are required"}), 400

    # Decode image
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_b64)
    img_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Load template
    td = _get_template(product_id)
    if td is None:
        return jsonify({"error": f"Template not found for {product_id}"}), 404

    # Merge params: stored + custom overrides
    from param_schema import get_defaults
    params = get_defaults()
    params.update(td.params)
    if custom_params:
        params.update(custom_params)

    try:
        if detection_type == "side":
            if not side_name:
                return jsonify({"error": "side_name is required for side detection"}), 400
            result = pipeline.run_side_detection(td, side_name, image, params)
        else:
            result = pipeline.run_main_detection(td, image, params)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/products/<product_id>/clear-cache", methods=["POST"])
def api_clear_cache(product_id):
    if product_id in _template_cache:
        del _template_cache[product_id]
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8089))
    print(f"sorteradc-helper starting on port {port}")
    print(f"  POC_ROOT = {os.environ.get('POC_ROOT', '/app')}")
    print(f"  TESTDATA_DIR = {os.environ.get('TESTDATA_DIR', '/testdata')}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)


@app.route("/api/load-image")
def api_load_image():
    """Load an image file by path (for test image selection)."""
    from flask import send_file
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "path is required"}), 400
    if not os.path.exists(path):
        return jsonify({"error": f"File not found: {path}"}), 404
    return send_file(path, mimetype="image/jpeg")
