"""Flask server for sorteradc-helper.

Flow-agnostic API: any flow (template creation, detection tuning, etc.)
uses the same endpoints. The server just routes to the flow engine.
"""
import os
import sys
import json
import base64
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from framework import FlowRunner, Context
from flows import get_flow, list_flows, reset_flow
from param_schema import get_schema, get_defaults
import template_loader

app = Flask(__name__, static_folder=os.path.join(HERE, "..", "frontend"))
CORS(app)

# Caches
_template_cache = {}      # product_id -> TemplateData
_flow_contexts = {}      # session_id -> Context (for partial re-runs)


def _get_template(product_id):
    if product_id not in _template_cache:
        td = template_loader.load_main_template(product_id)
        if td is not None:
            _template_cache[product_id] = td
    return _template_cache.get(product_id)


# ---- Static frontend ----

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ---- Flow API (generic, works with any flow) ----

@app.route("/api/flows")
def api_flows():
    return jsonify(list_flows())


@app.route("/api/flows/<flow_id>/meta")
def api_flow_meta(flow_id):
    try:
        flow = get_flow(flow_id)
        meta = FlowRunner.get_flow_meta(flow)
        return jsonify({"flow_id": flow_id, "name": flow.name,
                        "description": flow.description, "operators": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/flows/<flow_id>/run", methods=["POST"])
def api_flow_run(flow_id):
    """Run a flow (full or partial).

    Request body:
    {
        "image": "base64...",          # input image (for detection flows)
        "product_id": "M34256...",     # product ID (loads template)
        "params": {...},               # parameter overrides
        "start_from": 0,               # optional: start from operator N
        "interactive_data": {...},     # optional: data from interactive ops (e.g. ROI)
    }
    """
    try:
        flow = get_flow(flow_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    data = request.get_json()
    image_b64 = data.get("image", "")
    product_id = data.get("product_id", "")
    custom_params = data.get("params", {})
    start_from = data.get("start_from", 0)
    interactive_data = data.get("interactive_data", {})
    new_product_id = data.get("new_product_id", "")

    # Build context
    ctx_kwargs = {"params": _merge_params(product_id, custom_params)}

    # Load image
    if image_b64:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        import numpy as np, cv2
        img_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400
        ctx_kwargs["image"] = image

    # Load template (for detection flows)
    if product_id:
        td = _get_template(product_id)
        if td is None:
            return jsonify({"error": f"Template not found: {product_id}"}), 404
        ctx_kwargs["template"] = td.template_image
        ctx_kwargs["mask_border"] = td.mask_border
        ctx_kwargs["mask_ball"] = td.mask_ball
        ctx_kwargs["mask_foreground"] = td.mask_foreground
        ctx_kwargs["mask_trace"] = td.mask_trace
        ctx_kwargs["mask_expand"] = td.mask_expand
        ctx_kwargs["tpl_gx"] = td.tpl_gx
        ctx_kwargs["tpl_gy"] = td.tpl_gy
        ctx_kwargs["tpl_mag"] = td.tpl_mag
        ctx_kwargs["tpl_contour"] = td.tpl_contour
        ctx_kwargs["tpl_xym"] = td.tpl_xym

    # Interactive data (e.g. user-drawn ROI rectangle)
    ctx_kwargs["new_product_id"] = new_product_id
    if interactive_data:
        if "roi" in interactive_data:
            ctx_kwargs["user_roi"] = tuple(interactive_data["roi"])

    ctx = Context(**ctx_kwargs)

    # Execute
    try:
        if start_from > 0:
            results = FlowRunner.run_from(flow, ctx, start_from)
            meta = FlowRunner.get_flow_meta(flow)[start_from:]
        else:
            results, meta = FlowRunner.run_all(flow, ctx)

        # Serialize results
        steps = []
        for sr in results:
            steps.append({
                "name": sr.name, "title": sr.title, "group": sr.group,
                "image": sr.image, "metrics": sr.metrics, "status": sr.status,
                "error": sr.error, "param_keys": sr.param_keys,
                "interactive_data": sr.interactive_data,
            })

        return jsonify({
            "steps": steps,
            "operators": meta,
            "overall": ctx.get("overall_result", "OK"),
            "ng_reasons": ctx.get("ng_reasons", []),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _merge_params(product_id, custom_params):
    """Merge defaults + stored params + custom overrides."""
    params = get_defaults()
    if product_id:
        td = _get_template(product_id)
        if td:
            params.update(td.params)
    params.update(custom_params)
    return params


# ---- Product/Template API ----

@app.route("/api/products")
def api_products():
    return jsonify(template_loader.list_products())


@app.route("/api/products/<product_id>/params")
def api_params(product_id):
    td = _get_template(product_id)
    if td is None:
        return jsonify({"error": f"Product {product_id} not found"}), 404
    defaults = get_defaults()
    defaults.update(td.params)
    return jsonify({
        "product_id": product_id,
        "params": defaults,
        "meta": {"width": td.meta.get("width", 0), "height": td.meta.get("height", 0),
                  "state": td.meta.get("state", "unknown")},
        "sides": list(td.side_templates.keys()) if td.side_templates else [],
    })


@app.route("/api/products/<product_id>/params", methods=["POST"])
def api_save_params(product_id):
    data = request.get_json()
    params = data.get("params", {})
    if product_id in _template_cache:
        del _template_cache[product_id]
    ok = template_loader.save_params(product_id, params)
    return jsonify({"success": ok})


@app.route("/api/products/<product_id>/clear-cache", methods=["POST"])
def api_clear_cache(product_id):
    _template_cache.pop(product_id, None)
    return jsonify({"success": True})


# ---- Test images ----

@app.route("/api/test-images")
def api_test_images():
    return jsonify(template_loader.list_test_images())


@app.route("/api/load-image")
def api_load_image():
    from flask import send_file
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="image/jpeg")


# ---- Param schema ----

@app.route("/api/param-schema")
def api_schema():
    return jsonify(get_schema())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8089))
    print(f"sorteradc-helper starting on port {port}")
    print(f"  POC_ROOT = {os.environ.get('POC_ROOT', '/app')}")
    print(f"  TESTDATA_DIR = {os.environ.get('TESTDATA_DIR', '/testdata')}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
