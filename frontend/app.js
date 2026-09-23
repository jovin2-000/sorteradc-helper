// sorteradc-helper frontend logic

const State = {
  schema: null,
  products: [],
  currentProduct: null,
  params: {},
  image: null,         // base64 string
  imageName: null,
  detectionType: "main",
  sideName: null,
  testImages: [],
  collapsedCategories: new Set(),
  rerunTimer: null,
};

const API = "";

// ---- Utility ----
function $(id) { return document.getElementById(id); }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}
function setStatus(text, cls) {
  const s = $("status-indicator");
  s.textContent = text;
  s.className = cls || "status-idle";
}

// ---- Init ----
async function init() {
  // Load schema
  try {
    const r = await fetch(`${API}/api/param-schema`);
    State.schema = await r.json();
  } catch (e) {
    console.error("Failed to load schema", e);
    $("param-panel").innerHTML = '<div class="sidebar-loading">Schema加载失败</div>';
    return;
  }

  // Load products
  try {
    const r = await fetch(`${API}/api/products`);
    State.products = await r.json();
    const sel = $("product-select");
    sel.innerHTML = '<option value="">选择产品...</option>';
    for (const p of State.products) {
      const opt = document.createElement("option");
      opt.value = p.product_id;
      opt.textContent = `${p.product_id} (${p.state})`;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error("Failed to load products", e);
  }

  // Load test images
  try {
    const r = await fetch(`${API}/api/test-images`);
    State.testImages = await r.json();
    const sel = $("test-image-select");
    sel.innerHTML = '<option value="">或选择测试图...</option>';
    for (const img of State.testImages) {
      const opt = document.createElement("option");
      opt.value = img.path;
      opt.textContent = img.label;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error("Failed to load test images", e);
  }

  // Build parameter panel
  buildParamPanel();

  // Event listeners
  bindEvents();
}

// ---- Build parameter panel from schema ----
function buildParamPanel() {
  const panel = $("param-panel");
  panel.innerHTML = "";

  // Group params by category
  const byCat = {};
  for (const p of State.schema.params) {
    if (!byCat[p.category]) byCat[p.category] = [];
    byCat[p.category].push(p);
  }

  for (const cat of State.schema.categories) {
    const params = byCat[cat.id];
    if (!params || params.length === 0) continue;

    const section = el("div", "param-category");
    if (State.collapsedCategories.has(cat.id)) section.classList.add("collapsed");

    const header = el("div", "param-category-header");
    header.innerHTML = `<span>${cat.title}</span><span class="chevron">&#9660;</span>`;
    header.onclick = () => {
      section.classList.toggle("collapsed");
      if (section.classList.contains("collapsed"))
        State.collapsedCategories.add(cat.id);
      else
        State.collapsedCategories.delete(cat.id);
    };

    const body = el("div", "param-category-body");
    for (const p of params) {
      body.appendChild(buildParamRow(p));
    }

    section.appendChild(header);
    section.appendChild(body);
    panel.appendChild(section);
  }
}

function buildParamRow(p) {
  const row = el("div", "param-row");
  const label = el("span", "param-label", p.label);
  label.title = p.label;
  row.appendChild(label);

  const control = el("div", "param-control");

  switch (p.type) {
    case "bool":
      const toggle = el("div", "param-toggle");
      if (State.params[p.name]) toggle.classList.add("on");
      toggle.onclick = () => {
        State.params[p.name] = !State.params[p.name];
        toggle.classList.toggle("on");
        scheduleRerun();
      };
      control.appendChild(toggle);
      break;

    case "float":
    case "int":
      const slider = el("input", "param-slider");
      slider.type = "range";
      slider.min = p.min;
      slider.max = p.max;
      slider.step = p.step;
      slider.value = State.params[p.name] ?? p.default;
      const input = el("input", "param-input");
      input.type = "number";
      input.min = p.min;
      input.max = p.max;
      input.step = p.step;
      input.value = State.params[p.name] ?? p.default;
      slider.oninput = () => {
        State.params[p.name] = parseFloat(slider.value);
        input.value = slider.value;
        scheduleRerun();
      };
      input.onchange = () => {
        let v = parseFloat(input.value);
        if (isNaN(v)) v = p.default;
        v = Math.max(p.min, Math.min(p.max, v));
        State.params[p.name] = v;
        slider.value = v;
        input.value = v;
        scheduleRerun();
      };
      control.appendChild(slider);
      control.appendChild(input);
      break;

    case "select":
      const sel = el("select", "param-select");
      for (const opt of p.options) {
        const o = el("option", null, opt);
        if (opt === State.params[p.name]) o.selected = true;
        sel.appendChild(o);
      }
      sel.onchange = () => {
        State.params[p.name] = sel.value;
        scheduleRerun();
      };
      control.appendChild(sel);
      break;

    case "multiselect":
      const chips = el("div", "multiselect-row");
      for (const opt of p.options) {
        const chip = el("span", "multiselect-chip", opt);
        if (State.params[p.name] && State.params[p.name].includes(opt))
          chip.classList.add("active");
        chip.onclick = () => {
          let arr = State.params[p.name] || [];
          if (arr.includes(opt)) {
            arr = arr.filter(x => x !== opt);
          } else {
            arr = [...arr, opt];
          }
          State.params[p.name] = arr;
          chip.classList.toggle("active");
          scheduleRerun();
        };
        chips.appendChild(chip);
      }
      control.appendChild(chips);
      break;

    case "weights":
      const w = State.params[p.name] || p.default;
      for (const opt of p.options) {
        const wr = el("div", "weight-row");
        const wl = el("span", "weight-label", opt);
        const wi = el("input", "weight-input");
        wi.type = "number";
        wi.step = "0.05";
        wi.min = "0";
        wi.max = "1";
        wi.value = w[opt] ?? 0;
        wi.onchange = () => {
          let v = parseFloat(wi.value);
          if (isNaN(v)) v = 0;
          v = Math.max(0, Math.min(1, v));
          if (!State.params[p.name]) State.params[p.name] = {};
          State.params[p.name][opt] = v;
          wi.value = v;
          scheduleRerun();
        };
        wr.appendChild(wl);
        wr.appendChild(wi);
        control.appendChild(wr);
      }
      break;

    case "tuple_float":
      const arr = State.params[p.name] || p.default;
      const ti1 = el("input", "param-input");
      ti1.type = "number";
      ti1.step = "0.1";
      ti1.value = arr[0] ?? 0;
      const ti2 = el("input", "param-input");
      ti2.type = "number";
      ti2.step = "0.1";
      ti2.value = arr[1] ?? 0;
      const updater = () => {
        State.params[p.name] = [parseFloat(ti1.value) || 0, parseFloat(ti2.value) || 0];
        scheduleRerun();
      };
      ti1.onchange = updater;
      ti2.onchange = updater;
      control.appendChild(ti1);
      control.appendChild(ti2);
      break;
  }

  row.appendChild(control);
  return row;
}

// ---- Event bindings ----
function bindEvents() {
  $("product-select").onchange = onProductChange;
  $("detection-type").onchange = onDetectionTypeChange;
  $("btn-upload").onclick = () => $("file-input").click();
  $("file-input").onchange = onFileUpload;
  $("test-image-select").onchange = onTestImageSelect;
  $("btn-run").onclick = runDetection;
  $("btn-save").onclick = saveParams;
  $("btn-reset").onclick = resetParams;
  $("modal-close").onclick = () => $("modal").style.display = "none";
  $("lightbox-close").onclick = () => $("lightbox").style.display = "none";
  $("lightbox").onclick = (e) => { if (e.target === $("lightbox")) $("lightbox").style.display = "none"; };
}

async function onProductChange() {
  const pid = $("product-select").value;
  if (!pid) return;
  State.currentProduct = pid;

  // Clear template cache on server
  await fetch(`${API}/api/products/${encodeURIComponent(pid)}/clear-cache`, { method: "POST" });

  // Load params
  try {
    const r = await fetch(`${API}/api/products/${encodeURIComponent(pid)}/params`);
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    State.params = data.params;
    // Populate side select
    const ss = $("side-select");
    ss.innerHTML = '<option value="">选择侧面...</option>';
    if (data.sides) {
      for (const s of data.sides) {
        const o = el("option", null, s);
        o.value = s;
        ss.appendChild(o);
      }
    }
    // Rebuild panel with loaded params
    buildParamPanel();
    // Auto-run if image is loaded
    if (State.image) scheduleRerun();
  } catch (e) {
    console.error("Failed to load params", e);
    alert("加载参数失败: " + e.message);
  }
}

function onDetectionTypeChange() {
  State.detectionType = $("detection-type").value;
  $("side-select").style.display = State.detectionType === "side" ? "" : "none";
  if (State.detectionType === "side") {
    $("side-select").onchange = () => {
      State.sideName = $("side-select").value;
      if (State.image) scheduleRerun();
    };
  }
  if (State.image) scheduleRerun();
}

function onFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    State.image = reader.result; // data:image/...;base64,...
    State.imageName = file.name;
    setStatus("图片已加载", "status-idle");
    if (State.currentProduct) scheduleRerun();
  };
  reader.readAsDataURL(file);
}

async function onTestImageSelect() {
  const path = $("test-image-select").value;
  if (!path) return;
  setStatus("加载测试图...", "status-running");
  try {
    const r = await fetch(`${API}/api/test-images`);
    const images = await r.json();
    const imgInfo = images.find(i => i.path === path);
    if (!imgInfo) return;

    // Read the image file via a separate API endpoint
    // We need to read the file from the server
    const resp = await fetch(`${API}/api/load-image?path=${encodeURIComponent(path)}`);
    if (!resp.ok) throw new Error("Failed to load image");
    const blob = await resp.blob();
    const reader = new FileReader();
    reader.onload = () => {
      State.image = reader.result;
      State.imageName = imgInfo.name;
      // Auto-switch detection type based on image type
      if (imgInfo.type === "side") {
        $("detection-type").value = "side";
        State.detectionType = "side";
        $("side-select").style.display = "";
        // Try to match side name
        if (imgInfo.side) {
          const sideMap = { "l00": "left_l00", "l01": "left_l01" };
          const mapped = sideMap[imgInfo.side.toLowerCase()];
          if (mapped) {
            $("side-select").value = mapped;
            State.sideName = mapped;
          }
        }
      } else {
        $("detection-type").value = "main";
        State.detectionType = "main";
        $("side-select").style.display = "none";
      }
      setStatus("图片已加载", "status-idle");
      if (State.currentProduct) scheduleRerun();
    };
    reader.readAsDataURL(blob);
  } catch (e) {
    console.error("Failed to load test image", e);
    setStatus("加载失败", "status-ng");
  }
}

// ---- Run detection ----
function scheduleRerun() {
  if (!State.currentProduct || !State.image) return;
  clearTimeout(State.rerunTimer);
  State.rerunTimer = setTimeout(runDetection, 500);
}

async function runDetection() {
  if (!State.currentProduct) { alert("请先选择产品"); return; }
  if (!State.image) { alert("请先选择或上传图片"); return; }

  setStatus("检测中...", "status-running");
  $("btn-run").disabled = true;

  try {
    const body = {
      product_id: State.currentProduct,
      image: State.image,
      params: State.params,
      detection_type: State.detectionType,
    };
    if (State.detectionType === "side" && State.sideName) {
      body.side_name = State.sideName;
    }

    const r = await fetch(`${API}/api/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await r.json();

    if (result.error) {
      setStatus("检测失败", "status-ng");
      alert("检测失败: " + result.error);
      return;
    }

    renderPipeline(result);
    const overall = result.result?.overall || "OK";
    if (overall === "NG") {
      setStatus(`NG - ${result.result.ng_reasons?.join("; ") || ""}`, "status-ng");
    } else {
      setStatus(`OK (${result.result?.time_total_ms || 0}ms)`, "status-ok");
    }
  } catch (e) {
    console.error("Detection failed", e);
    setStatus("检测异常", "status-ng");
    alert("检测异常: " + e.message);
  } finally {
    $("btn-run").disabled = false;
  }
}

// ---- Render pipeline steps ----
function renderPipeline(result) {
  const view = $("pipeline-view");
  view.innerHTML = "";

  for (const step of result.steps) {
    const card = el("div", "step-card");

    // Header
    const header = el("div", "step-card-header");
    const title = el("span", "step-title", step.title);
    header.appendChild(title);
    if (step.status !== "ok") {
      const badge = el("span", `step-status-badge ${step.status}`, step.status.toUpperCase());
      header.appendChild(badge);
    }
    card.appendChild(header);

    // Body
    const body = el("div", "step-body");
    if (step.image) {
      const imgContainer = el("div", "step-image-container");
      const img = el("img", "step-image");
      img.src = step.image;
      img.onclick = () => showLightbox(step.image);
      imgContainer.appendChild(img);
      body.appendChild(imgContainer);
    }

    // Metrics
    const metrics = el("div", "step-metrics");
    if (step.metrics) {
      for (const [key, val] of Object.entries(step.metrics)) {
        const mr = el("div", "metric-row");
        const ml = el("span", "metric-label", key);
        let valStr = val;
        if (typeof val === "boolean") valStr = val ? "True" : "False";
        if (typeof val === "number") valStr = val.toString();
        if (Array.isArray(val)) valStr = val.join(", ");
        const mv = el("span", "metric-value", String(valStr));
        if (key === "is_ng" && val === true) mv.classList.add("ng");
        if (key === "is_ng" && val === false) mv.classList.add("ok");
        if (key === "overall" && val === "NG") mv.classList.add("ng");
        if (key === "overall" && val === "OK") mv.classList.add("ok");
        mr.appendChild(ml);
        mr.appendChild(mv);
        metrics.appendChild(mr);
      }
    }
    body.appendChild(metrics);

    card.appendChild(body);
    view.appendChild(card);
  }
}

// ---- Save params ----
async function saveParams() {
  if (!State.currentProduct) { alert("请先选择产品"); return; }
  setStatus("保存中...", "status-running");
  try {
    const r = await fetch(`${API}/api/products/${encodeURIComponent(State.currentProduct)}/params`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: State.params }),
    });
    const data = await r.json();
    if (data.success) {
      setStatus("参数已保存", "status-ok");
    } else {
      setStatus("保存失败", "status-ng");
      alert("保存失败");
    }
  } catch (e) {
    setStatus("保存异常", "status-ng");
    alert("保存异常: " + e.message);
  }
}

// ---- Reset params ----
function resetParams() {
  if (!State.schema) return;
  for (const p of State.schema.params) {
    State.params[p.name] = p.default;
  }
  buildParamPanel();
  if (State.image && State.currentProduct) scheduleRerun();
}

// ---- Lightbox ----
function showLightbox(src) {
  $("lightbox-img").src = src;
  $("lightbox").style.display = "flex";
}

// ---- Start ----
init();
