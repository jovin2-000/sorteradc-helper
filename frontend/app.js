// sorteradc-helper - flow-agnostic frontend

const State = {
  flows: [],
  currentFlow: null,
  flowMeta: [],
  schema: null,
  products: [],
  currentProduct: null,
  params: {},
  image: null,
  imageName: null,
  testImages: [],
  rerunTimer: null,
};

const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if(cls) e.className=cls; if(text) e.textContent=text; return e; };
function setStatus(t, c) { const s=$("status-indicator"); s.textContent=t; s.className=c||"status-idle"; }

async function init() {
  try {
    const r = await fetch('/api/flows');
    State.flows = await r.json();
    const sel = $("flow-select");
    sel.innerHTML = '<option value="">选择流程...</option>';
    for (const f of State.flows) {
      const o = el("option", null, `${f.name} (${f.operator_count}步)`);
      o.value = f.id;
      sel.appendChild(o);
    }
  } catch(e) { console.error("flows load failed", e); }

  try {
    const r = await fetch('/api/param-schema');
    State.schema = await r.json();
  } catch(e) { console.error("schema load failed", e); }

  try {
    const r = await fetch('/api/products');
    State.products = await r.json();
    const sel = $("product-select");
    sel.innerHTML = '<option value="">选择产品...</option>';
    for (const p of State.products) {
      const o = el("option", null, `${p.product_id} (${p.state})`);
      o.value = p.product_id;
      sel.appendChild(o);
    }
  } catch(e) { console.error("products load failed", e); }

  try {
    const r = await fetch('/api/test-images');
    State.testImages = await r.json();
    const sel = $("test-image-select");
    sel.innerHTML = '<option value="">或选择测试图...</option>';
    for (const img of State.testImages) {
      const o = el("option", null, img.label);
      o.value = img.path;
      sel.appendChild(o);
    }
  } catch(e) { console.error("test images load failed", e); }

  bindEvents();
}

function bindEvents() {
  $("flow-select").onchange = onFlowChange;
  $("product-select").onchange = onProductChange;
  $("btn-upload").onclick = () => $("file-input").click();
  $("file-input").onchange = onFileUpload;
  $("test-image-select").onchange = onTestImageSelect;
  $("btn-run").onclick = runFlow;
  $("btn-save").onclick = saveParams;
  $("btn-reset").onclick = resetParams;
  $("lightbox-close").onclick = () => $("lightbox").style.display="none";
  $("lightbox").onclick = e => { if(e.target===$("lightbox")) $("lightbox").style.display="none"; };
}

async function onFlowChange() {
  const fid = $("flow-select").value;
  if (!fid) return;
  State.currentFlow = fid;

  // Show/hide product selector based on flow type
  const isTemplateFlow = fid === "template_create";
  $("product-select").style.display = isTemplateFlow ? "none" : "";

  // Load flow meta
  try {
    const r = await fetch(`/api/flows/${fid}/meta`);
    const data = await r.json();
    State.flowMeta = data.operators || [];
    // Show which params this flow uses
    const usedParams = new Set();
    for (const op of State.flowMeta) {
      for (const pk of (op.param_keys||[])) usedParams.add(pk);
    }
    State._usedParams = usedParams;
  } catch(e) { console.error("flow meta failed", e); }

  // Load params from product if selected, or defaults
  if (State.currentProduct && !isTemplateFlow) {
    await loadProductParams(State.currentProduct);
  } else {
    State.params = getDefaultsFromSchema();
    buildParamPanel();
  }
  if (State.image) scheduleRerun();
}

function getDefaultsFromSchema() {
  if (!State.schema) return {};
  const d = {};
  for (const p of State.schema.params) d[p.name] = p.default;
  return d;
}

async function onProductChange() {
  const pid = $("product-select").value;
  if (!pid) return;
  State.currentProduct = pid;
  await loadProductParams(pid);
  if (State.image && State.currentFlow) scheduleRerun();
}

async function loadProductParams(pid) {
  try {
    const r = await fetch(`/api/products/${encodeURIComponent(pid)}/params`);
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    State.params = data.params;
    buildParamPanel();
  } catch(e) { alert("加载参数失败: " + e.message); }
}

function onFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    State.image = reader.result;
    State.imageName = file.name;
    setStatus("图片已加载", "status-idle");
    if (State.currentFlow) scheduleRerun();
  };
  reader.readAsDataURL(file);
}

async function onTestImageSelect() {
  const path = $("test-image-select").value;
  if (!path) return;
  setStatus("加载测试图...", "status-running");
  try {
    const resp = await fetch(`/api/load-image?path=${encodeURIComponent(path)}`);
    if (!resp.ok) throw new Error("Failed");
    const blob = await resp.blob();
    const reader = new FileReader();
    reader.onload = () => {
      State.image = reader.result;
      State.imageName = path.split("/").pop();
      setStatus("图片已加载", "status-idle");
      if (State.currentFlow) scheduleRerun();
    };
    reader.readAsDataURL(blob);
  } catch(e) { setStatus("加载失败", "status-ng"); }
}

function scheduleRerun() {
  if (!State.currentFlow || !State.image) return;
  clearTimeout(State.rerunTimer);
  State.rerunTimer = setTimeout(runFlow, 600);
}

async function runFlow() {
  if (!State.currentFlow) { alert("请先选择流程"); return; }
  if (!State.image) { alert("请先选择或上传图片"); return; }

  setStatus("运行中...", "status-running");
  $("btn-run").disabled = true;

  try {
    const body = {
      image: State.image,
      params: State.params,
    };
    if (State.currentProduct) body.product_id = State.currentProduct;

    const r = await fetch(`/api/flows/${State.currentFlow}/run`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const result = await r.json();
    if (result.error) {
      setStatus("运行失败", "status-ng");
      alert("运行失败: " + result.error);
      return;
    }
    renderPipeline(result);
    const overall = result.overall || "OK";
    if (overall === "NG") {
      setStatus(`NG - ${(result.ng_reasons||[]).join("; ")}`, "status-ng");
    } else {
      setStatus("OK", "status-ok");
    }
  } catch(e) {
    setStatus("运行异常", "status-ng");
    alert("运行异常: " + e.message);
  } finally {
    $("btn-run").disabled = false;
  }
}

function renderPipeline(result) {
  const view = $("pipeline-view");
  view.innerHTML = "";
  let lastGroup = "";
  for (const step of (result.steps||[])) {
    // Group separator
    if (step.group !== lastGroup) {
      lastGroup = step.group;
      const sep = el("div", "group-sep", step.group);
      view.appendChild(sep);
    }
    view.appendChild(buildStepCard(step));
  }
}

function buildStepCard(step) {
  const card = el("div", "step-card");
  if (step.status === "skip") card.classList.add("card-skip");

  const header = el("div", "step-card-header");
  header.appendChild(el("span", "step-title", step.title));
  if (step.status !== "ok") {
    header.appendChild(el("span", `step-status-badge ${step.status}`, step.status.toUpperCase()));
  }
  card.appendChild(header);

  const body = el("div", "step-body");
  if (step.image) {
    const imgC = el("div", "step-image-container");
    const img = el("img", "step-image");
    img.src = step.image;
    img.onclick = () => showLightbox(step.image);
    imgC.appendChild(img);
    body.appendChild(imgC);
  }

  const metrics = el("div", "step-metrics");
  if (step.metrics) {
    for (const [k,v] of Object.entries(step.metrics)) {
      if (k === "op_index" || k === "time_ms") continue;
      const mr = el("div", "metric-row");
      mr.appendChild(el("span", "metric-label", k));
      let vs = v; if(typeof v==="boolean") vs=v?"True":"False";
      if(Array.isArray(v)) vs=v.join(", ");
      const mv = el("span", "metric-value", String(vs));
      if(k==="is_ng"&&v===true) mv.classList.add("ng");
      if(k==="is_ng"&&v===false) mv.classList.add("ok");
      if(k==="overall"&&v==="NG") mv.classList.add("ng");
      if(k==="overall"&&v==="OK") mv.classList.add("ok");
      mr.appendChild(mv);
      metrics.appendChild(mr);
    }
    // Time at bottom
    if (step.metrics.time_ms) {
      const tr = el("div", "metric-row");
      tr.appendChild(el("span", "metric-label", "time"));
      tr.appendChild(el("span", "metric-value", `${step.metrics.time_ms}ms`));
      metrics.appendChild(tr);
    }
  }
  body.appendChild(metrics);
  card.appendChild(body);
  return card;
}

// ---- Parameter panel ----
function buildParamPanel() {
  const panel = $("param-panel");
  panel.innerHTML = "";
  if (!State.schema) return;

  const usedParams = State._usedParams || null; // null = show all
  const byCat = {};
  for (const p of State.schema.params) {
    if (usedParams && !usedParams.has(p.name)) continue;
    if (!byCat[p.category]) byCat[p.category] = [];
    byCat[p.category].push(p);
  }

  for (const cat of State.schema.categories) {
    const params = byCat[cat.id];
    if (!params || !params.length) continue;
    const section = el("div", "param-category");
    const header = el("div", "param-category-header");
    header.innerHTML = `<span>${cat.title}</span><span class="chevron">&#9660;</span>`;
    header.onclick = () => section.classList.toggle("collapsed");
    const body = el("div", "param-category-body");
    for (const p of params) body.appendChild(buildParamRow(p));
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

  switch(p.type) {
    case "bool": {
      const t = el("div", "param-toggle");
      if(State.params[p.name]) t.classList.add("on");
      t.onclick = () => { State.params[p.name]=!State.params[p.name]; t.classList.toggle("on"); scheduleRerun(); };
      control.appendChild(t); break;
    }
    case "float": case "int": {
      const sl = el("input","param-slider"); sl.type="range"; sl.min=p.min; sl.max=p.max; sl.step=p.step; sl.value=State.params[p.name]??p.default;
      const ip = el("input","param-input"); ip.type="number"; ip.min=p.min; ip.max=p.max; ip.step=p.step; ip.value=State.params[p.name]??p.default;
      sl.oninput = () => { State.params[p.name]=parseFloat(sl.value); ip.value=sl.value; scheduleRerun(); };
      ip.onchange = () => { let v=parseFloat(ip.value)||p.default; v=Math.max(p.min,Math.min(p.max,v)); State.params[p.name]=v; sl.value=v; ip.value=v; scheduleRerun(); };
      control.appendChild(sl); control.appendChild(ip); break;
    }
    case "select": {
      const s = el("select","param-select");
      for(const o of p.options){ const op=el("option",null,o); if(o===State.params[p.name]) op.selected=true; s.appendChild(op); }
      s.onchange = () => { State.params[p.name]=s.value; scheduleRerun(); };
      control.appendChild(s); break;
    }
    case "multiselect": {
      const chips = el("div","multiselect-row");
      for(const o of p.options){ const c=el("span","multiselect-chip",o); if(State.params[p.name]?.includes(o)) c.classList.add("active");
        c.onclick=()=>{ let a=State.params[p.name]||[]; a=a.includes(o)?a.filter(x=>x!==o):[...a,o]; State.params[p.name]=a; c.classList.toggle("active"); scheduleRerun(); };
        chips.appendChild(c); }
      control.appendChild(chips); break;
    }
    case "weights": {
      const w = State.params[p.name]||p.default;
      for(const o of p.options){ const wr=el("div","weight-row"); wr.appendChild(el("span","weight-label",o));
        const wi=el("input","weight-input"); wi.type="number"; wi.step="0.05"; wi.min="0"; wi.max="1"; wi.value=w[o]??0;
        wi.onchange=()=>{ let v=parseFloat(wi.value)||0; v=Math.max(0,Math.min(1,v)); if(!State.params[p.name])State.params[p.name]={}; State.params[p.name][o]=v; wi.value=v; scheduleRerun(); };
        wr.appendChild(wi); control.appendChild(wr); }
      break;
    }
    case "tuple_float": {
      const a=State.params[p.name]||p.default;
      const i1=el("input","param-input"); i1.type="number"; i1.step="0.1"; i1.value=a[0]??0;
      const i2=el("input","param-input"); i2.type="number"; i2.step="0.1"; i2.value=a[1]??0;
      const upd=()=>{ State.params[p.name]=[parseFloat(i1.value)||0, parseFloat(i2.value)||0]; scheduleRerun(); };
      i1.onchange=upd; i2.onchange=upd;
      control.appendChild(i1); control.appendChild(i2); break;
    }
  }
  row.appendChild(control);
  return row;
}

async function saveParams() {
  if(!State.currentProduct){ alert("请先选择产品"); return; }
  setStatus("保存中...","status-running");
  try {
    const r = await fetch(`/api/products/${encodeURIComponent(State.currentProduct)}/params`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({params:State.params})});
    const d = await r.json();
    if(d.success) setStatus("参数已保存","status-ok"); else { setStatus("保存失败","status-ng"); alert("保存失败"); }
  } catch(e){ setStatus("保存异常","status-ng"); alert("保存异常: "+e.message); }
}

function resetParams() {
  if(!State.schema) return;
  for(const p of State.schema.params) State.params[p.name]=p.default;
  buildParamPanel();
  if(State.image && State.currentFlow) scheduleRerun();
}

function showLightbox(src){ $("lightbox-img").src=src; $("lightbox").style.display="flex"; }

init();
