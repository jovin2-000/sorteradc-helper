// Flow builder: compose custom operator flows

const Builder = {
  allOps: [],
  selectedOps: [],
  initialized: false,

  async init() {
    if (this.initialized) return;
    try {
      const r = await fetch('/api/operators');
      this.allOps = await r.json();
      this.initialized = true;
    } catch(e) { console.error('Failed to load operators', e); }
  },

  show() {
    const overlay = $('flow-builder-overlay');
    overlay.style.display = 'flex';
    this.render();
  },

  hide() {
    $('flow-builder-overlay').style.display = 'none';
  },

  render() {
    // Group available operators
    const byGroup = {};
    for (const op of this.allOps) {
      if (!byGroup[op.group]) byGroup[op.group] = [];
      byGroup[op.group].push(op);
    }

    const availEl = $('fb-available');
    availEl.innerHTML = '';
    for (const [group, opsList] of Object.entries(byGroup)) {
      const groupEl = el('div', 'fb-group');
      groupEl.appendChild(el('div', 'fb-group-title', group));
      for (const op of opsList) {
        const item = el('div', 'fb-op-item');
        const isInSelected = this.selectedOps.some(s => s.name === op.name);
        if (isInSelected) item.classList.add('selected');
        item.innerHTML = `<span>${op.title}</span><span class="fb-op-params">${(op.param_keys||[]).length}p</span>`;
        item.onclick = () => {
          if (isInSelected) {
            this.selectedOps = this.selectedOps.filter(s => s.name !== op.name);
          } else {
            this.selectedOps.push(op);
          }
          this.render();
        };
        groupEl.appendChild(item);
      }
      availEl.appendChild(groupEl);
    }

    // Selected operators
    const selEl = $('fb-selected');
    selEl.innerHTML = '';
    if (this.selectedOps.length === 0) {
      selEl.innerHTML = '<div class="fb-empty">点击左侧算子添加到流程</div>';
    } else {
      this.selectedOps.forEach((op, i) => {
        const item = el('div', 'fb-sel-item');
        item.innerHTML = `<span class="fb-sel-num">${i+1}</span><span class="fb-sel-title">${op.title}</span>`;
        const btns = el('div', 'fb-sel-btns');
        if (i > 0) {
          const up = el('button', 'btn btn-sm', '↑');
          up.onclick = () => { [this.selectedOps[i-1], this.selectedOps[i]] = [this.selectedOps[i], this.selectedOps[i-1]]; this.render(); };
          btns.appendChild(up);
        }
        if (i < this.selectedOps.length - 1) {
          const down = el('button', 'btn btn-sm', '↓');
          down.onclick = () => { [this.selectedOps[i+1], this.selectedOps[i]] = [this.selectedOps[i], this.selectedOps[i+1]]; this.render(); };
          btns.appendChild(down);
        }
        const rm = el('button', 'btn btn-sm', '✕');
        rm.onclick = () => { this.selectedOps.splice(i, 1); this.render(); };
        btns.appendChild(rm);
        item.appendChild(btns);
        selEl.appendChild(item);
      });
    }
    $('fb-count').textContent = `${this.selectedOps.length} 个算子`;
  },

  async loadExisting(flowId) {
    try {
      const r = await fetch(`/api/flows/custom`);
      const flows = await r.json();
      const flow = flows.find(f => f.id === flowId);
      if (flow) {
        this.selectedOps = (flow.operators || []).map(name =>
          this.allOps.find(o => o.name === name)).filter(o => o);
        $('fb-name').value = flow.name;
        $('fb-id').value = flow.id;
      }
    } catch(e) {}
  },

  async save() {
    const name = $('fb-name').value.trim();
    if (!name) { alert('请输入流程名称'); return; }
    if (this.selectedOps.length === 0) { alert('请至少选择一个算子'); return; }
    let flowId = $('fb-id').value.trim();
    if (!flowId) flowId = 'custom_' + Date.now();
    const opNames = this.selectedOps.map(o => o.name);
    try {
      const r = await fetch('/api/flows/custom', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: flowId, name: name, description: '', operators: opNames}),
      });
      const result = await r.json();
      if (result.error) { alert(result.error); return; }
      // Reload flows list
      const flowsR = await fetch('/api/flows');
      State.flows = await flowsR.json();
      const sel = $('flow-select');
      sel.innerHTML = '<option value="">选择流程...</option>';
      for (const f of State.flows) {
        const o = el('option', null, `${f.name} (${f.operator_count}步)${f.custom?' *':''}`);
        o.value = f.id; sel.appendChild(o);
      }
      sel.value = flowId;
      this.hide();
      // Auto-run if image is loaded
      State.currentFlow = flowId;
      const isTemplate = false; // custom flows are not template_create
      if (State.image) scheduleRerun();
    } catch(e) { alert('保存失败: ' + e.message); }
  },
};

// Add flow builder button to topbar
function addFlowBuilderButton() {
  const btn = el('button', 'btn btn-sm', '编辑流程');
  btn.id = 'btn-flow-builder';
  btn.onclick = async () => { await Builder.init(); Builder.show(); };
  document.querySelector('.topbar-right').insertBefore(btn, $('btn-save'));
}

// Call after init
setTimeout(addFlowBuilderButton, 1000);

// Bind close/save buttons
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    const closeBtn = document.getElementById('fb-close');
    if (closeBtn) closeBtn.onclick = () => Builder.hide();
    const saveBtn = document.getElementById('fb-save');
    if (saveBtn) saveBtn.onclick = () => Builder.save();
  }, 1500);
});
