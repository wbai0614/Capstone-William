const els = {
  model: document.getElementById("model"),
  presetSelect: document.getElementById("presetSelect"),
  loadPresetBtn: document.getElementById("loadPresetBtn"),
  formArea: document.getElementById("formArea"),
  featuresBox: document.getElementById("featuresBox"),
  output: document.getElementById("output"),
  pretty: document.getElementById("prettyOutput"),
  sendBtn: document.getElementById("sendBtn"),
  resetBtn: document.getElementById("resetBtn"),
  healthBtn: document.getElementById("healthBtn"),
  copyCurlBtn: document.getElementById("copyCurlBtn"),
  copyToast: document.getElementById("copyToast"),
  themeToggle: document.getElementById("themeToggle"),
  notice: document.getElementById("notice"),
};

// Hard-lock to same origin
const API_BASE = ""; // "" + "/predict" => "/predict"

const state = { schema:null, currentModel:null, fields:[], samples:{} };

// Known categorical field options (UX-friendly)
const CATS = {
  gender: ["Female","Male","Other"],
  region: ["North","South","East","West"],
  segment: ["Corporate","Small Business","Home Office"],
  category: ["Electronics","Furniture","Office Supplies"],
  product_name: ["Projector","Desk","Chair","Printer","Monitor","Mouse","Keyboard"],
  sentiment: ["Positive","Neutral","Negative"]
};

init();

async function init(){
  // Theme
  const saved = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);
  els.themeToggle.checked = saved === "dark";
  els.themeToggle.addEventListener("change", () => {
    const t = els.themeToggle.checked ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("theme", t);
  });

  await loadSchema();
  buildModelOptions();
  els.model.addEventListener("change", onModelChange);
  els.loadPresetBtn.addEventListener("click", loadSelectedPreset);
  els.sendBtn.addEventListener("click", sendPrediction);
  els.resetBtn.addEventListener("click", resetForm);
  els.healthBtn.addEventListener("click", checkHealth);
  els.copyCurlBtn.addEventListener("click", copyAsCurl);

  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  buildPresetOptions();
  loadSelectedPreset(); // load a sample by default
}

async function loadSchema(){
  try{
    const res = await fetch(`${API_BASE}/schema`);
    state.schema = await res.json();
    deriveSamples();
  }catch{ toast("Couldn’t load /schema"); }
}

function buildModelOptions(){
  const models = state.schema?.models || {};
  const order = ["logreg_churn","decision_tree_churn","svm_churn","kmeans_clusters","linreg_sales"];
  els.model.innerHTML = "";
  order.forEach(k=>{
    if(!models[k]) return;
    const mt = models[k].model_type;
    const opt = document.createElement("option");
    opt.value = mt; opt.textContent = pretty(mt);
    els.model.appendChild(opt);
  });
}

function onModelChange(){
  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  buildPresetOptions();
  loadSelectedPreset();
}

function rebuildForm(kind){
  const m = state.schema?.models; if(!m) return;
  let fields = [];
  if(kind==="kmeans") fields = m.kmeans_clusters.required_numeric_fields||[];
  else if(kind==="linreg") fields = m.linreg_sales.required_fields||[];
  else if(kind==="logreg") fields = m.logreg_churn.required_fields||[];
  else if(kind==="dtree") fields = m.decision_tree_churn.required_fields||[];
  else if(kind==="svm") fields = m.svm_churn.required_fields||[];
  state.fields = fields;

  els.formArea.innerHTML = "";
  const frag = document.createDocumentFragment();

  fields.forEach(name=>{
    const wrap = document.createElement("div");
    wrap.className = "field";
    const isNum = ["price","quantity","total_value","age","tenure_months"].includes(name);
    if(!isNum && Array.isArray(CATS[name])){
      wrap.innerHTML = `
        <label>${name}</label>
        <select data-key="${name}">
          ${CATS[name].map(o=>`<option value="${o}">${o}</option>`).join("")}
        </select>
      `;
    } else {
      wrap.innerHTML = `
        <label>${name}</label>
        <input data-key="${name}" type="${isNum?"number":"text"}" placeholder="${isNum?"0":name}">
      `;
    }
    frag.appendChild(wrap);
  });

  els.formArea.appendChild(frag);
  els.featuresBox.value = JSON.stringify(formToObject(), null, 2);
}

function buildPresetOptions(){
  const list = presetsFor(state.currentModel);
  els.presetSelect.innerHTML = "";
  list.forEach((p,i)=>{
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = p.name;
    els.presetSelect.appendChild(opt);
  });
}

function loadSelectedPreset(){
  const idx = Number(els.presetSelect.value || 0);
  const sample = presetsFor(state.currentModel)[idx]?.data || {};
  state.fields.forEach(name=>{
    const el = els.formArea.querySelector(`[data-key="${name}"]`);
    if(!el) return;
    const v = sample[name];
    if(el.tagName==="SELECT"){
      if(CATS[name]?.includes(v)) el.value = v;
      else el.value = CATS[name]?.[0] || "";
    }else{
      el.value = (typeof v === "number") ? v : (v ?? "");
    }
  });
  els.featuresBox.value = JSON.stringify(formToObject(), null, 2);
  setNotice("");
}

function formToObject(){
  const obj = {};
  for(const name of state.fields){
    const sel = els.formArea.querySelector(`[data-key="${name}"]`);
    if(!sel) continue;
    const isNum = sel.tagName==="INPUT" && sel.type==="number";
    const val = sel.value;
    obj[name] = isNum && val!=="" ? Number(val) : val;
  }
  return obj;
}

function deriveSamples(){
  const m = state.schema?.models; if(!m) return;
  state.samples = {
    logreg: [
      {name:"Corporate Elect. (example)", data:m.logreg_churn.example_payload.features},
      {name:"Small Biz Furniture", data:{
        price:12000,quantity:2,total_value:24000,age:40,tenure_months:37,
        gender:"Female",region:"West",segment:"Small Business",product_name:"Desk",category:"Furniture",sentiment:"Negative"
      }},
    ],
    dtree: [
      {name:"Default DT sample", data:m.decision_tree_churn.example_payload.features},
      {name:"Young, short tenure", data:{
        price:8000,quantity:1,total_value:8000,age:26,tenure_months:6,gender:"Male",
        region:"East",segment:"Home Office",product_name:"Chair",category:"Furniture",sentiment:"Neutral"
      }},
    ],
    svm: [
      {name:"Corporate High Value", data:m.svm_churn.example_payload.features},
      {name:"West, Low Value", data:{
        price:3000,quantity:1,total_value:3000,age:52,tenure_months:60,gender:"Female",
        region:"West",segment:"Corporate",product_name:"Mouse",category:"Electronics",sentiment:"Positive"
      }},
    ],
    kmeans: [
      {name:"Mid Spender", data:m.kmeans_clusters.example_payload_dict.features},
      {name:"High Spender", data:{price:70000,quantity:3,total_value:210000,age:45,tenure_months:80}}
    ],
    linreg: [
      {name:"Projector Corporate", data:m.linreg_sales.example_payload.features},
      {name:"Desk Small Biz", data:{
        price:12000,quantity:2,age:40,tenure_months:37,gender:"Female",
        region:"West",segment:"Small Business",product_name:"Desk",category:"Furniture",sentiment:"Negative"
      }},
    ]
  };
}

function presetsFor(kind){ return state.samples[kind] || []; }

function pretty(kind){
  return {logreg:"Churn — Logistic Regression",dtree:"Churn — Decision Tree",svm:"Churn — SVM",kmeans:"Clustering — KMeans",linreg:"Sales — Linear Regression"}[kind] || kind;
}

function setNotice(msg){
  els.notice.hidden = !msg;
  els.notice.textContent = msg || "";
}

function toggleLoading(b){
  [els.sendBtn, els.healthBtn].forEach(btn=>{
    btn.disabled = b; btn.classList.toggle("loading", b);
  });
}

async function sendPrediction(){
  // Prefer form values, but allow JSON textarea; sanitize to allowed fields only.
  let fromJson = {};
  try{ if(els.featuresBox.value.trim()) fromJson = JSON.parse(els.featuresBox.value); }catch{}
  const formObj = formToObject();

  const allowed = new Set(state.fields);
  const sanitized = {};
  const extraKeys = [];

  const candidate = {...formObj, ...fromJson};
  Object.keys(candidate).forEach(k=>{
    if(allowed.has(k)) sanitized[k] = candidate[k];
    else extraKeys.push(k);
  });

  // Never send churn
  delete sanitized.churn;

  if(extraKeys.length){
    const filtered = extraKeys.filter(k=>k!=="churn");
    if(filtered.length) setNotice(`Ignored unknown field(s): ${filtered.join(", ")}`);
    else setNotice("");
  } else setNotice("");

  const payload = { model_type: els.model.value, features: sanitized };

  toggleLoading(true);
  try{
    const res = await fetch(`${API_BASE}/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
    renderPretty(data);
  }catch(e){
    els.output.textContent = "Request failed: " + e.message;
    els.pretty.innerHTML = "";
  }finally{ toggleLoading(false); }
}

async function checkHealth(){
  toggleLoading(true);
  try{
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
    els.pretty.innerHTML = `<div class="card-sm kv"><div>Health</div><div class="badge ok">OK</div></div>`;
  }catch(e){
    els.output.textContent = "Health check failed: " + e.message;
    els.pretty.innerHTML = `<div class="card-sm kv"><div>Health</div><div class="badge bad">DOWN</div></div>`;
  }finally{ toggleLoading(false); }
}

function renderPretty(resp){
  const kind = resp?.model_type;
  if(!kind){ els.pretty.innerHTML = ""; return; }

  if(kind==="kmeans"){
    const cluster = resp.prediction_cluster;
    els.pretty.innerHTML = `
      <div class="card-sm">
        <div class="kv">
          <div><strong>Cluster</strong></div>
          <div class="badge info">#${cluster}</div>
        </div>
        <div class="small mt-sm">Order: ${resp.order?.join(", ")}</div>
      </div>
    `;
    return;
  }

  if(kind==="linreg"){
    const v = Number(resp.predicted_sales_value || 0);
    els.pretty.innerHTML = `
      <div class="card-sm">
        <div class="kv">
          <div><strong>Predicted Sales</strong></div>
          <div class="badge ok">${fmtCurrency(v)}</div>
        </div>
      </div>
    `;
    return;
  }

  // Classifiers (logreg, dtree, svm)
  const pred = Number(resp.prediction);
  const proba = typeof resp.probability_of_churn === "number" ? resp.probability_of_churn : null;
  els.pretty.innerHTML = `
    <div class="card-sm">
      <div class="kv">
        <div><strong>Prediction</strong></div>
        <div class="${pred===1?"badge bad":"badge ok"}">${pred===1?"Churn":"Stay"}</div>
      </div>
      ${proba!=null ? `
      <div class="mt-sm small">Churn probability: ${(Math.max(0,Math.min(1,proba))*100).toFixed(1)}%</div>
      <div class="bar"><span style="width:${(Math.max(0,Math.min(1,proba))*100).toFixed(0)}%"></span></div>
      ` : `<div class="mt-sm small">Probability not available for this model.</div>`}
    </div>
  `;
}

function fmtCurrency(n){
  try{ return new Intl.NumberFormat(undefined,{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n); }
  catch{ return "$"+Math.round(n).toLocaleString(); }
}

function resetForm(){
  rebuildForm(state.currentModel);
  buildPresetOptions();
  els.presetSelect.value = 0;
  loadSelectedPreset();
  els.output.textContent = "—";
  els.pretty.innerHTML = "";
  setNotice("");
}

async function copyAsCurl(){
  const body = JSON.stringify({ model_type: els.model.value, features: formToObject() });
  const curl = `curl -X POST /predict -H "Content-Type: application/json" -d '${body.replaceAll("'", "'\\''")}'`;
  await navigator.clipboard.writeText(curl);
  els.copyToast.hidden = false; setTimeout(()=>els.copyToast.hidden=true, 1200);
}

function toast(msg){ els.output.textContent = msg; }
