// DOM
const els = {
  model: document.getElementById("model"),
  presetSelect: document.getElementById("presetSelect"),
  loadPresetBtn: document.getElementById("loadPresetBtn"),
  formArea: document.getElementById("formArea"),
  sendBtn: document.getElementById("sendBtn"),
  resetBtn: document.getElementById("resetBtn"),
  healthBtn: document.getElementById("healthBtn"),
  output: document.getElementById("output"),
  pretty: document.getElementById("prettyOutput"),
  copyCurlBtn: document.getElementById("copyCurlBtn"),
  copyToast: document.getElementById("copyToast"),
  notice: document.getElementById("notice"),
};

const API_BASE = ""; // same-origin
const state = { schema:null, currentModel:null, fields:[], samples:{}, clusterLabels:null };

// Categorical options
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
  await loadSchema();
  buildModelOptions();

  // listeners
  els.model.addEventListener("change", onModelChange);
  els.loadPresetBtn.addEventListener("click", loadSelectedPreset);
  els.sendBtn.addEventListener("click", sendPrediction);
  els.resetBtn.addEventListener("click", resetForm);
  els.healthBtn.addEventListener("click", checkHealth);
  els.copyCurlBtn.addEventListener("click", copyAsCurl);

  // initial
  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  buildPresetOptions();
  loadSelectedPreset();
}

async function loadSchema(){
  const res = await fetch(`${API_BASE}/schema`);
  const data = await res.json();
  state.schema = data;
  // read cluster labels if provided
  const lbl = data?.models?.kmeans_clusters?.cluster_labels || null;
  state.clusterLabels = lbl;
  deriveSamples();
}

function buildModelOptions(){
  const models = state.schema?.models || {};
  const order = ["logreg_churn","decision_tree_churn","svm_churn","kmeans_clusters","linreg_sales"];
  els.model.innerHTML = "";
  order.forEach(k=>{
    if(!models[k]) return;
    const opt = document.createElement("option");
    opt.value = models[k].model_type;
    opt.textContent = prettyName(models[k].model_type);
    els.model.appendChild(opt);
  });
}

function onModelChange(){
  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  buildPresetOptions();
  // don’t auto-overwrite user inputs unless they load a sample
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
    const isNum = ["price","quantity","total_value","age","tenure_months"].includes(name);
    const wrap = document.createElement("div");
    wrap.className = "field";

    if(!isNum && Array.isArray(CATS[name])){
      wrap.innerHTML = `
        <label class="label">${name}</label>
        <select class="input" data-key="${name}">
          ${CATS[name].map(v=>`<option value="${v}">${v}</option>`).join("")}
        </select>
      `;
    } else {
      const step = name === "quantity" ? "1" : "any";
      const min = name === "quantity" ? "0" : "";
      wrap.innerHTML = `
        <label class="label">${name}</label>
        <input class="input" data-key="${name}" type="${isNum?'number':'text'}" ${isNum?`step="${step}" ${min?`min="${min}"`:""}`:""} placeholder="${isNum?'0':name}">
      `;
    }
    frag.appendChild(wrap);
  });

  els.formArea.appendChild(frag);
}

function formToObject(){
  const obj = {};
  for(const name of state.fields){
    const el = els.formArea.querySelector(`[data-key="${name}"]`);
    if(!el) continue;
    const isNum = el.tagName==="INPUT" && el.type==="number";
    const val = el.value;
    obj[name] = isNum && val!=="" ? Number(val) : val;
  }
  return obj;
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
      el.value = CATS[name]?.includes(v) ? v : (CATS[name]?.[0] || "");
    }else{
      el.value = (typeof v === "number") ? v : (v ?? "");
    }
  });
  setNotice("Sample loaded. You can edit any field before predicting.");
}

function deriveSamples(){
  const m = state.schema?.models; if(!m) return;
  state.samples = {
    logreg: [
      {name:"Corporate Electronics", data:m.logreg_churn.example_payload.features},
      {name:"Small Biz Furniture", data:{
        price:12000,quantity:2,total_value:24000,age:40,tenure_months:37,
        gender:"Female",region:"West",segment:"Small Business",product_name:"Desk",category:"Furniture",sentiment:"Negative"
      }},
    ],
    dtree: [
      {name:"Default Tree Sample", data:m.decision_tree_churn.example_payload.features},
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

function prettyName(kind){
  return {
    logreg:"Churn — Logistic Regression",
    dtree: "Churn — Decision Tree",
    svm:   "Churn — SVM",
    kmeans:"Clustering — KMeans",
    linreg:"Sales — Linear Regression"
  }[kind] || kind;
}

function setNotice(msg){
  els.notice.hidden = !msg;
  els.notice.textContent = msg || "";
}

function toggleLoading(btn, on){
  if(on){ btn.disabled = true; btn.classList.add("is-loading"); }
  else { btn.disabled = false; btn.classList.remove("is-loading"); }
}

/* ======= Actions ======= */
async function sendPrediction(){
  const payload = { model_type: els.model.value, features: formToObject() };
  delete payload.features.churn; // safety

  // Basic validation for numeric fields
  const mustBeNum = ["price","quantity","total_value","age","tenure_months"];
  const missing = mustBeNum.filter(k => state.currentModel !== "linreg" ? payload.features[k] === "" || payload.features[k] == null
                                       : ["price","quantity","age","tenure_months"].includes(k) && (payload.features[k] === "" || payload.features[k] == null));

  if(missing.length){
    setNotice(`Please fill numeric fields: ${missing.join(", ")}`);
    return;
  } else {
    setNotice("");
  }

  toggleLoading(els.sendBtn, true);
  try{
    const res = await fetch(`${API_BASE}/predict`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
    renderPretty(data);
  }catch(e){
    els.output.textContent = "Request failed: " + e.message;
    els.pretty.innerHTML = "";
  }finally{
    toggleLoading(els.sendBtn, false);
  }
}

async function checkHealth(){
  toggleLoading(els.healthBtn, true);
  try{
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
    els.pretty.innerHTML = `<div class="card-sm kv"><div>Health</div><div class="badge ok">OK</div></div>`;
  }catch(e){
    els.output.textContent = "Health check failed: " + e.message;
    els.pretty.innerHTML = `<div class="card-sm kv"><div>Health</div><div class="badge bad">DOWN</div></div>`;
  }finally{
    toggleLoading(els.healthBtn, false);
  }
}

async function copyAsCurl(){
  const body = JSON.stringify({ model_type: els.model.value, features: formToObject() });
  const curl = `curl -X POST /predict -H "Content-Type: application/json" -d '${body.replaceAll("'", "'\\''")}'`;
  await navigator.clipboard.writeText(curl);
  els.copyToast.hidden = false; setTimeout(()=>els.copyToast.hidden=true, 1200);
}

/* ======= Pretty results ======= */
function renderPretty(resp){
  const kind = resp?.model_type;
  if(!kind){ els.pretty.innerHTML = ""; return; }

  if(kind==="kmeans"){
    const id = Number(resp.prediction_cluster);
    // prefer schema labels, fallback to simple mapping
    const labels = state.clusterLabels || {"0":"Low Spenders","1":"Mid Spenders","2":"High Spenders"};
    const label = labels[String(id)] || `Cluster ${id}`;
    els.pretty.innerHTML = `
      <div class="card-sm">
        <div class="kv"><div><strong>Segment</strong></div><div class="badge info">${label}</div></div>
        <div class="kv"><div>Cluster ID</div><div>#${id}</div></div>
      </div>
    `;
    return;
  }

  if(kind==="linreg"){
    const v = Number(resp.predicted_sales_value || 0);
    els.pretty.innerHTML = `
      <div class="card-sm">
        <div class="kv"><div><strong>Predicted Sales</strong></div><div class="badge ok">${fmtCurrency(v)}</div></div>
      </div>
    `;
    return;
  }

  // Classifiers: logreg, dtree, svm
  const pred = Number(resp.prediction);
  const proba = typeof resp.probability_of_churn === "number" ? clamp01(resp.probability_of_churn) : null;
  els.pretty.innerHTML = `
    <div class="card-sm">
      <div class="kv"><div><strong>Prediction</strong></div><div class="${pred===1?"badge bad":"badge ok"}">${pred===1?"Churn":"Stay"}</div></div>
      ${proba!=null ? `
        <div style="margin-top:8px;font-size:.9rem;color:#6b7280">Churn probability: ${(proba*100).toFixed(1)}%</div>
        <div class="bar"><span style="width:${(proba*100).toFixed(0)}%"></span></div>
      ` : `<div style="margin-top:8px;font-size:.9rem;color:#6b7280">Probability not available for this model.</div>`}
    </div>
  `;
}

/* ======= Utils ======= */
function clamp01(x){ if(isNaN(x)) return 0; return Math.max(0, Math.min(1, Number(x))); }
function fmtCurrency(n){
  try{ return new Intl.NumberFormat(undefined, {style:"currency",currency:"USD",maximumFractionDigits:0}).format(n); }
  catch{ return "$"+Math.round(n).toLocaleString(); }
}
function resetForm(){
  rebuildForm(state.currentModel);
  buildPresetOptions();
  els.presetSelect.value = 0;
  els.output.textContent = "—";
  els.pretty.innerHTML = "";
  setNotice("");
}
