const els = {
  model: document.getElementById("model"),
  baseUrl: document.getElementById("baseUrl"),
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
};

const state = { schema:null, currentModel:null, fields:[], samples:{} };

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
  buildPresetOptions();

  els.model.addEventListener("change", onModelChange);
  els.loadPresetBtn.addEventListener("click", loadSelectedPreset);
  els.sendBtn.addEventListener("click", sendPrediction);
  els.resetBtn.addEventListener("click", resetForm);
  els.healthBtn.addEventListener("click", checkHealth);
  els.copyCurlBtn.addEventListener("click", copyAsCurl);

  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  seedJsonFromForm();
}

function cleanBase(){ return (els.baseUrl.value || "/").replace(/\/$/, ""); }

async function loadSchema(){
  try{
    const res = await fetch(`${cleanBase()}/schema`);
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

function buildPresetOptions(){
  els.presetSelect.innerHTML = "";
  const kind = els.model.value || "logreg";
  const list = presetsFor(kind);
  list.forEach((p,i)=>{
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = p.name;
    els.presetSelect.appendChild(opt);
  });
}

function onModelChange(){
  state.currentModel = els.model.value;
  rebuildForm(state.currentModel);
  buildPresetOptions();
  loadSelectedPreset(); // auto-load first preset
}

function deriveSamples(){
  // Build default samples from example_payloads in schema
  const m = state.schema?.models; if(!m) return;
  state.samples = {
    logreg: [
      {name:"Corporate Elect. (likely stay)", data:m.logreg_churn.example_payload.features},
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

function presetsFor(kind){
  return state.samples[kind] || [];
}

function loadSelectedPreset(){
  const idx = Number(els.presetSelect.value || 0);
  const list = presetsFor(state.currentModel);
  const sample = list[idx]?.data || {};
  // Fill form inputs
  for(const name of state.fields){
    const el = els.formArea.querySelector(`input[data-key="${name}"]`);
    if(!el) continue;
    const v = sample[name];
    el.value = (typeof v === "number") ? v : (v ?? "");
  }
  seedJsonFromForm();
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
    const div = document.createElement("div");
    div.className="field";
    div.innerHTML = `
      <label>${name}</label>
      <input data-key="${name}" type="${isNum?"number":"text"}" placeholder="${isNum?"0":name}">
    `;
    frag.appendChild(div);
  });
  els.formArea.appendChild(frag);
}

function seedJsonFromForm(){
  const obj = {};
  for(const name of state.fields){
    const el = els.formArea.querySelector(`input[data-key="${name}"]`);
    if(!el) continue;
    obj[name] = (el.type==="number" && el.value!=="") ? Number(el.value) : el.value;
  }
  if(state.currentModel==="kmeans"){
    // Keep dict format by default; users can convert to list if they want
    els.featuresBox.value = JSON.stringify(obj, null, 2);
  }else{
    els.featuresBox.value = JSON.stringify(obj, null, 2);
  }
}

function pretty(kind){
  return {logreg:"Churn — Logistic Regression",dtree:"Churn — Decision Tree",svm:"Churn — SVM",kmeans:"Clustering — KMeans",linreg:"Sales — Linear Regression"}[kind] || kind;
}

function toggleLoading(b){
  [els.sendBtn, els.healthBtn].forEach(btn=>{
    btn.disabled = b; btn.classList.toggle("loading", b);
  });
}

async function sendPrediction(){
  const base = cleanBase();
  let features;
  try{ features = JSON.parse(els.featuresBox.value || "{}"); }
  catch{ return toast("Invalid JSON in Features"); }

  const payload = { model_type: els.model.value, features };
  toggleLoading(true);
  try{
    const res = await fetch(`${base}/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)
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
  const base = cleanBase();
  toggleLoading(true);
  try{
    const res = await fetch(`${base}/health`);
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
  const badge = pred===1 ? "badge bad">Churn : "badge ok">Stay;
  const pct = proba!=null ? Math.max(0, Math.min(1, proba)) : null;

  els.pretty.innerHTML = `
    <div class="card-sm">
      <div class="kv">
        <div><strong>Prediction</strong></div>
        <div class="${pred===1?"badge bad":"badge ok"}">${pred===1?"Churn":"Stay"}</div>
      </div>
      ${pct!=null ? `
      <div class="mt-sm small">Churn probability: ${(pct*100).toFixed(1)}%</div>
      <div class="bar"><span style="width:${(pct*100).toFixed(0)}%"></span></div>
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
}

async function copyAsCurl(){
  const base = cleanBase();
  let features;
  try{ features = JSON.parse(els.featuresBox.value || "{}"); }
  catch{ return toast("Invalid JSON"); }
  const body = JSON.stringify({ model_type: els.model.value, features });
  const curl = `curl -X POST ${base}/predict -H "Content-Type: application/json" -d '${body.replaceAll("'", "'\\''")}'`;
  await navigator.clipboard.writeText(curl);
  els.copyToast.hidden = false; setTimeout(()=>els.copyToast.hidden=true, 1200);
}

function toast(msg){ els.output.textContent = msg; }
