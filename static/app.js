const state = {
  schema: null,
  currentModel: null,
  fieldsMeta: {},   // per-model array of fields
};

const els = {
  model: document.getElementById("model"),
  baseUrl: document.getElementById("baseUrl"),
  formArea: document.getElementById("formArea"),
  featuresBox: document.getElementById("featuresBox"),
  output: document.getElementById("output"),
  sendBtn: document.getElementById("sendBtn"),
  resetBtn: document.getElementById("resetBtn"),
  healthBtn: document.getElementById("healthBtn"),
  copyCurlBtn: document.getElementById("copyCurlBtn"),
  copyToast: document.getElementById("copyToast"),
  themeToggle: document.getElementById("themeToggle"),
};

init();

async function init() {
  // Theme
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  els.themeToggle.checked = savedTheme === "dark";
  els.themeToggle.addEventListener("change", () => {
    const t = els.themeToggle.checked ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("theme", t);
  });

  // Fetch schema & build model dropdown
  await loadSchema();
  buildModelSelect();

  // Events
  els.model.addEventListener("change", () => {
    state.currentModel = els.model.value;
    buildFormForModel(state.currentModel);
    populateJsonBox(state.currentModel);
  });

  els.sendBtn.addEventListener("click", sendPrediction);
  els.resetBtn.addEventListener("click", () => {
    buildFormForModel(state.currentModel, true);
    populateJsonBox(state.currentModel);
    els.output.textContent = "—";
  });

  els.healthBtn.addEventListener("click", checkHealth);
  els.copyCurlBtn.addEventListener("click", copyAsCurl);

  // Initial UI
  state.currentModel = els.model.value;
  buildFormForModel(state.currentModel);
  populateJsonBox(state.currentModel);
}

async function loadSchema() {
  try {
    const base = cleanBase();
    const res = await fetch(`${base}/schema`);
    state.schema = await res.json();
  } catch (e) {
    state.schema = null;
    toast("Couldn’t load /schema");
  }
}

function buildModelSelect() {
  const m = state.schema?.models || {};
  const order = [
    "logreg_churn", "decision_tree_churn", "svm_churn",
    "kmeans_clusters", "linreg_sales"
  ];
  els.model.innerHTML = "";
  order.forEach(key => {
    if (m[key]) {
      const opt = document.createElement("option");
      opt.value = m[key].model_type;
      opt.textContent = prettyName(m[key].model_type);
      els.model.appendChild(opt);
    }
  });
}

function prettyName(kind) {
  switch (kind) {
    case "logreg": return "Churn — Logistic Regression";
    case "dtree":  return "Churn — Decision Tree";
    case "svm":    return "Churn — SVM";
    case "kmeans": return "Clustering — KMeans";
    case "linreg": return "Sales — Linear Regression";
    default:       return kind;
  }
}

function buildFormForModel(kind, reset=false) {
  const m = state.schema?.models;
  if (!m) return;

  let fields = [];
  if (kind === "kmeans") {
    fields = m.kmeans_clusters.required_numeric_fields || [];
  } else if (kind === "linreg") {
    fields = m.linreg_sales.required_fields || [];
  } else if (kind === "logreg") {
    fields = m.logreg_churn.required_fields || [];
  } else if (kind === "dtree") {
    fields = m.decision_tree_churn.required_fields || [];
  } else if (kind === "svm") {
    fields = m.svm_churn.required_fields || [];
  }
  state.fieldsMeta[kind] = fields;

  // Default values from examples
  let example = getExampleFor(kind);
  let defaults = example?.features || {};

  // Render inputs
  els.formArea.innerHTML = "";
  const frag = document.createDocumentFragment();
  fields.forEach(name => {
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `
      <label>${name}</label>
      <input data-key="${name}" type="${isNumeric(name) ? "number" : "text"}"
             value="${reset ? "" : (defaults[name] ?? "")}">
    `;
    frag.appendChild(field);
  });
  els.formArea.appendChild(frag);
}

function populateJsonBox(kind) {
  const example = getExampleFor(kind);
  const features = {};
  const fields = state.fieldsMeta[kind] || [];
  // Grab current form values if present; otherwise use example
  fields.forEach(name => {
    const el = els.formArea.querySelector(`input[data-key="${name}"]`);
    if (el) {
      features[name] = el.type === "number" && el.value !== "" ? Number(el.value) : el.value;
    } else {
      features[name] = example?.features?.[name];
    }
  });
  // For KMeans, allow list format too (advanced users can replace in textarea)
  els.featuresBox.value = JSON.stringify(features, null, 2);
}

function getExampleFor(kind) {
  const m = state.schema?.models;
  if (!m) return null;
  switch (kind) {
    case "logreg": return m.logreg_churn.example_payload;
    case "dtree":  return m.decision_tree_churn.example_payload;
    case "svm":    return m.svm_churn.example_payload;
    case "kmeans": return m.kmeans_clusters.example_payload_dict;
    case "linreg": return m.linreg_sales.example_payload;
    default:       return null;
  }
}

function isNumeric(name) {
  return ["price","quantity","total_value","age","tenure_months"].includes(name);
}

function cleanBase() {
  // Allow "/" (same-origin) or a full URL
  return (els.baseUrl.value || "/").replace(/\/$/, "");
}

async function sendPrediction() {
  const base = cleanBase();
  let payload;
  try {
    // Prefer JSON box (what the user sees/edits), but align model_type from select
    const featuresJson = JSON.parse(els.featuresBox.value || "{}");
    payload = { model_type: els.model.value, features: featuresJson };
  } catch {
    toast("Invalid JSON in Features");
    return;
  }

  toggleLoading(true);
  try {
    const res = await fetch(`${base}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    els.output.textContent = "Request failed: " + e.message;
  } finally {
    toggleLoading(false);
  }
}

async function checkHealth() {
  const base = cleanBase();
  toggleLoading(true);
  try {
    const res = await fetch(`${base}/health`);
    const data = await res.json();
    els.output.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    els.output.textContent = "Health check failed: " + e.message;
  } finally {
    toggleLoading(false);
  }
}

function toggleLoading(isLoading) {
  els.sendBtn.disabled = isLoading;
  els.healthBtn.disabled = isLoading;
  els.sendBtn.classList.toggle("loading", isLoading);
  els.healthBtn.classList.toggle("loading", isLoading);
}

async function copyAsCurl() {
  const base = cleanBase();
  let features;
  try {
    features = JSON.parse(els.featuresBox.value || "{}");
  } catch {
    toast("Invalid JSON");
    return;
  }
  const body = JSON.stringify({ model_type: els.model.value, features }, null, 0);
  const curl = `curl -X POST ${base}/predict -H "Content-Type: application/json" -d '${body.replaceAll("'", "'\\''")}'`;
  await navigator.clipboard.writeText(curl);
  els.copyToast.hidden = false;
  setTimeout(() => els.copyToast.hidden = true, 1200);
}

function toast(msg) {
  els.output.textContent = msg;
}
