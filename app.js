const API_BASE_URL = "http://localhost:5000";

const form = document.getElementById("mainForm");
const formCard = document.getElementById("formCard");
const resultCard = document.getElementById("resultCard");
const submitBtn = document.getElementById("submitBtn");
const btnLabel = document.querySelector(".btn-label");
const btnSpin = document.querySelector(".btn-spin");
const errMsg = document.getElementById("errMsg");

const verdictBox = document.getElementById("verdictBox");
const verdictLabel = document.getElementById("verdictLabel");
const statBMI = document.getElementById("statBMI");
const statConf = document.getElementById("statConf");
const statBMICat = document.getElementById("statBMICat");
const scaleBMIVal = document.getElementById("scaleBMIVal");
const scaleNeedle = document.getElementById("scaleNeedle");
const probsContainer = document.getElementById("probsContainer");

const NUMERIC_FIELDS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"];

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  btnLabel.style.display = isLoading ? "none" : "inline";
  btnSpin.style.display = isLoading ? "inline" : "none";
}

function showError(message) {
  errMsg.textContent = message;
  errMsg.style.display = "block";
}

function hideError() {
  errMsg.style.display = "none";
  errMsg.textContent = "";
}

function formatClassName(name) {
  return String(name || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getWhoCategory(bmi) {
  if (bmi < 18.5) return "Underweight";
  if (bmi < 25) return "Normal";
  if (bmi < 30) return "Overweight";
  if (bmi < 35) return "Obesity I";
  if (bmi < 40) return "Obesity II";
  return "Obesity III";
}

function updateBmiNeedle(bmi) {
  const min = 10;
  const max = 40;
  const clamped = Math.min(max, Math.max(min, bmi));
  const percent = ((clamped - min) / (max - min)) * 100;
  scaleNeedle.style.left = `${percent}%`;
}

function renderProbabilities(allProbabilities) {
  probsContainer.innerHTML = "";

  const entries = Object.entries(allProbabilities || {}).sort((a, b) => b[1] - a[1]);

  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.style.marginBottom = "10px";

    const top = document.createElement("div");
    top.style.display = "flex";
    top.style.justifyContent = "space-between";
    top.style.fontSize = "0.92rem";
    top.style.marginBottom = "4px";

    const nameEl = document.createElement("span");
    nameEl.textContent = formatClassName(label);

    const valueEl = document.createElement("span");
    valueEl.textContent = `${Number(value).toFixed(2)}%`;

    const track = document.createElement("div");
    track.style.height = "9px";
    track.style.borderRadius = "999px";
    track.style.background = "rgba(255,255,255,0.12)";
    track.style.overflow = "hidden";

    const fill = document.createElement("div");
    fill.style.height = "100%";
    fill.style.width = `${Math.min(100, Math.max(0, value))}%`;
    fill.style.background = "linear-gradient(90deg, #06b6d4, #3b82f6)";

    top.appendChild(nameEl);
    top.appendChild(valueEl);
    track.appendChild(fill);
    row.appendChild(top);
    row.appendChild(track);
    probsContainer.appendChild(row);
  });
}

function collectPayload() {
  const data = Object.fromEntries(new FormData(form).entries());

  for (const field of NUMERIC_FIELDS) {
    const parsed = Number.parseFloat(data[field]);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Invalid value for ${field}`);
    }
    data[field] = parsed;
  }

  return data;
}

async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    if (!response.ok) {
      showError("Backend is running but health check failed.");
    }
  } catch (_err) {
    showError("Cannot reach backend at http://localhost:5000. Start app.py first.");
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  hideError();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  let payload;
  try {
    payload = collectPayload();
  } catch (err) {
    showError(err.message);
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Prediction failed");
    }

    const label = result.display_name || formatClassName(result.prediction);
    const bmi = Number(result.bmi);
    const confidence = Number(result.confidence);

    verdictLabel.textContent = label;
    verdictBox.style.borderColor = result.color || "#22c55e";
    verdictBox.style.boxShadow = `0 0 0 1px ${result.color || "#22c55e"}33`;

    statBMI.textContent = Number.isFinite(bmi) ? bmi.toFixed(2) : "-";
    statConf.textContent = Number.isFinite(confidence) ? `${confidence.toFixed(2)}%` : "-";
    statBMICat.textContent = Number.isFinite(bmi) ? getWhoCategory(bmi) : "-";

    scaleBMIVal.textContent = Number.isFinite(bmi) ? bmi.toFixed(2) : "-";
    if (Number.isFinite(bmi)) updateBmiNeedle(bmi);

    renderProbabilities(result.all_probabilities || {});

    formCard.style.display = "none";
    resultCard.style.display = "block";
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
}

function resetUI() {
  hideError();
  form.reset();
  resultCard.style.display = "none";
  formCard.style.display = "block";
}

window.resetUI = resetUI;
form.addEventListener("submit", handleSubmit);
checkBackendHealth();
