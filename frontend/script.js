/**
 * script.js
 * ---------
 * Frontend logic for the Clinical NLP Diagnosis Assistant.
 *
 * Responsibilities:
 *  - Collect clinical note from textarea or file upload
 *  - POST to FastAPI /diagnose endpoint
 *  - Render diagnosis, confidence, entities, and probability distribution
 *  - Handle loading, error, and empty states
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── DOM references ────────────────────────────────────────────────────────────
const noteInput      = document.getElementById("note-input");
const charCount      = document.getElementById("char-count");
const analyzeBtn     = document.getElementById("analyze-btn");
const spinner        = document.getElementById("spinner");
const btnLabel       = document.getElementById("btn-label");
const errorToast     = document.getElementById("error-toast");
const emptyState     = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");
const uploadZone     = document.getElementById("upload-zone");
const fileInput      = document.getElementById("file-input");
const fileNameEl     = document.getElementById("file-name");
const statusPill     = document.getElementById("status-pill");

// ── Startup: check backend connectivity ──────────────────────────────────────
(async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusPill.textContent = "● API Online";
      statusPill.className   = "pill pill-green";
    } else {
      throw new Error("not ok");
    }
  } catch {
    statusPill.textContent = "● API Offline";
    statusPill.className   = "pill";
    statusPill.style.background   = "rgba(248,81,73,0.12)";
    statusPill.style.color        = "var(--danger)";
    statusPill.style.borderColor  = "rgba(248,81,73,0.3)";
  }
})();

// ── Character counter ─────────────────────────────────────────────────────────
noteInput.addEventListener("input", () => {
  const n = noteInput.value.length;
  charCount.textContent = `${n} char${n !== 1 ? "s" : ""}`;
  hideError();
});

// ── Example chips ─────────────────────────────────────────────────────────────
document.querySelectorAll(".example-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    noteInput.value = chip.dataset.note;
    charCount.textContent = `${chip.dataset.note.length} chars`;
    noteInput.focus();
    hideError();
  });
});

// ── File upload – click ───────────────────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", () => {
  readFile(fileInput.files[0]);
});

// ── File upload – drag and drop ───────────────────────────────────────────────
uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) readFile(file);
});

function readFile(file) {
  if (!file) return;
  if (!file.name.endsWith(".txt") && file.type !== "text/plain") {
    showError("Please upload a plain .txt file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = ev => {
    noteInput.value = ev.target.result;
    charCount.textContent = `${ev.target.result.length} chars`;
    fileNameEl.textContent = `✓ ${file.name}`;
    hideError();
  };
  reader.onerror = () => showError("Could not read the file. Please try again.");
  reader.readAsText(file);
}

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

noteInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalysis();
});

async function runAnalysis() {
  const text = noteInput.value.trim();

  if (!text) {
    showError("Please enter a clinical note before analyzing.");
    return;
  }
  if (text.length < 10) {
    showError("The note is too short. Please provide more clinical detail.");
    return;
  }

  setLoading(true);
  hideError();

  try {
    const response = await fetch(`${API_BASE}/diagnose`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
      signal:  AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResults(data);

  } catch (err) {
    if (err.name === "TimeoutError") {
      showError("Request timed out. Is the backend running on port 8000?");
    } else if (err.message.includes("fetch") || err.message.includes("Failed")) {
      showError(
        "Cannot connect to the backend API. " +
        "Start it with: cd backend && uvicorn main:app --reload"
      );
    } else {
      showError(`Error: ${err.message}`);
    }
  } finally {
    setLoading(false);
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  emptyState.style.display    = "none";
  resultsContent.style.display = "block";

  renderDiagnosis(data.diagnosis, data.confidence);
  renderEntities(data.symptoms, data.medications, data.conditions);
  renderProbabilities(data.all_probabilities, data.diagnosis);

  // Smooth scroll
  resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderDiagnosis(diagnosis, confidence) {
  const pct = Math.round(confidence * 100);

  document.getElementById("diagnosis-name").textContent = diagnosis;
  document.getElementById("confidence-pct").textContent = `${pct}%`;
  document.getElementById("conf-pct-label").textContent = `${pct}%`;

  // Animate confidence bar
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById("conf-bar-fill").style.width = `${pct}%`;
    }, 80);
  });

  // Confidence level label + color
  let level, color;
  if (pct >= 80) {
    level = "High confidence — strong signal detected";
    color = "var(--success)";
  } else if (pct >= 55) {
    level = "Moderate confidence — review recommended";
    color = "var(--warn)";
  } else {
    level = "Low confidence — insufficient clinical detail";
    color = "var(--danger)";
  }

  document.getElementById("conf-dot").style.background    = color;
  document.getElementById("conf-level-text").textContent  = level;
  document.getElementById("conf-level-text").style.color  = color;
  document.getElementById("confidence-pct").style.color   = color;
}

function renderEntities(symptoms, medications, conditions) {
  renderTagGroup("symptoms-tags",    "symptom-count",    symptoms,    "tag-symptom");
  renderTagGroup("medications-tags", "medication-count", medications, "tag-medication");
  renderTagGroup("conditions-tags",  "condition-count",  conditions,  "tag-condition");
}

function renderTagGroup(containerId, countId, items, tagClass) {
  const container = document.getElementById(containerId);
  const countEl   = document.getElementById(countId);

  countEl.textContent = items.length;

  if (!items || items.length === 0) {
    container.innerHTML = `<span class="no-entities">None detected</span>`;
    return;
  }

  container.innerHTML = items
    .map((item, i) =>
      `<span class="tag ${tagClass}" style="animation-delay:${i * 0.04}s">${capitalize(item)}</span>`
    )
    .join("");
}

function renderProbabilities(probs, topDiagnosis) {
  const list   = document.getElementById("prob-list");
  const sorted = Object.entries(probs).sort(([, a], [, b]) => b - a);
  const maxVal = sorted[0]?.[1] ?? 1;

  const PALETTE = [
    "#2ea4a0", "#58a6ff", "#3fb950", "#d29922",
    "#f85149", "#bc8cff", "#ff7b72", "#ffa657",
    "#79c0ff", "#56d364",
  ];

  list.innerHTML = sorted.map(([disease, prob], i) => {
    const pct      = Math.round(prob * 100);
    const barWidth = maxVal > 0 ? Math.round((prob / maxVal) * 100) : 0;
    const isTop    = disease === topDiagnosis;
    const color    = PALETTE[i % PALETTE.length];

    return `
      <div class="prob-item">
        <div class="prob-meta">
          <div class="prob-disease">
            <span class="prob-name" style="${isTop ? "font-weight:700;" : ""}">${disease}</span>
            ${isTop ? `<span class="prob-top-badge">✓ Top</span>` : ""}
          </div>
          <span class="prob-pct">${pct}%</span>
        </div>
        <div class="prob-bar-bg">
          <div
            class="prob-bar-fill"
            style="background:${color}"
            data-width="${barWidth}%"
          ></div>
        </div>
      </div>
    `;
  }).join("");

  // Animate bars on next frame
  requestAnimationFrame(() => {
    setTimeout(() => {
      list.querySelectorAll(".prob-bar-fill").forEach(el => {
        el.style.width = el.dataset.width;
      });
    }, 100);
  });
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled        = on;
  spinner.style.display      = on ? "block" : "none";
  btnLabel.textContent       = on ? "Analyzing…" : "🔬 Analyze Clinical Note";
}

function showError(msg) {
  errorToast.textContent   = `⚠ ${msg}`;
  errorToast.style.display = "block";
}

function hideError() {
  errorToast.style.display = "none";
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}
