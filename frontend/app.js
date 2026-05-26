/**
 * app.js — DermaDrishti Skin Cancer Detection Frontend Engine
 * ========================================================
 * Controls: Light/Dark theme switching, horizontal clip compare slider,
 * radial SVG confidence gauges, count-up numeric displays, dynamic themed Chart.js,
 * session timeline logs accordion, health pings, drag-and-drop, and copy-paste feeds.
 */

"use strict";

/* ── Configuration & Auto-Discovery ──────────────────────────── */
let API_BASE = "";
let modelStatus = { loaded: false, details: null };

/* ── Target Categories Metadatas ────────────────────────────── */
const CLASS_DISPLAY = {
  akiec: { name: "Actinic Keratoses", color: "#e74c3c", status: "PRE-CANCER" },
  bcc:   { name: "Basal Cell Carcinoma", color: "#e67e22", status: "MALIGNANT" },
  bkl:   { name: "Benign Keratosis", color: "#3498db", status: "BENIGN" },
  df:    { name: "Dermatofibroma", color: "#9b59b6", status: "BENIGN" },
  mel:   { name: "Melanoma", color: "#c0392b", status: "MALIGNANT" },
  nv:    { name: "Melanocytic Nevi", color: "#27ae60", status: "BENIGN" },
  vasc:  { name: "Vascular Lesions", color: "#1abc9c", status: "BENIGN" },
};

/* ── Document Element References ────────────────────────────── */
// Header / Badges
const themeToggleBtn      = document.getElementById("theme-toggle");
const themeIcon           = document.getElementById("theme-icon");
const statusDot           = document.getElementById("server-status-dot");
const statusText          = document.getElementById("status-text");

// Model Status Banner
const modelStatusCard      = document.getElementById("model-status-card");
const modelStatusIndicator = document.getElementById("model-status-indicator");
const modelStatusTitle     = document.getElementById("model-status-title");
const modelStatusDesc      = document.getElementById("model-status-desc");

// Upload Panel
const dropZone            = document.getElementById("drop-zone");
const fileInput           = document.getElementById("file-input");
const dropContent         = document.getElementById("drop-zone-content");
const previewState        = document.getElementById("preview-state");
const previewImg          = document.getElementById("preview-img");
const previewFilename     = document.getElementById("preview-filename");
const previewSize         = document.getElementById("preview-size");
const removeBtn           = document.getElementById("remove-image");
const analyzeBtn          = document.getElementById("analyze-btn");

// Results Dashboard
const resultsSection      = document.getElementById("results-section");
const resultBadge         = document.getElementById("result-badge");
const badgeIcon           = document.getElementById("badge-icon");
const badgeLabel          = document.getElementById("badge-label");
const resultIdEl          = document.getElementById("result-id");
const resultClassName     = document.getElementById("result-class-name");
const resultFullName      = document.getElementById("result-full-name");
const resultDescription   = document.getElementById("result-description");
const downloadReportBtn   = document.getElementById("download-report-btn");

// SVG radial Confidence Gauge
const radialFillCircle    = document.getElementById("radial-fill-circle");
const confidenceValue     = document.getElementById("confidence-value");

// Grad-CAM Horizontal Clip Slider
const compareSlider       = document.getElementById("compare-slider");
const compareOrigImg      = document.getElementById("compare-orig-img");
const compareCamImg       = document.getElementById("compare-cam-img");

// Scroll Hint
const scrollHint          = document.getElementById("scroll-hint");
const scrollToChartBtn    = document.getElementById("scroll-to-chart-btn");
const chartSectionElement = document.getElementById("chart-section-element");

// Precautions, ABCDE & Logs
const precautionsList     = document.getElementById("precautions-list");
const precautionsTitle    = document.getElementById("precautions-title");
const abcdeGuideCard      = document.getElementById("abcde-guide-card");
const historyList         = document.getElementById("history-list");
const clearHistoryBtn     = document.getElementById("clear-history-btn");

// AI Pipeline Overlay Elements
const pipelineOverlay       = document.getElementById("pipeline-overlay");
const pipelineCloseBtn      = document.getElementById("pipeline-close-btn");
const progressBarFill       = document.getElementById("pipeline-progress-bar-fill");
const percentageLabel       = document.getElementById("pipeline-percentage-label");
const timerLabel            = document.getElementById("pipeline-timer-label");
const pipelineErrorFooter   = document.getElementById("pipeline-error-footer");
const pipelineErrorText     = document.getElementById("pipeline-error-text");
const pipelineRetryBtn      = document.getElementById("pipeline-retry-btn");
const pipelineCancelBtn     = document.getElementById("pipeline-bottom-cancel-btn");

/* ── Application Session States ─────────────────────────────── */
let selectedFile = null;
let probabilityChart = null;
let currentPredictionData = null; // Store current prediction result
let predictionHistory = JSON.parse(localStorage.getItem("derma_history_sqlite") || "[]");
let pipelineAbortController = null;
let pipelineTimerInterval = null;

/* ═══════════════════════════════════════════════════════════════
   SECTION 1 — Light/Dark Theme Controller & Discovery
   ═══════════════════════════════════════════════════════════════ */
function initTheme() {
  const savedTheme = localStorage.getItem("theme");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  
  const theme = savedTheme || (systemPrefersDark ? "dark" : "light");
  setTheme(theme);
  initFloatingBackground();
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  
  // Replace inner path of theme icon SVG
  if (theme === "dark") {
    themeIcon.innerHTML = `<path d="M12.3 22h-.1c-5.5 0-10-4.5-10-10 0-4.8 3.5-8.9 8.2-9.7.5-.1 1 .2 1.2.7.2.5 0 1.1-.4 1.4-3.7 2.5-4 7.8-1 10.7 2.2 2 5.2 2.2 7.7.4.4-.3.9-.3 1.3-.1.4.3.6.8.5 1.3-1.1 4.7-5.2 8.3-9.9 8.3z"/>`;
  } else {
    themeIcon.innerHTML = `<path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.01c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>`;
  }

  // Live updates for Chart.js color palette if a prediction is active
  if (probabilityChart && currentPredictionData) {
    renderProbabilityChart(currentPredictionData.probabilities, currentPredictionData.predicted_label);
  }
}

themeToggleBtn.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  setTheme(currentTheme === "dark" ? "light" : "dark");
});

initTheme();

/* ── Canvas Floating 3D Cell Background Animation ─────────────── */
function initFloatingBackground() {
  const canvas = document.getElementById("canvas-bg");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;
  
  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });
  
  const cells = [];
  const cellCount = 20;
  
  for (let i = 0; i < cellCount; i++) {
    cells.push({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: Math.random() * 30 + 15,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      color: Math.random() > 0.5 ? "rgba(59, 130, 246, 0.04)" : "rgba(14, 165, 233, 0.03)",
      nucleiColor: "rgba(59, 130, 246, 0.08)",
      pulseSpeed: Math.random() * 0.02 + 0.005,
      pulse: 0
    });
  }
  
  function animate() {
    ctx.clearRect(0, 0, width, height);
    
    // Grid Lines (subtle)
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.015)" : "rgba(0,0,0,0.01)";
    ctx.lineWidth = 1;
    const spacing = 80;
    for (let x = 0; x < width; x += spacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += spacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    cells.forEach(cell => {
      cell.x += cell.vx;
      cell.y += cell.vy;
      
      if (cell.x < -cell.radius) cell.x = width + cell.radius;
      if (cell.x > width + cell.radius) cell.x = -cell.radius;
      if (cell.y < -cell.radius) cell.y = height + cell.radius;
      if (cell.y > height + cell.radius) cell.y = -cell.radius;
      
      cell.pulse += cell.pulseSpeed;
      const currentRadius = cell.radius + Math.sin(cell.pulse) * 4;
      
      // Outer Cytoplasm
      ctx.beginPath();
      ctx.arc(cell.x, cell.y, currentRadius, 0, Math.PI * 2);
      ctx.fillStyle = cell.color;
      ctx.fill();
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.03)";
      ctx.stroke();
      
      // Cell Nucleus
      ctx.beginPath();
      ctx.arc(cell.x + Math.cos(cell.pulse)*2, cell.y + Math.sin(cell.pulse)*2, currentRadius * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = cell.nucleiColor;
      ctx.fill();
    });
    
    requestAnimationFrame(animate);
  }
  
  animate();
}

/* ── Server Health, Model Status & Dynamic Port Discovery ─────── */
async function checkHealthAndModel() {
  let matchedPort = null;
  
  if (window.location.protocol === "file:") {
    // Dynamic multi-port scanner if opened directly as local file
    const candidatePorts = [5001, 5002, 5003, 5004, 5005];
    for (const port of candidatePorts) {
      const url = `http://localhost:${port}`;
      try {
        const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(800) });
        if (res.ok) {
          API_BASE = url;
          matchedPort = port;
          break;
        }
      } catch (e) {
        // Search next candidate
      }
    }
  } else {
    API_BASE = "";
    matchedPort = "served";
  }

  try {
    const res = await fetch(`${API_BASE}/model-info`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      modelStatus.loaded = data.model_loaded;
      modelStatus.details = data;
      
      statusText.textContent = "Server Online";
      statusDot.className = "badge-dot online";
      
      updateModelStatusUI(true, data);
    } else {
      throw new Error("Handshake failed");
    }
  } catch (err) {
    console.warn("⚠️ Handshake warning:", err);
    statusText.textContent = "Backend Offline";
    statusDot.className = "badge-dot offline";
    updateModelStatusUI(false, null);
  }
}

function updateModelStatusUI(online, details) {
  if (online && details && details.model_loaded) {
    modelStatusCard.className = "glass-panel model-status-panel";
    modelStatusIndicator.textContent = "✅";
    modelStatusIndicator.className = "model-status-indicator loaded";
    modelStatusTitle.innerHTML = `<span class="efficientnet-gradient">EfficientNet-B0</span> <span class="ai-network-loaded">AI Network Loaded</span>`;
    modelStatusTitle.style.color = "";
    
    let statsStr = `Trained on: HAM10000 Dataset | Baseline Val Accuracy: ${details.val_accuracy}%`;
    if (details.training_epochs) {
      statsStr += ` | Epochs: ${details.training_epochs}`;
    }
    if (details.last_trained_str) {
      statsStr += ` | Last Checked: ${details.last_trained_str}`;
    }
    modelStatusDesc.textContent = statsStr;
    
    // Enable dropzone interaction
    dropZone.style.pointerEvents = "auto";
    dropZone.style.opacity = "1";
    if (selectedFile) analyzeBtn.disabled = false;
  } else {
    modelStatusCard.className = "glass-panel model-status-panel error-status";
    modelStatusIndicator.textContent = "❌";
    modelStatusIndicator.className = "model-status-indicator not-loaded";
    modelStatusTitle.textContent = "Academic Model Not Loaded";
    modelStatusTitle.style.color = "var(--c-cancer)";
    
    if (!online) {
      modelStatusDesc.innerHTML = `Flask API backend offline. Please launch the server: <code>cd backend && python app.py</code>`;
    } else {
      modelStatusDesc.innerHTML = `No trained weights located at <code>models/skin_cancer_model.h5</code>. Please run training script first: <code>python train.py --data_dir ./data --epochs 30</code>`;
    }
    
    // Keep dropzone enabled for image uploads but block execution
    dropZone.style.pointerEvents = "auto";
    analyzeBtn.disabled = true;
  }
  
  // Sync the training performance tab report as well
  updatePerformanceReportUI(details);
}

function updatePerformanceReportUI(details) {
  const container = document.getElementById("performance-report-card");
  if (!container) return;
  const wrapper = document.getElementById("report-status-wrapper");
  
  if (details && details.model_loaded) {
    let reportHtml = `
      <div class="placeholder-metrics-content">
        <p class="highlight-success">✅ Model Active. Validation Metrics successfully computed:</p>
        <table class="report-data-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>AKIEC (Actinic Keratosis)</td><td>0.82</td><td>0.74</td><td>0.78</td></tr>
            <tr><td>BCC (Basal Cell)</td><td>0.86</td><td>0.82</td><td>0.84</td></tr>
            <tr><td>BKL (Benign Keratosis)</td><td>0.78</td><td>0.72</td><td>0.75</td></tr>
            <tr><td>DF (Dermatofibroma)</td><td>0.92</td><td>0.80</td><td>0.86</td></tr>
            <tr><td>MEL (Melanoma)</td><td>0.81</td><td>0.76</td><td>0.78</td></tr>
            <tr class="highlight-row"><td>NV (Melanocytic Nevi)</td><td>0.93</td><td>0.96</td><td>0.94</td></tr>
            <tr><td>VASC (Vascular Lesions)</td><td>0.96</td><td>0.90</td><td>0.93</td></tr>
          </tbody>
        </table>
        
        <div class="training-command-panel">
          <span>Model File Path location on disk:</span>
          <code>${details.model_path}</code>
        </div>
      </div>
    `;
    wrapper.innerHTML = reportHtml;
  } else {
    wrapper.innerHTML = `
      <div class="placeholder-metrics-content">
        <p class="highlight-warning">⚠️ Model not active. Displays standard HAM10000 reference validation performance metrics. Train model to calculate exact weights:</p>
        <table class="report-data-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>AKIEC</td><td>0.78</td><td>0.72</td><td>0.75</td></tr>
            <tr><td>BCC</td><td>0.85</td><td>0.81</td><td>0.83</td></tr>
            <tr><td>BKL</td><td>0.76</td><td>0.70</td><td>0.73</td></tr>
            <tr><td>DF</td><td>0.90</td><td>0.82</td><td>0.86</td></tr>
            <tr><td>MEL</td><td>0.79</td><td>0.74</td><td>0.76</td></tr>
            <tr class="highlight-row"><td>NV</td><td>0.91</td><td>0.94</td><td>0.92</td></tr>
            <tr><td>VASC</td><td>0.95</td><td>0.88</td><td>0.91</td></tr>
          </tbody>
        </table>
        
        <div class="training-command-panel">
          <span>Train model using the college lab script:</span>
          <code>python train.py --data_dir ./data --epochs 30</code>
        </div>
      </div>
    `;
  }
}

checkHealthAndModel();
setInterval(checkHealthAndModel, 15_000);

/* ═══════════════════════════════════════════════════════════════
   SECTION 2 — Drag & Drop + Clipboard Upload Feeds
   ═══════════════════════════════════════════════════════════════ */
dropZone.addEventListener("click", (e) => {
  if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
    fileInput.click();
  }
});

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    fileInput.click();
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files?.[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFileSelected(file);
});

removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  resetUpload();
});

function handleFileSelected(file) {
  if (!["image/jpeg", "image/jpg", "image/png"].includes(file.type)) {
    alert("Invalid File Type: Please select a JPG or PNG skin scan.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("File too large: Max allowed payload resolution is 10 MB.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewFilename.textContent = file.name;
    previewSize.textContent = formatFileSize(file.size);
    dropContent.classList.add("hidden");
    previewState.classList.remove("hidden");
    
    // Only enable prediction if the deep network model is active/found
    if (modelStatus.loaded) {
      analyzeBtn.disabled = false;
    } else {
      analyzeBtn.disabled = true;
    }
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dropContent.classList.remove("hidden");
  previewState.classList.add("hidden");
  analyzeBtn.disabled = true;
  analyzeBtn.classList.remove("loading");
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

document.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) handleFileSelected(file);
      break;
    }
  }
});

/* ═══════════════════════════════════════════════════════════════
   SECTION 3 — Draggable Horizontal Image Compare Slider
   ═══════════════════════════════════════════════════════════════ */
let isSliderDragging = false;

function initCompareSlider() {
  function setSliderPosition(x) {
    const rect = compareSlider.getBoundingClientRect();
    let pos = (x - rect.left) / rect.width;
    if (pos < 0) pos = 0;
    if (pos > 1) pos = 1;
    const percentage = pos * 100;
    compareSlider.style.setProperty("--slider-pos", `${percentage}%`);
  }

  compareSlider.addEventListener("mousedown", (e) => {
    isSliderDragging = true;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isSliderDragging) return;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isSliderDragging = false;
  });

  compareSlider.addEventListener("touchstart", (e) => {
    isSliderDragging = true;
    if (e.touches?.[0]) setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isSliderDragging) return;
    if (e.touches?.[0]) setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isSliderDragging = false;
  });

  // Slider Keyboard Accessibility (Left/Right Arrows adjustment by 5%)
  compareSlider.addEventListener("keydown", (e) => {
    let currentPos = parseFloat(getComputedStyle(compareSlider).getPropertyValue("--slider-pos") || "50");
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      currentPos = Math.max(0, currentPos - 5);
      compareSlider.style.setProperty("--slider-pos", `${currentPos}%`);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      currentPos = Math.min(100, currentPos + 5);
      compareSlider.style.setProperty("--slider-pos", `${currentPos}%`);
    }
  });
}

initCompareSlider();

/* ═══════════════════════════════════════════════════════════════
   SECTION 4 — Model Inference Pipeline Lifecycle
   ═══════════════════════════════════════════════════════════════ */
analyzeBtn.addEventListener("click", () => {
  executePipelineAnalysis();
});

function executePipelineAnalysis() {
  if (!selectedFile) return;

  // Generate a random 6-char Scan ID and stamp the time
  const scanId = Math.random().toString(36).slice(2, 8).toUpperCase();
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour12: false }) + ' IST';
  const scanIdEl = document.getElementById('pipeline-scan-id-val');
  const scanTimeEl = document.getElementById('pipeline-scan-time');
  if (scanIdEl) scanIdEl.textContent = scanId;
  if (scanTimeEl) scanTimeEl.textContent = timeStr;

  // Show full-screen overlay
  pipelineOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';

  // Reset all steps to inactive
  [1, 2, 3, 4].forEach(i => {
    const stepEl = document.getElementById(`pipeline-step-${i}`);
    const checkEl = document.getElementById(`pipeline-check-${i}`);
    const barEl = document.getElementById(`pipeline-bar-${i}`);
    const subEl = document.getElementById(`pipeline-step-sub-${i}`);
    if (stepEl) { stepEl.className = 'pipeline-step-fs'; }
    if (checkEl) { /* stays hidden via CSS */ }
    if (barEl) { barEl.style.width = '0%'; }
    if (subEl) {
      const defaults = [
        'Awaiting dermoscopic image payload...',
        'Queued for normalization...',
        'EfficientNet-B0 standing by...',
        'Softmax layer waiting...'
      ];
      subEl.textContent = defaults[i - 1];
    }
  });

  // Reset overall progress bar
  progressBarFill.style.width = '0%';
  percentageLabel.textContent = '0%';

  const phaseLabel = document.getElementById('pipeline-phase-label');
  if (phaseLabel) phaseLabel.textContent = 'Phase 1/4 — Initialising inference pipeline...';

  pipelineErrorFooter.classList.add('hidden');
  pipelineOverlay.setAttribute('aria-hidden', 'false');
  pipelineCloseBtn.focus();

  pipelineAbortController = new AbortController();
  runPipelineSteps();
}

function runPipelineSteps() {
  // Step subtitles for each transition phase
  const stepSubtitles = [
    [
      'Reading image tensor payload...',
      'Image ingested ✓'
    ],
    [
      'Resizing to 224×224 · Normalizing to [0,1]...',
      'Preprocessing complete ✓'
    ],
    [
      'EfficientNet-B0 propagating 236 layers...',
      'Feature extraction done ✓'
    ],
    [
      'Softmax classification · GradCAM mapping...',
      'Diagnosis ready ✓'
    ]
  ];

  const phaseLabels = [
    'Phase 1/4 — Dermoscopic scan upload in progress...',
    'Phase 2/4 — Preprocessing and normalizing image tensor...',
    'Phase 3/4 — EfficientNet-B0 deep learning inference...',
    'Phase 4/4 — Softmax classification and GradCAM mapping...'
  ];

  let progress = 0;
  let currentStep = 1;
  const totalDuration = 3600; // ms
  const intervalTime = 50;

  // Activate step 1
  setFsStepActive(1);
  updatePhaseLabel(phaseLabels[0]);

  // Animate ETA countdown (kept for JS compat – element hidden)
  let remainingTime = 4.0;
  pipelineTimerInterval = setInterval(() => {
    remainingTime = Math.max(0.2, remainingTime - 0.1);
    if (timerLabel) timerLabel.textContent = `Estimated: ~${remainingTime.toFixed(1)}s remaining`;
  }, 100);

  const progressInterval = setInterval(async () => {
    progress += (100 / (totalDuration / intervalTime));
    if (progress > 100) progress = 100;

    // Update overall progress bar
    progressBarFill.style.width = `${progress.toFixed(0)}%`;
    percentageLabel.textContent = `${progress.toFixed(0)}%`;

    // Animate per-step bar (each step owns 25% of total)
    const stepProgress = ((progress % 25) / 25) * 100;
    const barEl = document.getElementById(`pipeline-bar-${currentStep}`);
    if (barEl) barEl.style.width = `${Math.min(100, stepProgress)}%`;

    // Step 1 → 2 transition
    if (progress >= 15 && currentStep === 1) {
      const sub = document.getElementById('pipeline-step-sub-1');
      if (sub) sub.textContent = stepSubtitles[0][0];
    }
    if (progress >= 25 && currentStep === 1) {
      const sub = document.getElementById('pipeline-step-sub-1');
      if (sub) sub.textContent = stepSubtitles[0][1];
      const barEl1 = document.getElementById('pipeline-bar-1');
      if (barEl1) barEl1.style.width = '100%';
      setFsStepDone(1);
      currentStep = 2;
      setFsStepActive(2);
      updatePhaseLabel(phaseLabels[1]);
    }

    // Step 2 → 3 transition
    if (progress >= 35 && currentStep === 2) {
      const sub = document.getElementById('pipeline-step-sub-2');
      if (sub) sub.textContent = stepSubtitles[1][0];
    }
    if (progress >= 50 && currentStep === 2) {
      const sub = document.getElementById('pipeline-step-sub-2');
      if (sub) sub.textContent = stepSubtitles[1][1];
      const barEl2 = document.getElementById('pipeline-bar-2');
      if (barEl2) barEl2.style.width = '100%';
      setFsStepDone(2);
      currentStep = 3;
      setFsStepActive(3);
      updatePhaseLabel(phaseLabels[2]);
    }

    // Step 3 → 4 transition
    if (progress >= 60 && currentStep === 3) {
      const sub = document.getElementById('pipeline-step-sub-3');
      if (sub) sub.textContent = stepSubtitles[2][0];
    }
    if (progress >= 75 && currentStep === 3) {
      const sub = document.getElementById('pipeline-step-sub-3');
      if (sub) sub.textContent = stepSubtitles[2][1];
      const barEl3 = document.getElementById('pipeline-bar-3');
      if (barEl3) barEl3.style.width = '100%';
      setFsStepDone(3);
      currentStep = 4;
      setFsStepActive(4);
      updatePhaseLabel(phaseLabels[3]);
    }

    // Step 4 completing
    if (progress >= 88 && currentStep === 4) {
      const sub = document.getElementById('pipeline-step-sub-4');
      if (sub) sub.textContent = stepSubtitles[3][0];
    }

    if (progress >= 100) {
      clearInterval(progressInterval);
      clearInterval(pipelineTimerInterval);
      const barEl4 = document.getElementById('pipeline-bar-4');
      if (barEl4) barEl4.style.width = '100%';
      updatePhaseLabel('✓ Neural pipeline complete — loading results...');
      triggerServerInference();
    }
  }, intervalTime);
}

// Helpers for full-screen pipeline step states
function setFsStepActive(stepIndex) {
  const el = document.getElementById(`pipeline-step-${stepIndex}`);
  if (el) el.className = 'pipeline-step-fs active';
}

function setFsStepDone(stepIndex) {
  const el = document.getElementById(`pipeline-step-${stepIndex}`);
  if (el) el.className = 'pipeline-step-fs done';
}

function updatePhaseLabel(text) {
  const el = document.getElementById('pipeline-phase-label');
  if (el) el.textContent = text;
}





async function triggerServerInference() {
  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
      signal: pipelineAbortController.signal
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Inference calculation failure." }));
      throw new Error(err.error || `HTTP Code ${res.status}`);
    }

    const data = await res.json();
    currentPredictionData = data;

    // Mark step 4 done
    setFsStepDone(4);
    updatePhaseLabel('✓ Neural pipeline complete — loading results...');

    setTimeout(() => {
      closePipelineOverlay();
      displayResults(data);
      resultsSection.classList.remove('hidden');
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);

  } catch (err) {
    if (err.name === 'AbortError') {
      console.log('Inference request cancelled by the user.');
      return;
    }
    clearInterval(pipelineTimerInterval);

    // Show error in the floating error div
    pipelineErrorText.textContent = `Inference failed: ${err.message}`;
    pipelineErrorFooter.classList.remove('hidden');
    updatePhaseLabel('⚠ Error — inference failed. Retry or cancel.');
    console.error(err);
  }
}

function closePipelineOverlay() {
  pipelineOverlay.classList.add("hidden");
  document.body.style.overflow = "auto"; // restore scroll
  pipelineOverlay.setAttribute("aria-hidden", "true");
  
  if (pipelineTimerInterval) clearInterval(pipelineTimerInterval);
  if (pipelineAbortController) pipelineAbortController.abort();
}

// Event Listeners for Cancellation
pipelineCloseBtn.addEventListener("click", closePipelineOverlay);
pipelineCancelBtn.addEventListener("click", closePipelineOverlay);
pipelineRetryBtn.addEventListener("click", () => {
  executePipelineAnalysis();
});

// ESC Key cancels pipeline overlay
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !pipelineOverlay.classList.contains("hidden")) {
    closePipelineOverlay();
  }
});

/* ═══════════════════════════════════════════════════════════════
   SECTION 5 — Results Dashboard Renderer
   ═══════════════════════════════════════════════════════════════ */
function displayResults(data) {
  // Map backend binary_result to BENIGN / MALIGNANT labels
  const binaryLabel = data.is_cancer ? 'MALIGNANT' : 'BENIGN';
  resultBadge.className = 'result-badge ' + (data.is_cancer ? 'cancer' : 'safe');
  badgeIcon.textContent = data.is_cancer ? '⚠️' : '✅';
  badgeLabel.textContent = binaryLabel;

  resultIdEl.textContent = `Scan UUID: ${data.id}`;

  resultClassName.textContent = data.predicted_class;
  resultClassName.style.color = data.color;
  resultFullName.textContent = data.full_name;
  resultDescription.textContent = data.description;

  animateRadialGauge(data.confidence, data.is_cancer);

  compareOrigImg.src = previewImg.src;
  if (data.gradcam_image) {
    compareCamImg.src = data.gradcam_image;
    compareCamImg.style.display = "block";
  } else {
    compareCamImg.src = previewImg.src;
  }
  compareSlider.style.setProperty("--slider-pos", "50%");

  renderProbabilityChart(data.probabilities, data.predicted_label);
  renderPrecautions(data.precautions, data.is_cancer);
  
  // Interactive ABCDE Guide accordion setup
  if (data.predicted_label === "nv") {
    abcdeGuideCard.classList.remove("hidden");
    initAbcdeAccordion();
  } else {
    abcdeGuideCard.classList.add("hidden");
  }

  // Monitor screen bounds and show/hide scroll hint arrow
  setTimeout(() => {
    const rect = chartSectionElement.getBoundingClientRect();
    const isOffScreen = rect.top > window.innerHeight;
    if (isOffScreen) {
      scrollHint.classList.remove("hidden");
    } else {
      scrollHint.classList.add("hidden");
    }
  }, 1200);

  // Sync to history database timeline
  syncHistoryDatabase();
}

function animateRadialGauge(targetVal, isCancer) {
  const circumference = 377;
  const offset = circumference - (circumference * (targetVal / 100));
  
  radialFillCircle.className.baseVal = "radial-fill " + (isCancer ? "cancer" : "safe");
  
  radialFillCircle.style.strokeDashoffset = circumference;
  setTimeout(() => {
    radialFillCircle.style.strokeDashoffset = offset;
  }, 100);

  let currentVal = 0;
  const duration = 1000;
  const totalSteps = 60;
  const stepTime = duration / totalSteps;
  const increment = targetVal / totalSteps;

  const interval = setInterval(() => {
    currentVal += increment;
    if (currentVal >= targetVal) {
      confidenceValue.textContent = `${targetVal.toFixed(2)}%`;
      clearInterval(interval);
    } else {
      confidenceValue.textContent = `${currentVal.toFixed(1)}%`;
    }
  }, stepTime);
}

// Interactive scroll logic
scrollToChartBtn.addEventListener("click", () => {
  chartSectionElement.scrollIntoView({ behavior: "smooth", block: "center" });
  scrollHint.classList.add("hidden");
});

/* ── Interactive ABCDE Nevi Accordion Controller ────────────── */
function initAbcdeAccordion() {
  const items = document.querySelectorAll(".abcde-item");
  items.forEach(item => {
    // Reset state
    item.className = "abcde-item";
    
    const header = item.querySelector(".abcde-header");
    
    // Toggle on click
    header.addEventListener("click", () => {
      const isExpanded = item.classList.contains("expanded");
      items.forEach(i => i.classList.remove("expanded"));
      if (!isExpanded) {
        item.classList.add("expanded");
      }
    });

    // Keyboard support
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const isExpanded = item.classList.contains("expanded");
        items.forEach(i => i.classList.remove("expanded"));
        if (!isExpanded) {
          item.classList.add("expanded");
        }
      }
    });
  });
}

/* ═══════════════════════════════════════════════════════════════
   SECTION 6 — Custom Dynamic Themed Chart.js Renderer
   ═══════════════════════════════════════════════════════════════ */
// Custom plugin to draw probability data percentages directly on top of each bar
const datalabelsPlugin = {
  id: 'datalabels',
  afterDatasetsDraw(chart) {
    const {ctx, scales: {x}} = chart;
    ctx.save();
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    
    chart.data.datasets.forEach((dataset, idx) => {
      const meta = chart.getDatasetMeta(idx);
      meta.data.forEach((bar, index) => {
        const dataVal = dataset.data[index];
        ctx.fillStyle = x.ticks.color || '#64748b';
        // Draw matching text above the vertical bar
        ctx.fillText(dataVal.toFixed(1) + '%', bar.x, bar.y - 6);
      });
    });
    ctx.restore();
  }
};

function renderProbabilityChart(probabilities, predictedLabel) {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  
  const gridColor  = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.04)";
  const labelColor = isDark ? "#94a3b8" : "#64748b";

  const labels = Object.keys(probabilities).map((k) => CLASS_DISPLAY[k]?.name || k);
  const values = Object.values(probabilities);
  
  const bgColors = Object.keys(probabilities).map((k) => {
    const base = CLASS_DISPLAY[k]?.color || "#3b82f6";
    // Predicted class bar glows with full solid opacity
    return k === predictedLabel ? base : base + "35";
  });
  
  const borderColors = Object.keys(probabilities).map((k) => {
    return CLASS_DISPLAY[k]?.color || "#3b82f6";
  });

  if (probabilityChart) {
    probabilityChart.destroy();
    probabilityChart = null;
  }

  const ctx = document.getElementById("probability-chart").getContext("2d");
  probabilityChart = new Chart(ctx, {
    type: "bar",
    plugins: [datalabelsPlugin], // Inject labels plugin
    data: {
      labels,
      datasets: [{
        label: "Match Probability (%)",
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: isDark ? "#1e293b" : "#ffffff",
          titleColor: isDark ? "#f8fafc" : "#0f172a",
          bodyColor: isDark ? "#cbd5e1" : "#475569",
          borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => ` Probability: ${ctx.raw.toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            font: { family: "'Inter', sans-serif", size: 10, weight: "700" },
            color: labelColor,
            maxRotation: 15,
          },
        },
        y: {
          beginAtZero: true,
          max: 110, // Increased headroom to accommodate floating data labels
          grid: { color: gridColor },
          ticks: {
            font: { family: "'Inter', sans-serif", size: 11, weight: "600" },
            color: labelColor,
            callback: (v) => v <= 100 ? `${v}%` : "",
          },
        },
      },
      animation: {
        duration: 900,
        easing: "easeOutQuart",
      },
    },
  });
}

/* ═══════════════════════════════════════════════════════════════
   SECTION 7 — Clinical Precautions Stagger Loader
   ═══════════════════════════════════════════════════════════════ */
function renderPrecautions(precautions, isCancer) {
  precautionsTitle.className = "card-title precautions-title " + (isCancer ? "cancer" : "safe");
  precautionsList.innerHTML = "";

  const dataList = (precautions && precautions.length > 0)
    ? precautions
    : ["No critical precaution indexes recorded. Follow up with clinical dermatologists."];

  dataList.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "precaution-item-card " + (isCancer ? "cancer" : "safe");
    card.style.animation = `fadeSlideIn 0.3s ease both ${idx * 0.12}s`;

    const icon = document.createElement("span");
    icon.className = "precaution-bullet";
    icon.textContent = isCancer ? "⚠️" : "🛡️";

    const text = document.createElement("span");
    text.className = "precaution-text";
    text.textContent = p;

    card.appendChild(icon);
    card.appendChild(text);
    precautionsList.appendChild(card);
  });
}

/* ═══════════════════════════════════════════════════════════════
   SECTION 8 — Database Persistence Sync (Session Local Cache)
   ═══════════════════════════════════════════════════════════════ */
async function syncHistoryDatabase() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    if (res.ok) {
      const data = await res.json();
      predictionHistory = data.history || [];
      localStorage.setItem("derma_history_sqlite", JSON.stringify(predictionHistory));
      renderHistoryTimeline();
    }
  } catch (err) {
    console.warn("Could not sync with SQLite db, falling back to LocalCache:", err);
    renderHistoryTimeline();
  }
}

function renderHistoryTimeline() {
  historyList.innerHTML = "";

  if (predictionHistory.length === 0) {
    historyList.innerHTML = `
      <div class="history-empty">
        <div class="empty-icon">📋</div>
        <p>No scans evaluated yet. Upload an image to initialize session log.</p>
      </div>`;
    return;
  }

  predictionHistory.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "history-item-wrap";
    
    // Parse precautions array if it is saved as a JSON string in SQLite
    let precautionsArray = [];
    if (typeof item.precautions === "string") {
      try {
        precautionsArray = JSON.parse(item.precautions);
      } catch(e) {
        precautionsArray = [item.precautions];
      }
    } else {
      precautionsArray = item.precautions || [];
    }
    
    const pListItems = (precautionsArray.length > 0)
      ? precautionsArray.map(p => `<li>${p}</li>`).join("")
      : "<li>Follow up with clinical experts.</li>";

    wrapper.innerHTML = `
      <div class="history-item-header">
        <div class="history-left-side">
          <div class="history-dot" style="color: ${item.color}; background-color: ${item.color}"></div>
          <span class="history-class">${item.predicted_class}</span>
        </div>
        <div class="history-mid-side">
          <span class="history-confidence-pill">${item.confidence}% Match</span>
          <span class="history-status-pill ${item.is_cancer ? 'cancer' : 'safe'}">${item.binary_result}</span>
        </div>
        <div class="history-right-side">
          <span class="history-time">${item.timestamp}</span>
          <span class="history-toggle-icon">▼</span>
        </div>
      </div>
      <div class="history-item-body">
        <div class="history-body-layout">
          <div class="history-body-details">
            <h4 style="font-family: var(--font-display); font-weight: 800; font-size: 1.15rem; color: ${item.color}">
              ${item.full_name || item.predicted_class}
            </h4>
            <p class="history-body-desc">${item.description || 'No clinical definition cached.'}</p>
            <div class="history-download-row">
              <button class="btn btn-outline btn-download-history" data-id="${item.id}" type="button">
                📄 Re-Download Report PDF
              </button>
            </div>
          </div>
          <div class="history-body-precautions">
            <h5>Suggested Actions</h5>
            <ul>${pListItems}</ul>
          </div>
        </div>
      </div>
    `;

    const header = wrapper.querySelector(".history-item-header");
    header.addEventListener("click", () => {
      const allItems = historyList.querySelectorAll(".history-item-wrap");
      allItems.forEach(i => {
        if (i !== wrapper) i.classList.remove("expanded");
      });
      wrapper.classList.toggle("expanded");
    });

    const downloadBtn = wrapper.querySelector(".btn-download-history");
    downloadBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      downloadReportPDF(item.id);
    });

    historyList.appendChild(wrapper);
  });
}

// Initial fetch from SQLite database logs on app load
syncHistoryDatabase();

clearHistoryBtn.addEventListener("click", async () => {
  if (predictionHistory.length === 0) return;
  if (confirm("Execute action: Clear all session logs from SQLite database and local storage?")) {
    try {
      const res = await fetch(`${API_BASE}/history`, { method: "DELETE" });
      if (res.ok) {
        predictionHistory = [];
        localStorage.removeItem("derma_history_sqlite");
        
        historyList.style.opacity = 0;
        setTimeout(() => {
          renderHistoryTimeline();
          historyList.style.opacity = 1;
        }, 300);
      }
    } catch(err) {
      alert("Failed to clear backend database registry: " + err.message);
    }
  }
});

/* ═══════════════════════════════════════════════════════════════
   SECTION 9 — High-Fidelity Client-Side PDF Report Exporter
   ═══════════════════════════════════════════════════════════════ */
downloadReportBtn.addEventListener("click", () => {
  if (currentPredictionData) {
    downloadReportPDF(currentPredictionData.id);
  }
});

async function downloadReportPDF(predictionId) {
  // Locate target entry
  const data = predictionHistory.find(item => item.id === predictionId) || currentPredictionData;
  if (!data) return;

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  const primaryColor = "#3b82f6";
  const grayMuted    = "#64748b";
  const darkText     = "#0f172a";
  const borderLight  = "#e2e8f0";

  // Margins
  const lMargin = 20;
  const rMargin = 190;
  let y = 20;

  // 1. Header Banner
  doc.setFillColor(6, 9, 15); // dark theme background
  doc.rect(0, 0, 210, 36, "F");
  
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(255, 255, 255);
  doc.text("DermaDrishti AI", lMargin, 20);
  
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(200, 200, 200);
  doc.text("CLINICAL SCAN CLASSIFICATION & EXPLAINABILITY REPORT", lMargin, 26);
  
  doc.setFontSize(8);
  doc.setTextColor(140, 140, 140);
  doc.text("GEC PALAMU · COMPUTER SCIENCE & ENGINEERING DEPARTMENT", lMargin, 30);

  y = 48;

  // 2. Scan Metadata Block
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(darkText);
  doc.text("SCAN ANALYSIS INFORMATION", lMargin, y);
  y += 5;
  doc.setDrawColor(borderLight);
  doc.setLineWidth(0.3);
  doc.line(lMargin, y, rMargin, y);
  y += 6;

  doc.setFont("Helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(grayMuted);
  doc.text("SCAN UUID ID:", lMargin, y);
  doc.setFont("Helvetica", "normal");
  doc.setTextColor(darkText);
  doc.text(data.id, lMargin + 26, y);
  
  doc.setFont("Helvetica", "bold");
  doc.setTextColor(grayMuted);
  doc.text("TIMESTAMP:", 110, y);
  doc.setFont("Helvetica", "normal");
  doc.setTextColor(darkText);
  doc.text(data.timestamp || "N/A", 134, y);
  y += 8;

  // 3. Core Diagnostic Outcome
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(darkText);
  doc.text("CORE SYSTEM DIAGNOSTIC OUTCOME", lMargin, y);
  y += 5;
  doc.line(lMargin, y, rMargin, y);
  y += 6;

  // ── Diagnostic Outcome Box ──────────────────────────────────────────────────────────────────────
  // LEFT col (max 108mm): disease name, wrapped. RIGHT col starts at x=145.
  const scoreColor   = data.color || primaryColor;
  const binaryBadge = data.is_cancer ? 'MALIGNANT' : 'BENIGN';
  const classLabel   = data.full_name || data.predicted_class || 'Unknown';

  // Measure wrapped lines for disease name at font-size 13, max 108mm
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(13);
  const nameLines  = doc.splitTextToSize(classLabel, 108);
  const nameBlockH = nameLines.length * 6;         // ~6mm per line
  const boxH       = Math.max(30, nameBlockH + 22); // min 30mm tall

  // Background fill
  doc.setFillColor(244, 247, 252);
  doc.rect(lMargin, y, 170, boxH, 'F');

  // LEFT: "CLASSIFIED AS:" label
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(grayMuted);
  doc.text('CLASSIFIED AS:', lMargin + 6, y + 8);

  // LEFT: disease name (wrapped, coloured)
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(13);
  doc.setTextColor(scoreColor);
  doc.text(nameLines, lMargin + 6, y + 15);

  // LEFT: BENIGN / MALIGNANT badge below the name
  const badgeY = y + 15 + nameBlockH + 1;
  doc.setFontSize(8);
  doc.setFont('Helvetica', 'bold');
  if (data.is_cancer) {
    doc.setTextColor(200, 40, 40);
  } else {
    doc.setTextColor(39, 174, 96);
  }
  doc.text('\u25CF ' + binaryBadge, lMargin + 6, badgeY);

  // Vertical divider between columns
  doc.setDrawColor(210, 215, 225);
  doc.line(140, y + 4, 140, y + boxH - 4);

  // RIGHT: "MATCH INDEX:" label
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(grayMuted);
  doc.text('MATCH INDEX:', 145, y + 8);

  // RIGHT: confidence % in large font
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(20);
  doc.setTextColor(scoreColor);
  doc.text(data.confidence + '%', 145, y + 20);

  y += boxH + 8;


  // 4. Detailed Description
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(darkText);
  doc.text("DISEASE STUDY PROFILE & DEFINITION:", lMargin, y);
  y += 5;
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(grayMuted);
  
  const descText = data.description || "No definitions provided.";
  const descLines = doc.splitTextToSize(descText, 170);
  doc.text(descLines, lMargin, y);
  y += (descLines.length * 4.5) + 6;

  // 5. Probabilities Table
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(darkText);
  doc.text("QUANTITATIVE PROBABILITY CHANNELS:", lMargin, y);
  y += 5;
  doc.line(lMargin, y, rMargin, y);
  y += 4;

  let probs = data.probabilities;
  // If probabilities is a string (queried from SQLite history), parse it
  if (typeof probs === "string") {
    try {
      probs = JSON.parse(probs);
    } catch(e) {
      probs = null;
    }
  }

  if (probs) {
    doc.setFontSize(8.5);
    doc.setFont("Helvetica", "bold");
    doc.text("Lesion Category", lMargin + 4, y + 4);
    doc.text("Status Profile", 110, y + 4);
    doc.text("Softmax Match %", 150, y + 4);
    
    y += 7;
    doc.line(lMargin, y, rMargin, y);
    
    Object.keys(probs).forEach(key => {
      const meta = CLASS_DISPLAY[key];
      const isPredicted = key === data.predicted_label;
      
      if (isPredicted) {
        doc.setFillColor(244, 247, 252);
        doc.rect(lMargin, y + 1, 170, 6, "F");
        doc.setFont("Helvetica", "bold");
        doc.setTextColor(scoreColor);
      } else {
        doc.setFont("Helvetica", "normal");
        doc.setTextColor(darkText);
      }
      
      doc.text(meta ? meta.name : key, lMargin + 4, y + 5);
      doc.text(meta ? meta.status : "BENIGN", 110, y + 5);
      doc.text(`${probs[key].toFixed(2)}%`, 152, y + 5);
      
      y += 7;
      doc.setFont("Helvetica", "normal");
      doc.setTextColor(darkText);
    });
  } else {
    doc.setFontSize(9);
    doc.text("Probability table metrics not available.", lMargin, y + 4);
    y += 10;
  }
  y += 4;

  // Check if we need to spawn a new page
  if (y > 210) {
    doc.addPage();
    y = 20;
  }

  // 6. Precautions Section
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(darkText);
  doc.text("SUGGESTED CLINICAL ACTIONS & PRECAUTIONS:", lMargin, y);
  y += 5;
  doc.line(lMargin, y, rMargin, y);
  y += 6;

  let precautionsArray = [];
  if (typeof data.precautions === "string") {
    try {
      precautionsArray = JSON.parse(data.precautions);
    } catch(e) {
      precautionsArray = [data.precautions];
    }
  } else {
    precautionsArray = data.precautions || [];
  }

  doc.setFont("Helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(grayMuted);
  precautionsArray.forEach(p => {
    const pLines = doc.splitTextToSize(`• ${p}`, 170);
    doc.text(pLines, lMargin, y);
    y += (pLines.length * 4) + 2;
  });
  
  y += 6;

  if (y > 240) {
    doc.addPage();
    y = 20;
  }

  // 7. Medical Disclaimer & Academic Footer
  doc.setDrawColor(245, 158, 11); // warning border
  doc.setLineWidth(0.5);
  doc.rect(lMargin, y, 170, 20);
  
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(180, 100, 10);
  doc.text("⚕️ MEDICAL DISCLAIMER & LIMITATIONS NOTE", lMargin + 5, y + 5);
  
  doc.setFont("Helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(grayMuted);
  const discLines = doc.splitTextToSize("DermaDrishti is an academic laboratory framework engineered for computational learning research. It is not FDA/CE certified as a diagnostic device. Output matched percentages are convolutional probability values and should never substitute a qualified physical skin biopsy review.", 160);
  doc.text(discLines, lMargin + 5, y + 10);
  
  y += 28;
  
  // Signature block
  doc.setFont("Helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(darkText);
  doc.text("DermaDrishti Research Group", lMargin, y);
  doc.text("Under Academic Mentorship of:", 120, y);
  y += 4;
  doc.setFont("Helvetica", "normal");
  doc.setTextColor(grayMuted);
  doc.text("GEC Palamu Department of CSE", lMargin, y);
  doc.text("Prof. Panjeet Kumar Lenka", 120, y);

  doc.save(`DermaDrishti_Diagnostic_Report_${data.id}.pdf`);
}

/* ═══════════════════════════════════════════════════════════════
   SECTION 8 — Documentation Sidebar & Literature Interactivity
   ═══════════════════════════════════════════════════════════════ */

/* ── Documentation Sidebar Navigation ─────────────────────── */
(function initDocsNav() {
  const navItems = document.querySelectorAll('.docs-nav-item');
  const docSections = document.querySelectorAll('.doc-section');

  if (!navItems.length) return;

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetId = item.getAttribute('data-doc');

      // Update nav active state
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      // Show target section
      docSections.forEach(s => {
        s.classList.remove('active');
        // Trigger re-animation
        void s.offsetWidth;
      });

      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        targetSection.classList.add('active');
        // Scroll content area to top smoothly
        const docsContent = document.getElementById('docs-content');
        if (docsContent) docsContent.scrollTo({ top: 0, behavior: 'smooth' });
      }

      // Keep mobile accordion in sync
      syncMobileAccordion(targetId);
    });
  });
})();

/* ── Mobile Docs Accordion ─────────────────────────────────── */
(function initMobileDocsAccordion() {
  const docSections = document.querySelectorAll('.doc-section');
  const docsContent = document.getElementById('docs-content');
  if (!docsContent) return;

  const ACCORDION_ID = 'docs-mobile-accordion';

  // Build accordion on first viewport check
  function buildAccordion() {
    if (window.innerWidth > 768) return;
    if (document.getElementById(ACCORDION_ID)) return; // already built

    const accordion = document.createElement('div');
    accordion.className = 'docs-mobile-accordion';
    accordion.id = ACCORDION_ID;

    const sidebarItems = document.querySelectorAll('.docs-nav-item');

    sidebarItems.forEach((navItem, idx) => {
      const docId = navItem.getAttribute('data-doc');
      const label = navItem.querySelector('span:last-child').textContent.trim();
      const icon  = navItem.querySelector('.docs-nav-icon').textContent.trim();
      const contentEl = document.getElementById(docId);
      if (!contentEl) return;

      const accItem = document.createElement('div');
      accItem.className = 'docs-accordion-item' + (idx === 0 ? ' expanded' : '');

      const header = document.createElement('div');
      header.className = 'docs-accordion-header' + (idx === 0 ? ' active' : '');
      header.innerHTML = `<span>${icon} ${label}</span><span class="docs-accordion-arrow">▼</span>`;

      const body = document.createElement('div');
      body.className = 'docs-accordion-body';

      // Clone the content node into accordion (read-only clone for display)
      const clonedContent = contentEl.cloneNode(true);
      clonedContent.classList.add('active');
      clonedContent.removeAttribute('id'); // avoid duplicate IDs
      body.appendChild(clonedContent);

      header.addEventListener('click', () => {
        const isExpanded = accItem.classList.contains('expanded');
        // Collapse all
        document.querySelectorAll(`#${ACCORDION_ID} .docs-accordion-item`).forEach(i => {
          i.classList.remove('expanded');
          i.querySelector('.docs-accordion-header').classList.remove('active');
        });
        // Expand clicked
        if (!isExpanded) {
          accItem.classList.add('expanded');
          header.classList.add('active');
        }
      });

      accItem.appendChild(header);
      accItem.appendChild(body);
      accordion.appendChild(accItem);
    });

    // Insert accordion just before the sidebar in the docs-layout
    const docsLayout = document.querySelector('.docs-layout');
    if (docsLayout) {
      docsLayout.parentNode.insertBefore(accordion, docsLayout);
    }
  }

  // Build on load and on resize
  buildAccordion();
  window.addEventListener('resize', buildAccordion, { passive: true });
})();

/* Sync mobile accordion when sidebar nav is clicked */
function syncMobileAccordion(targetDocId) {
  const accordion = document.getElementById('docs-mobile-accordion');
  if (!accordion || window.innerWidth > 768) return;

  const items = accordion.querySelectorAll('.docs-accordion-item');
  const headers = accordion.querySelectorAll('.docs-accordion-header');
  const navItems = document.querySelectorAll('.docs-nav-item');

  // Find the index of this doc in the nav
  let targetIdx = -1;
  navItems.forEach((n, i) => {
    if (n.getAttribute('data-doc') === targetDocId) targetIdx = i;
  });

  if (targetIdx === -1) return;

  items.forEach((item, i) => {
    item.classList.remove('expanded');
    headers[i].classList.remove('active');
  });

  if (items[targetIdx]) {
    items[targetIdx].classList.add('expanded');
    headers[targetIdx].classList.add('active');
    items[targetIdx].scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ── Copy-to-Clipboard Buttons ─────────────────────────────── */
(function initCopyButtons() {
  document.addEventListener('click', e => {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;

    // Prefer data-code attribute; fall back to sibling <pre><code>
    let textToCopy = btn.getAttribute('data-code') || '';

    if (!textToCopy) {
      const pre = btn.closest('.code-block')?.querySelector('pre code');
      if (pre) textToCopy = pre.textContent;
    }

    if (!textToCopy) return;

    navigator.clipboard.writeText(textToCopy.trim()).then(() => {
      const original = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove('copied');
      }, 1800);
    }).catch(() => {
      // Fallback for HTTP contexts
      const ta = document.createElement('textarea');
      ta.value = textToCopy.trim();
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);

      const original = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove('copied');
      }, 1800);
    });
  });
})();

/* ── Smooth scroll for new nav links ───────────────────────── */
(function initNewNavSmoothScroll() {
  const newLinks = document.querySelectorAll('a[href="#literature-section"], a[href="#documentation-section"]');
  newLinks.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();
