const credentialStatus = document.querySelector("#credentialStatus");
const mimoStatus = document.querySelector("#mimoStatus");
const hfStatus = document.querySelector("#hfStatus");
const settingsDialog = document.querySelector("#settingsDialog");
const settingsButton = document.querySelector("#settingsButton");
const closeSettings = document.querySelector("#closeSettings");
const saveSettings = document.querySelector("#saveSettings");
const settingsForm = document.querySelector("#settingsForm");
const settingsMessage = document.querySelector("#settingsMessage");
const jobForm = document.querySelector("#jobForm");
const audioInput = document.querySelector("#audioInput");
const dropzone = document.querySelector("#dropzone");
const fileName = document.querySelector("#fileName");
const formMessage = document.querySelector("#formMessage");
const startButton = document.querySelector("#startButton");
const logBox = document.querySelector("#logBox");
const jobState = document.querySelector("#jobState");
const results = document.querySelector("#results");
const pyannoteOptions = document.querySelector("#pyannoteOptions");
const speakerFixed = document.querySelectorAll(".speaker-fixed");
const speakerRange = document.querySelectorAll(".speaker-range");

let currentSettings = { mimoApiKeySet: false, hfTokenSet: false };
let pollTimer = null;

function setMessage(target, text, kind = "") {
  target.textContent = text;
  target.className = kind;
}

function renderSettingsStatus(settings) {
  currentSettings = settings;
  mimoStatus.textContent = `Xiaomi ASR API Key：${settings.mimoApiKeySet ? "已设置" : "未设置"}`;
  hfStatus.textContent = `Hugging Face Token：${settings.hfTokenSet ? "已设置" : "未设置"}`;
  credentialStatus.classList.toggle("warning", !settings.mimoApiKeySet);
}

async function loadSettings() {
  const response = await fetch("/api/settings");
  renderSettingsStatus(await response.json());
}

async function loadResults() {
  const response = await fetch("/api/results");
  const data = await response.json();
  results.innerHTML = "";
  if (!data.results.length) {
    results.innerHTML = '<p class="empty">暂无输出文件</p>';
    return;
  }
  for (const file of data.results) {
    const item = document.createElement("a");
    item.className = "result-item";
    item.href = `/download/${encodeURIComponent(file.name)}`;
    item.textContent = file.name;
    results.appendChild(item);
  }
}

settingsButton.addEventListener("click", () => {
  settingsForm.mimo_api_key.value = currentSettings.mimoApiKeySet ? "********" : "";
  settingsForm.hf_token.value = currentSettings.hfTokenSet ? "********" : "";
  setMessage(settingsMessage, "");
  settingsDialog.showModal();
});

closeSettings.addEventListener("click", () => settingsDialog.close());

saveSettings.addEventListener("click", async () => {
  const formData = new FormData(settingsForm);
  const response = await fetch("/api/settings", { method: "POST", body: formData });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    setMessage(settingsMessage, data.error || "保存失败", "error");
    return;
  }
  renderSettingsStatus(data);
  setMessage(settingsMessage, "已保存", "success");
  setTimeout(() => settingsDialog.close(), 500);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  if (event.dataTransfer.files.length) {
    audioInput.files = event.dataTransfer.files;
    fileName.textContent = event.dataTransfer.files[0].name;
  }
});

audioInput.addEventListener("change", () => {
  fileName.textContent = audioInput.files[0]?.name || "或点击选择本地音频";
});

function updateAnnotationUI() {
  const mode = jobForm.annotation_mode.value;
  const speakerMode = jobForm.speaker_mode.value;
  pyannoteOptions.hidden = mode !== "pyannote";
  speakerFixed.forEach((item) => {
    item.hidden = mode !== "pyannote" || speakerMode !== "fixed";
  });
  speakerRange.forEach((item) => {
    item.hidden = mode !== "pyannote" || speakerMode !== "range";
  });
}

jobForm.addEventListener("change", updateAnnotationUI);
updateAnnotationUI();

function validateBeforeStart() {
  const mode = jobForm.annotation_mode.value;
  if (!currentSettings.mimoApiKeySet) {
    return "请先在设置中填写 Xiaomi ASR API Key。";
  }
  if (mode === "pyannote" && !currentSettings.hfTokenSet) {
    return "选择 pyannote 方案时，请先在设置中填写 Hugging Face Token。";
  }
  if (!audioInput.files.length) {
    return "请先拖入或选择一个音频文件。";
  }
  return "";
}

async function pollJob(jobId) {
  const response = await fetch(`/api/status/${jobId}`);
  const data = await response.json();
  if (!response.ok || !data.ok) {
    setMessage(formMessage, data.error || "读取任务状态失败", "error");
    clearInterval(pollTimer);
    return;
  }
  jobState.textContent = data.status;
  logBox.textContent = data.logs.join("\n");
  logBox.scrollTop = logBox.scrollHeight;

  if (data.status === "done" || data.status === "failed") {
    clearInterval(pollTimer);
    startButton.disabled = false;
    setMessage(formMessage, data.status === "done" ? "处理完成" : data.error || "处理失败", data.status === "done" ? "success" : "error");
    await loadResults();
  }
}

jobForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage(formMessage, "");
  const validationError = validateBeforeStart();
  if (validationError) {
    setMessage(formMessage, validationError, "error");
    return;
  }
  const formData = new FormData(jobForm);
  startButton.disabled = true;
  jobState.textContent = "提交中";
  logBox.textContent = "";

  const response = await fetch("/api/start", { method: "POST", body: formData });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    startButton.disabled = false;
    setMessage(formMessage, data.error || "任务启动失败", "error");
    jobState.textContent = "空闲";
    return;
  }
  setMessage(formMessage, "任务已开始", "success");
  pollTimer = setInterval(() => pollJob(data.jobId), 1500);
  await pollJob(data.jobId);
});

loadSettings();
loadResults();
