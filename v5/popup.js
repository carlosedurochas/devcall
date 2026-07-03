const app = document.getElementById("app");
let timerInterval = null;

const manifest = chrome.runtime.getManifest();
document.getElementById("footer").textContent = manifest.name + " · v" + manifest.version;

// Re-renderiza sozinho quando o estado muda (gravando -> transcrevendo -> pronto).
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.state) render(changes.state.newValue);
});

init().catch(showError);

async function init() {
  const { state } = await chrome.storage.local.get("state");
  render(state);
}

function render(state) {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  if (!state) return renderTabPicker();
  if (state.phase === "recording") return renderRecording(state);
  if (state.phase === "transcribing") return renderTranscribing(state);
  renderTabPicker();
}

async function renderTabPicker() {
  const tabs = await chrome.tabs.query({});
  const capturable = tabs.filter((t) => /^(https?|file):/.test(t.url || ""));
  const { summaryEnabled } = await chrome.storage.local.get("summaryEnabled");
  let summaryOn = summaryEnabled !== false;

  app.innerHTML =
    '<h1>Transcrever audio de uma aba</h1>' +
    '<p class="hint">Escolha a aba cujo audio voce quer transcrever.</p>' +
    '<ul id="list"></ul>' +
    '<div class="toggle-row">' +
    '<span class="label">Gerar resumo</span>' +
    '<button id="summaryToggle" class="switch" type="button" role="switch"></button>' +
    '</div>' +
    '<button id="rec" class="rec" disabled>Gravar</button>';

  const list = document.getElementById("list");
  const recBtn = document.getElementById("rec");
  const summaryToggle = document.getElementById("summaryToggle");
  let selectedId = null;

  const renderToggle = () => {
    summaryToggle.classList.toggle("on", summaryOn);
    summaryToggle.setAttribute("aria-checked", String(summaryOn));
  };
  renderToggle();

  summaryToggle.addEventListener("click", async () => {
    summaryOn = !summaryOn;
    renderToggle();
    await chrome.storage.local.set({ summaryEnabled: summaryOn });
  });

  if (capturable.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nenhuma aba de pagina web aberta.";
    list.appendChild(li);
  }

  for (const tab of capturable) {
    const li = document.createElement("li");
    const icon = tab.favIconUrl
      ? Object.assign(document.createElement("img"), { src: tab.favIconUrl, alt: "" })
      : Object.assign(document.createElement("span"), { className: "ph" });
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = tab.title || tab.url;
    li.append(icon, title);
    li.addEventListener("click", () => {
      selectedId = tab.id;
      [...list.children].forEach((c) => c.classList.remove("selected"));
      li.classList.add("selected");
      recBtn.disabled = false;
    });
    list.appendChild(li);
  }

  recBtn.addEventListener("click", async () => {
    if (selectedId == null) return;
    recBtn.disabled = true;
    recBtn.textContent = "Iniciando...";
    const tab = capturable.find((t) => t.id === selectedId);
    const res = await chrome.runtime.sendMessage({
      target: "background",
      type: "start-recording",
      tabId: selectedId,
      tabTitle: tab.title || tab.url,
      summaryEnabled: summaryOn,
    });
    if (!res || !res.ok) {
      app.innerHTML =
        "<h1>Nao foi possivel gravar</h1>" +
        '<p class="hint">' + escapeHtml((res && res.error) || "Erro desconhecido") +
        ". Tente deixar a aba ativa e gravar de novo.</p>" +
        '<button id="back" class="stop">Voltar</button>';
      document.getElementById("back").addEventListener("click", renderTabPicker);
    }
    // Sucesso: o background grava o estado e o onChanged renderiza a tela de gravacao.
  });
}

function renderRecording(state) {
  app.innerHTML =
    '<div class="center">' +
    '<h1><span class="dot"></span>Gravando</h1>' +
    '<div class="timer" id="timer">00:00</div>' +
    '<div class="tab-name">' + escapeHtml(state.tabTitle || "") + "</div>" +
    '<button id="stop" class="stop">Finalizar e transcrever</button>' +
    "</div>";

  const started = state.startedAt || Date.now();
  const timerEl = document.getElementById("timer");
  const tick = () => {
    const s = Math.max(0, Math.floor((Date.now() - started) / 1000));
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    timerEl.textContent = mm + ":" + ss;
  };
  tick();
  timerInterval = setInterval(tick, 1000);

  document.getElementById("stop").addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ target: "background", type: "stop-recording" });
    // O onChanged levara para a tela de "Transcrevendo".
  });
}

function renderTranscribing(state) {
  app.innerHTML =
    '<div class="center">' +
    '<h1><span class="spin"></span>Processando</h1>' +
    '<div class="status" id="status">' + escapeHtml(state.label || "Transcrevendo...") + "</div>" +
    '<p class="hint">O .txt sera baixado automaticamente ao terminar. Pode fechar esta janela.</p>' +
    "</div>";
}

function showError(e) {
  app.innerHTML = "<h1>Erro ao abrir</h1><p class=\"hint\">" +
    escapeHtml((e && e.message) || String(e)) + "</p>";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
