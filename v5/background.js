// Service worker: orquestra captura, offscreen, estado e o download do .txt.

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== "background") return;

  if (msg.type === "start-recording") {
    startRecording(msg.tabId, msg.tabTitle, msg.summaryEnabled).then(
      () => sendResponse({ ok: true }),
      (err) => sendResponse({ ok: false, error: err && err.message ? err.message : String(err) })
    );
    return true; // resposta assincrona
  }

  if (msg.type === "stop-recording") {
    chrome.runtime.sendMessage({ target: "offscreen", type: "offscreen-stop" }).catch(() => {});
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "transcribe-status") {
    setTranscribing(msg.label);
    return false;
  }

  if (msg.type === "download-transcript") {
    downloadResults(msg.text, msg.summary).finally(async () => {
      await closeOffscreen();
      await chrome.storage.local.remove("state");
    });
    return false;
  }
});

async function startRecording(tabId, tabTitle, summaryEnabled) {
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
  await ensureOffscreen();
  // O offscreen pode demorar um instante para registrar o listener: tenta ate responder.
  await sendToOffscreen({
    target: "offscreen",
    type: "offscreen-start",
    streamId,
    summaryEnabled: summaryEnabled !== false,
  });
  await chrome.storage.local.set({
    state: { phase: "recording", tabTitle, startedAt: Date.now() },
  });
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function sendToOffscreen(payload) {
  let lastErr;
  for (let i = 0; i < 30; i++) {
    try { return await chrome.runtime.sendMessage(payload); }
    catch (e) { lastErr = e; await sleep(100); }
  }
  throw lastErr || new Error("offscreen nao respondeu a tempo");
}

async function setTranscribing(label) {
  const { state } = await chrome.storage.local.get("state");
  await chrome.storage.local.set({
    state: { phase: "transcribing", tabTitle: (state && state.tabTitle) || "", label: label || "Transcrevendo..." },
  });
}

async function downloadResults(text, summary) {
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  await downloadText("transcricao-" + stamp + ".txt", text || "(vazio)");
  if (summary) await downloadText("resumo-" + stamp + ".txt", summary);
}

async function downloadText(filename, text) {
  const url = "data:text/plain;charset=utf-8," + encodeURIComponent(text);
  await chrome.downloads.download({ url, filename, saveAs: false });
}

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Gravar e transcrever o audio da aba selecionada.",
  });
}

async function closeOffscreen() {
  if (await hasOffscreen()) await chrome.offscreen.closeDocument();
}
