// v2: STT = Moonshine-tiny | Diarização = wespeaker-resnet34
// Moonshine é encoder-decoder focado em inglês, ~5x mais rápido que whisper-base.
// Não aceita os parâmetros `language` e `task` (específicos do Whisper).

let recorder = null;
let chunks = [];
let audioContext = null;
let stream = null;
let transcriber = null;
let summarizer = null;
let libPromise = null;
let summaryEnabled = true;

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== "offscreen") return;
  if (msg.type === "offscreen-start") {
    summaryEnabled = msg.summaryEnabled !== false;
    start(msg.streamId);
    sendResponse({ ok: true });
  }
  if (msg.type === "offscreen-stop") { stop(); sendResponse({ ok: true }); }
  return false;
});

function loadLib() {
  if (!libPromise) {
    libPromise = import("./lib/transformers.min.js").then((mod) => {
      const env = mod.env;
      env.allowLocalModels = false;
      env.allowRemoteModels = true;
      env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("lib/");
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
      return mod;
    });
  }
  return libPromise;
}

const MODEL = "onnx-community/moonshine-tiny-ONNX";
const SUMMARY_MODEL = "onnx-community/gemma-3-270m-it-ONNX";

async function start(streamId) {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
    });
  } catch (e) {
    console.error("getUserMedia falhou:", e);
    return;
  }

  audioContext = new AudioContext();
  audioContext.createMediaStreamSource(stream).connect(audioContext.destination);

  chunks = [];
  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";
  recorder = new MediaRecorder(stream, { mimeType });
  recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
  recorder.onstop = finalize;
  recorder.start();
}

function stop() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

async function finalize() {
  const blob = new Blob(chunks, { type: "audio/webm" });
  chunks = [];
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioContext) { try { await audioContext.close(); } catch (_) {} }

  try {
    status("Preparando o audio...");
    const audio = await decodeTo16kMono(blob);

    let turns = [];
    try {
      turns = await diarizeAudio(audio);
    } catch (e) {
      console.error("Falha na diarizacao:", e);
    }

    const { pipeline } = await loadLib();

    if (!transcriber) {
      const device = navigator.gpu ? "webgpu" : "wasm";
      const dtype = device === "webgpu" ? "fp32" : "q8";
      transcriber = await pipeline("automatic-speech-recognition", MODEL, {
        device,
        dtype,
        progress_callback: (p) => {
          if (p.status === "progress") status("Baixando modelo... " + Math.round(p.progress || 0) + "%");
        },
      });
    }

    status("Transcrevendo...");
    // Moonshine não usa `language` nem `task` (Whisper-specific).
    const out = await transcriber(audio, {
      return_timestamps: true,
      chunk_length_s: 30,
      stride_length_s: 5,
    });

    const text = buildTranscriptText(out, turns, audio.length / 16000);

    let summary = "";
    if (summaryEnabled) {
      try {
        summary = await summarize(text);
      } catch (e) {
        console.error("Falha no resumo:", e);
        summary = "[erro ao gerar resumo: " + (e && e.message ? e.message : e) + "]";
      }
    }

    send({ type: "download-transcript", text, summary });
  } catch (e) {
    console.error("Falha na transcricao:", e);
    send({ type: "download-transcript", text: "[erro na transcricao: " + (e && e.message ? e.message : e) + "]", summary: "" });
  }
}

async function diarizeAudio(audio) {
  status("Carregando diarizacao...");
  const { diarize } = await import("./lib/sherpa/diarization.js");
  status("Identificando falantes...");
  return diarize(audio, (label) => status("Carregando diarizacao: " + label));
}

function buildTranscriptText(out, turns, durationSec) {
  const chunks = (out && out.chunks) || [];
  const fallback = ((out && out.text) || "").trim();
  if (chunks.length === 0 || !turns || turns.length === 0) return fallback;

  const linhas = chunks
    .map((c) => {
      const texto = (c.text || "").trim();
      if (!texto) return null;
      const ini = c.timestamp[0] || 0;
      const fim = c.timestamp[1] == null ? durationSec : c.timestamp[1];
      return `[${fmtTempo(ini)}] ${speakerFor(ini, fim, turns)}: ${texto}`;
    })
    .filter(Boolean);

  return linhas.length ? linhas.join("\n") : fallback;
}

function speakerFor(ini, fim, turns) {
  let best = "Falante ?", bestOverlap = 0;
  for (const t of turns) {
    const ov = Math.min(fim, t.end) - Math.max(ini, t.start);
    if (ov > bestOverlap) { bestOverlap = ov; best = t.speaker; }
  }
  return best;
}

function fmtTempo(s) {
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(Math.floor(s % 60)).padStart(2, "0");
  return mm + ":" + ss;
}

async function summarize(text) {
  if (!text) return "";

  const { pipeline } = await loadLib();

  if (!summarizer) {
    status("Baixando modelo de resumo...");
    const device = navigator.gpu ? "webgpu" : "wasm";
    const dtype = device === "webgpu" ? "fp32" : "q8";
    summarizer = await pipeline("text-generation", SUMMARY_MODEL, {
      device,
      dtype,
      progress_callback: (p) => {
        if (p.status === "progress") status("Baixando modelo de resumo... " + Math.round(p.progress || 0) + "%");
      },
    });
  }

  status("Gerando resumo...");
  const messages = [
    {
      role: "system",
      content:
        "Voce resume transcricoes de reunioes em portugues do Brasil. " +
        "Seja conciso e use apenas o conteudo da transcricao.",
    },
    {
      role: "user",
      content:
        "Resuma a transcricao abaixo em topicos curtos, com secoes " +
        "'Principais pontos', 'Decisoes' e 'Acoes' (liste 'nenhuma' se nao houver).\n\n" +
        "Transcricao:\n" + text,
    },
  ];

  const out = await summarizer(messages, { max_new_tokens: 512, do_sample: false });
  const generated = out && out[0] && out[0].generated_text;
  if (!Array.isArray(generated)) return "";
  const reply = generated[generated.length - 1];
  return ((reply && reply.content) || "").trim();
}

async function decodeTo16kMono(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 16000, 16000);
  const ab = await ctx.decodeAudioData(buf);
  if (ab.numberOfChannels > 1) {
    const l = ab.getChannelData(0);
    const r = ab.getChannelData(1);
    const m = new Float32Array(l.length);
    for (let i = 0; i < l.length; i++) m[i] = (l[i] + r[i]) / 2;
    return m;
  }
  return ab.getChannelData(0);
}

function status(label) { send({ type: "transcribe-status", label: label }); }
function send(payload) { chrome.runtime.sendMessage(Object.assign({ target: "background" }, payload)); }
