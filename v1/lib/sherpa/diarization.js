// Wrapper para a diarizacao com sherpa-onnx (WASM), rodando inteiramente no
// browser, antes da transcricao com Whisper.
//
// Modelos usados:
// - Segmentacao: pyannote-segmentation-3.0 (embutido no .data do sherpa-onnx)
// - Embedding de locutor: wespeaker (en_voxceleb_resnet34_LM), carregado a
//   parte e injetado no sistema de arquivos virtual do WASM.
//
// sherpa-onnx-wasm-main-speaker-diarization.js e um script classico (nao
// ESM) que espera um global `Module`, entao e carregado via <script> no
// documento do offscreen.

const BASE = chrome.runtime.getURL("lib/sherpa/");

let sdPromise = null;

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = url;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("falha ao carregar " + url));
    document.head.appendChild(s);
  });
}

async function init(onProgress) {
  if (!sdPromise) {
    sdPromise = (async () => {
      window.Module = window.Module || {};
      window.Module.locateFile = (path) => BASE + path;
      if (onProgress) window.Module.setStatus = (text) => { if (text) onProgress(text); };

      const ready = new Promise((resolve) => { window.Module.onRuntimeInitialized = resolve; });

      await loadScript(BASE + "sherpa-onnx-speaker-diarization.js");
      await loadScript(BASE + "sherpa-onnx-wasm-main-speaker-diarization.js");
      await ready;

      const Module = window.Module;

      const wespeakerBuf = await fetch(BASE + "wespeaker_en_voxceleb_resnet34_LM.onnx").then((r) => r.arrayBuffer());
      Module.FS_createDataFile("/", "wespeaker.onnx", new Uint8Array(wespeakerBuf), true, true, true);

      return window.createOfflineSpeakerDiarization(Module, {
        segmentation: { pyannote: { model: "./segmentation.onnx" } },
        embedding: { model: "./wespeaker.onnx" },
        clustering: { numClusters: -1, threshold: 0.5 },
        minDurationOn: 0.3,
        minDurationOff: 0.5,
      });
    })();
  }
  return sdPromise;
}

// audio: Float32Array mono na mesma taxa de amostragem esperada pelo modelo
// (sd.sampleRate, 16 kHz para o pyannote-segmentation-3.0).
// Retorna turns = [{ start, end, speaker }] ordenados por inicio.
export async function diarize(audio, onProgress) {
  const sd = await init(onProgress);
  const segments = sd.process(audio) || [];
  return segments.map((seg) => ({
    start: seg.start,
    end: seg.end,
    speaker: "Falante " + (seg.speaker + 1),
  }));
}

export async function getSampleRate(onProgress) {
  const sd = await init(onProgress);
  return sd.sampleRate;
}
