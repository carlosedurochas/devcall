// Diarização com cam++ (3D-Speaker) como embedding de locutor.
//
// Modelo necessário (baixar e colocar em lib/sherpa/):
//   cam_pp.onnx
//   Fonte: https://github.com/k2-fsa/sherpa-onnx/releases
//          Procure por "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
//          e renomeie para cam_pp.onnx  (ou ajuste EMBEDDING_FILE abaixo).
//
// Segmentação: pyannote-segmentation-3.0 (embutida no .data do sherpa-onnx,
// mesma de v1 — não precisa de arquivo adicional).

const BASE = chrome.runtime.getURL("lib/sherpa/");
const EMBEDDING_FILE = "cam_pp.onnx";

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

      const embBuf = await fetch(BASE + EMBEDDING_FILE).then((r) => r.arrayBuffer());
      Module.FS_createDataFile("/", "embedding.onnx", new Uint8Array(embBuf), true, true, true);

      return window.createOfflineSpeakerDiarization(Module, {
        segmentation: { pyannote: { model: "./segmentation.onnx" } },
        embedding: { model: "./embedding.onnx" },
        clustering: { numClusters: -1, threshold: 0.5 },
        minDurationOn: 0.3,
        minDurationOff: 0.5,
      });
    })();
  }
  return sdPromise;
}

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
