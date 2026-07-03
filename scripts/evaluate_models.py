# Avalia as combinacoes de modelos STT+diarizacao usadas em cada versao da
# extensao (v1-v5), sem precisar do Chrome: roda os mesmos pesos ONNX
# localmente via onnxruntime/sherpa-onnx.
#
# Uso:
#   scripts/.venv/Scripts/python.exe scripts/evaluate_models.py <pasta-com-videos> [--versions=1,2,3]
#
# Para cada video e cada versao, salva:
#   calls/vN/<nome-do-video>.txt   (transcricao diarizada "[mm:ss] Falante N: ...")
#   calls/vN/metadata.json         (tempo de diarizacao e de transcricao por video)
#
# Arquitetura: diariza o audio inteiro primeiro, depois transcreve cada
# segmento de fala (turn) separadamente. Isso evita ter que reimplementar o
# alinhamento chunk-a-chunk que o transformers.js faz no browser, e trata os
# modelos de STT de forma uniforme (so muda o audio recortado que entra).

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CALLS_DIR = ROOT_DIR / "calls"
CACHE_DIR = ROOT_DIR / "scripts" / ".cache"
SEG_MODEL = ROOT_DIR / "scripts" / ".models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"

# As DLLs de CUDA/cuDNN vem dos pacotes pip nvidia-cublas-cu13/nvidia-cudnn-cu13
# (nao ha CUDA Toolkit instalado no sistema); o onnxruntime-gpu nao acha essas
# DLLs sozinho no Windows, precisa que o processo registre o diretorio antes
# de importar onnxruntime.
if sys.platform == "win32":
    _site_packages = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
    _cuda_dirs = []
    for _rel in ["nvidia/cu13/bin/x86_64", "nvidia/cudnn/bin"]:
        _dir = _site_packages / _rel
        if _dir.exists():
            _cuda_dirs.append(str(_dir))
            os.add_dll_directory(str(_dir))
    if _cuda_dirs:
        # onnxruntime_providers_cuda.dll carrega dependencias via LoadLibrary
        # classico, que so olha o PATH — add_dll_directory sozinho nao basta.
        os.environ["PATH"] = os.pathsep.join(_cuda_dirs) + os.pathsep + os.environ["PATH"]

import numpy as np
import onnxruntime as ort
import soundfile as sf
import sherpa_onnx
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

# A CUDAExecutionProvider carrega (DLLs ok), mas trava na execucao real dos
# kernels de convolucao nesta GPU (RTX 5070 Ti / Blackwell) com o
# onnxruntime-gpu/cuDNN disponiveis via pip hoje (CUDNN_STATUS_EXECUTION_FAILED).
# Forcando CPU ate isso ser resolvido rio-acima.
STT_PROVIDER = "CPUExecutionProvider"
DIAR_PROVIDER = "cpu"
print(f"Rodando em CPU (provider STT={STT_PROVIDER}, diarizacao={DIAR_PROVIDER})")

STT_CONFIGS = {
    "whisper-base": {
        "model_id": "Xenova/whisper-base",
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model_merged.onnx",
    },
    "whisper-base-quantized": {
        "model_id": "Xenova/whisper-base",
        "encoder": "encoder_model_quantized.onnx",
        "decoder": "decoder_model_merged_quantized.onnx",
    },
    "whisper-tiny": {
        "model_id": "Xenova/whisper-tiny",
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model_merged.onnx",
    },
}
EMBEDDING_MODELS = {
    "wespeaker": ROOT_DIR / "v1" / "lib" / "sherpa" / "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "campplus": ROOT_DIR / "v4" / "lib" / "sherpa" / "cam_pp.onnx",
}
COMBOS = {
    1: {"stt": "whisper-base", "embedding": "wespeaker"},
    2: {"stt": "whisper-tiny", "embedding": "wespeaker"},
    3: {"stt": "whisper-base-quantized", "embedding": "wespeaker"},
    4: {"stt": "whisper-tiny", "embedding": "campplus"},
    5: {"stt": "whisper-base-quantized", "embedding": "campplus"},
}


def fmt_time(s):
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def extract_wav(video_path: Path) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = CACHE_DIR / f"{video_path.stem}.wav"
    if not wav_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)],
            check=True,
            capture_output=True,
        )
    return wav_path


def load_diarizer(embedding_path: Path):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            provider=DIAR_PROVIDER,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_path), provider=DIAR_PROVIDER),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def load_stt(stt_key):
    cfg = STT_CONFIGS[stt_key]
    tokenizer = AutoProcessor.from_pretrained(cfg["model_id"])
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        cfg["model_id"],
        subfolder="onnx",
        encoder_file_name=cfg["encoder"],
        decoder_file_name=cfg["decoder"],
        provider=STT_PROVIDER,
    )
    model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
        language="portuguese", task="transcribe"
    )
    return {"id": cfg["model_id"], "tokenizer": tokenizer, "model": model}


def transcribe_segment(stt, audio_16k: np.ndarray) -> str:
    max_new_tokens = max(16, min(400, int(len(audio_16k) / 16000 * 10) + 8))
    inputs = stt["tokenizer"](audio_16k, sampling_rate=16000, return_tensors="pt")
    gen = stt["model"].generate(**inputs, max_new_tokens=max_new_tokens)
    text = stt["tokenizer"].batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()


def process_video(video_path: Path, diarizer, stt):
    wav_path = extract_wav(video_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000

    t0 = time.time()
    segments = diarizer.process(audio).sort_by_start_time()
    diarization_ms = (time.time() - t0) * 1000

    lines = []
    t0 = time.time()
    for seg in segments:
        start_sample = int(seg.start * 16000)
        end_sample = int(seg.end * 16000)
        crop = audio[start_sample:end_sample]
        if len(crop) < 800:  # < 50ms, nada pra transcrever
            continue
        text = transcribe_segment(stt, crop)
        if text:
            lines.append(f"[{fmt_time(seg.start)}] Falante {seg.speaker + 1}: {text}")
    transcription_ms = (time.time() - t0) * 1000

    return {
        "text": "\n".join(lines),
        "duration_sec": len(audio) / 16000,
        "num_turns": len(segments),
        "diarization_ms": round(diarization_ms),
        "transcription_ms": round(transcription_ms),
        "total_ms": round(diarization_ms + transcription_ms),
    }


def run_version(n, video_files, diarizer_cache, stt_cache):
    combo = COMBOS[n]
    print(f"\n=== v{n}: stt={combo['stt']}  embedding={combo['embedding']} ===")

    if combo["embedding"] not in diarizer_cache:
        diarizer_cache[combo["embedding"]] = load_diarizer(EMBEDDING_MODELS[combo["embedding"]])
    diarizer = diarizer_cache[combo["embedding"]]

    if combo["stt"] not in stt_cache:
        print(f"[v{n}] carregando modelo {STT_CONFIGS[combo['stt']]['model_id']} ({combo['stt']})...")
        stt_cache[combo["stt"]] = load_stt(combo["stt"])
    stt = stt_cache[combo["stt"]]

    dest_dir = CALLS_DIR / f"v{n}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dest_dir / "metadata.json"
    results = []
    for video_path in video_files:
        print(f"[v{n}] processando {video_path.name}...")
        try:
            r = process_video(video_path, diarizer, stt)
            (dest_dir / f"{video_path.stem}.txt").write_text(r["text"] or "(vazio)", encoding="utf-8")
            print(
                f"[v{n}] {video_path.name} OK — "
                f"diarizacao={r['diarization_ms']/1000:.1f}s transcricao={r['transcription_ms']/1000:.1f}s "
                f"({r['num_turns']} turns)"
            )
            results.append({"video": video_path.name, **{k: v for k, v in r.items() if k != "text"}})
        except Exception as e:
            print(f"[v{n}] {video_path.name} FALHOU: {e}")
            results.append({"video": video_path.name, "error": str(e)})

        # Grava a cada video processado, nao so no final da versao.
        metadata_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v{n}] metadata.json salvo em {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir")
    parser.add_argument("--versions", default="1,2,3,4,5")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    video_files = sorted(
        p for p in videos_dir.iterdir() if p.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv")
    )
    if not video_files:
        print(f"nenhum video encontrado em {videos_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Videos encontrados ({len(video_files)}): {', '.join(v.name for v in video_files)}")

    versions = [int(v) for v in args.versions.split(",")]

    diarizer_cache = {}
    stt_cache = {}
    for n in versions:
        run_version(n, video_files, diarizer_cache, stt_cache)

    print("\nConcluido.")


if __name__ == "__main__":
    main()
