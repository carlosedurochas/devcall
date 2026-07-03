# Avalia a qualidade das transcricoes geradas (calls/vN/<video>.txt) contra o
# texto de referencia escrito a mao (calls/transcriptions/<video>.txt), usando
# WER (Word Error Rate) e CER (Character Error Rate).
#
# Uso:
#   scripts/.venv/Scripts/python.exe scripts/evaluate_transcription_quality.py [--versions=1,2,3,4,5]

import argparse
import json
import re
from pathlib import Path

import jiwer

ROOT_DIR = Path(__file__).resolve().parent.parent
CALLS_DIR = ROOT_DIR / "calls"
REFERENCE_DIR = CALLS_DIR / "transcriptions"

# Normalizacao antes de comparar: minusculas, sem pontuacao, espacos colapsados.
TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

HYPOTHESIS_LINE_RE = re.compile(r"^\s*(?:\[\d{2}:\d{2}\]\s*)?Falante\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
REFERENCE_LINE_RE = re.compile(r"^\s*Falante\s*\d+\s*:\s*(.*)$", re.IGNORECASE)


def extract_text(path: Path, line_re: re.Pattern) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if raw in ("", "(vazio)"):
        return ""
    parts = []
    for line in raw.splitlines():
        m = line_re.match(line)
        parts.append(m.group(1) if m else line)
    return " ".join(p.strip() for p in parts if p.strip())


def evaluate_video(hyp_path: Path, ref_path: Path):
    hyp_text = extract_text(hyp_path, HYPOTHESIS_LINE_RE)
    ref_text = extract_text(ref_path, REFERENCE_LINE_RE)
    if not ref_text:
        raise ValueError("referencia vazia")
    if not hyp_text:
        return {"wer": 1.0, "cer": 1.0}
    return {
        "wer": jiwer.wer(ref_text, hyp_text, reference_transform=TRANSFORM, hypothesis_transform=TRANSFORM),
        "cer": jiwer.cer(ref_text, hyp_text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="1,2,3,4,5")
    args = parser.parse_args()

    summary = {}
    for n in (int(v) for v in args.versions.split(",")):
        version_dir = CALLS_DIR / f"v{n}"
        if not version_dir.exists():
            continue
        per_video = []
        for ref_path in sorted(REFERENCE_DIR.glob("*.txt")):
            hyp_path = version_dir / ref_path.name
            if not hyp_path.exists():
                continue
            try:
                r = evaluate_video(hyp_path, ref_path)
                per_video.append({"video": ref_path.name, **r})
            except Exception as e:
                per_video.append({"video": ref_path.name, "error": str(e)})

        ok = [v for v in per_video if "error" not in v]
        avg_wer = sum(v["wer"] for v in ok) / len(ok) if ok else None
        avg_cer = sum(v["cer"] for v in ok) / len(ok) if ok else None
        summary[f"v{n}"] = {"avg_wer": avg_wer, "avg_cer": avg_cer, "videos": per_video}

        print(f"\n=== v{n} ===")
        for v in per_video:
            if "error" in v:
                print(f"  {v['video']}: ERRO ({v['error']})")
            else:
                print(f"  {v['video']}: WER={v['wer']*100:.1f}%  CER={v['cer']*100:.1f}%")
        if avg_wer is not None:
            print(f"  media: WER={avg_wer*100:.1f}%  CER={avg_cer*100:.1f}%")

    out_path = CALLS_DIR / "qualidade_transcricao.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main()
