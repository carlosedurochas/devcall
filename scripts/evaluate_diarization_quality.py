# Avalia a qualidade da diarizacao (calls/vN/<video>.txt) contra a referencia
# em calls/transcriptions/<video>.txt.
#
# A referencia nao tem timestamps, so "Falante N: texto" por turno, entao nao
# da pra calcular DER (Diarization Error Rate) de verdade, que exige alinhar
# por tempo. Em vez disso, comparamos a SEQUENCIA de falantes (quem fala depois
# de quem, quantas vezes cada um troca), que e o que da pra medir so com texto:
#
# 1. Reduz cada transcricao a sequencia de rotulos de falante por turno,
#    colapsando turnos consecutivos do mesmo falante (a hipotese tende a
#    fragmentar mais que a referencia por causa da segmentacao automatica).
# 2. Como o rotulo "Falante 2" do sistema nao necessariamente corresponde ao
#    "Falante 2" da referencia (problema de permutacao de cluster), aproximamos
#    a posicao temporal de cada turno da hipotese pela posicao proporcional
#    equivalente na referencia, contamos coocorrencia hipotese x referencia, e
#    resolvemos o mapeamento otimo de rotulos com o algoritmo hungaro (scipy).
#    Forca bruta com permutations() explode quando a hipotese super-segmenta
#    (ex.: CAM++ as vezes acha 12 "falantes" onde so tem 4-5) — 12! e alto
#    demais pra testar um por um.
#
# Uso:
#   scripts/.venv/Scripts/python.exe scripts/evaluate_diarization_quality.py [--versions=1,2,3,4,5]

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT_DIR = Path(__file__).resolve().parent.parent
CALLS_DIR = ROOT_DIR / "calls"
REFERENCE_DIR = CALLS_DIR / "transcriptions"

HYPOTHESIS_LINE_RE = re.compile(r"^\s*(?:\[\d{2}:\d{2}\]\s*)?Falante\s*(\d+)\s*:", re.IGNORECASE)
REFERENCE_LINE_RE = re.compile(r"^\s*Falante\s*(\d+)\s*:", re.IGNORECASE)


def extract_speaker_sequence(path: Path, line_re: re.Pattern):
    raw = path.read_text(encoding="utf-8").strip()
    if raw in ("", "(vazio)"):
        return []
    seq = []
    for line in raw.splitlines():
        m = line_re.match(line)
        if m:
            seq.append(int(m.group(1)))
    # Colapsa repeticoes consecutivas (turnos fragmentados do mesmo falante).
    collapsed = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
    return collapsed


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def best_label_mapping(hyp_seq, ref_seq):
    """Mapeia cada rotulo da hipotese pro rotulo de referencia mais provavel.

    Sem timestamps na referencia, aproxima o "tempo" de cada turno da
    hipotese pela sua posicao proporcional na sequencia de referencia (ex.:
    o turno do meio da hipotese cai por volta do turno do meio da
    referencia). Com isso monta uma matriz de coocorrencia hipotese x
    referencia e resolve o mapeamento otimo com o algoritmo hungaro
    (maximiza concordancia). Rotulos da hipotese sem correspondente (quando
    ela super-segmenta, ex.: CAM++ achando 12 falantes onde tem 5) ficam sem
    par e contam como erro na distancia de edicao final.
    """
    if not hyp_seq:
        return {}
    hyp_labels = sorted(set(hyp_seq))
    ref_labels = sorted(set(ref_seq))
    hyp_idx = {label: i for i, label in enumerate(hyp_labels)}
    ref_idx = {label: i for i, label in enumerate(ref_labels)}

    matrix = np.zeros((len(hyp_labels), len(ref_labels)), dtype=int)
    n_ref = len(ref_seq)
    for i, h in enumerate(hyp_seq):
        j = min(int(i * n_ref / len(hyp_seq)), n_ref - 1)
        matrix[hyp_idx[h]][ref_idx[ref_seq[j]]] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)  # maximiza concordancia
    mapping = {hyp_labels[r]: ref_labels[c] for r, c in zip(row_ind, col_ind)}

    unmapped_id = (max(ref_labels) if ref_labels else 0) + 1000
    for label in hyp_labels:
        if label not in mapping:
            mapping[label] = unmapped_id
            unmapped_id += 1
    return mapping


def evaluate_video(hyp_path: Path, ref_path: Path):
    hyp_seq = extract_speaker_sequence(hyp_path, HYPOTHESIS_LINE_RE)
    ref_seq = extract_speaker_sequence(ref_path, REFERENCE_LINE_RE)
    if not ref_seq:
        raise ValueError("referencia vazia")
    mapping = best_label_mapping(hyp_seq, ref_seq)
    remapped_hyp_seq = [mapping[s] for s in hyp_seq]
    dist = edit_distance(remapped_hyp_seq, ref_seq)
    return {
        "num_speakers_ref": len(set(ref_seq)),
        "num_speakers_hyp": len(set(hyp_seq)),
        "num_turns_ref": len(ref_seq),
        "num_turns_hyp": len(hyp_seq),
        "sequence_error_rate": dist / len(ref_seq),
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
        avg_ser = sum(v["sequence_error_rate"] for v in ok) / len(ok) if ok else None
        avg_speaker_diff = (
            sum(abs(v["num_speakers_hyp"] - v["num_speakers_ref"]) for v in ok) / len(ok) if ok else None
        )
        summary[f"v{n}"] = {
            "avg_sequence_error_rate": avg_ser,
            "avg_speaker_count_diff": avg_speaker_diff,
            "videos": per_video,
        }

        print(f"\n=== v{n} ===")
        for v in per_video:
            if "error" in v:
                print(f"  {v['video']}: ERRO ({v['error']})")
            else:
                print(
                    f"  {v['video']}: falantes ref={v['num_speakers_ref']} hip={v['num_speakers_hyp']}  "
                    f"turnos ref={v['num_turns_ref']} hip={v['num_turns_hyp']}  "
                    f"erro_sequencia={v['sequence_error_rate']*100:.1f}%"
                )
        if avg_ser is not None:
            print(f"  media: erro_sequencia={avg_ser*100:.1f}%  diff_num_falantes={avg_speaker_diff:.2f}")

    out_path = CALLS_DIR / "qualidade_diarizacao.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main()
