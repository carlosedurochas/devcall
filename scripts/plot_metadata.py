# Le calls/vN/metadata.json (gerado por evaluate_models.py) e plota a
# velocidade media (diarizacao vs transcricao) de cada versao.
#
# Uso:
#   scripts/.venv/Scripts/python.exe scripts/plot_metadata.py

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
CALLS_DIR = ROOT_DIR / "calls"

LABELS = {
    1: "v1\nwhisper-base\nwespeaker",
    2: "v2\nwhisper-tiny\nwespeaker",
    3: "v3\nwhisper-base-q8\nwespeaker",
    4: "v4\nwhisper-tiny\nCAM++",
    5: "v5\nwhisper-base-q8\nCAM++",
}

versions = []
diar_avg = []
trans_avg = []

for n in sorted(LABELS):
    path = CALLS_DIR / f"v{n}" / "metadata.json"
    if not path.exists():
        continue
    entries = json.loads(path.read_text(encoding="utf-8"))
    ok = [e for e in entries if "error" not in e]
    if not ok:
        continue
    versions.append(n)
    diar_avg.append(sum(e["diarization_ms"] for e in ok) / len(ok) / 1000)
    trans_avg.append(sum(e["transcription_ms"] for e in ok) / len(ok) / 1000)

x = range(len(versions))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar([i - width / 2 for i in x], diar_avg, width, label="Diarizacao", color="#4a90e2")
bars2 = ax.bar([i + width / 2 for i in x], trans_avg, width, label="Transcricao", color="#e0392b")

ax.set_ylabel("Tempo medio (s)")
ax.set_title("Velocidade media por versao (media dos videos em calls/videos)")
ax.set_xticks(list(x))
ax.set_xticklabels([LABELS[n] for n in versions], fontsize=8)
ax.legend()
ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)
fig.tight_layout()

out_path = CALLS_DIR / "velocidade_media.png"
fig.savefig(out_path, dpi=150)
print(f"Grafico salvo em {out_path}")
