# Compila calls/qualidade_transcricao.json + calls/qualidade_diarizacao.json +
# calls/vN/metadata.json (velocidade) num relatorio HTML unico e autocontido
# (graficos embutidos como PNG base64, sem dependencia de internet/CDN).
#
# Uso:
#   scripts/.venv/Scripts/python.exe scripts/generate_report.py

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
CALLS_DIR = ROOT_DIR / "calls"

VERSION_INFO = {
    1: {"stt": "Whisper-base (fp32)", "embedding": "wespeaker"},
    2: {"stt": "Whisper-tiny (fp32)", "embedding": "wespeaker"},
    3: {"stt": "Whisper-base (quantizado q8)", "embedding": "wespeaker"},
    4: {"stt": "Whisper-tiny (fp32)", "embedding": "CAM++"},
    5: {"stt": "Whisper-base (quantizado q8)", "embedding": "CAM++"},
}

COLOR_BLUE = "#4a90e2"
COLOR_RED = "#e0392b"
COLOR_GREEN = "#34c759"
COLOR_GRID = "#3a3a3c"
COLOR_TEXT = "#f2f2f2"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def style_dark_axes(ax, fig):
    fig.patch.set_facecolor("#1c1c1e")
    ax.set_facecolor("#1c1c1e")
    ax.tick_params(colors=COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def bar_chart(versions, series, ylabel, title):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    style_dark_axes(ax, fig)
    n = len(series)
    width = 0.8 / n
    x = range(len(versions))
    for i, (label, values, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, color=COLOR_TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"v{v}" for v in versions])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    legend = ax.legend(facecolor="#2a2a2c", edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    fig.tight_layout()
    return fig_to_base64(fig)


def main():
    trans_data = load_json(CALLS_DIR / "qualidade_transcricao.json")
    diar_data = load_json(CALLS_DIR / "qualidade_diarizacao.json")
    if not trans_data or not diar_data:
        raise SystemExit(
            "Rode antes: scripts/evaluate_transcription_quality.py e "
            "scripts/evaluate_diarization_quality.py"
        )

    versions = sorted(int(v.lstrip("v")) for v in trans_data)
    rows = []
    for n in versions:
        key = f"v{n}"
        t = trans_data.get(key, {})
        d = diar_data.get(key, {})
        meta_path = CALLS_DIR / key / "metadata.json"
        meta = load_json(meta_path) or []
        meta_ok = [m for m in meta if "error" not in m]
        diar_ms = sum(m["diarization_ms"] for m in meta_ok) / len(meta_ok) if meta_ok else None
        trans_ms = sum(m["transcription_ms"] for m in meta_ok) if meta_ok else None
        trans_ms = trans_ms / len(meta_ok) if meta_ok else None
        rows.append(
            {
                "version": n,
                "stt": VERSION_INFO.get(n, {}).get("stt", "?"),
                "embedding": VERSION_INFO.get(n, {}).get("embedding", "?"),
                "wer": t.get("avg_wer"),
                "cer": t.get("avg_cer"),
                "ser": d.get("avg_sequence_error_rate"),
                "speaker_diff": d.get("avg_speaker_count_diff"),
                "diar_ms": diar_ms,
                "trans_ms": trans_ms,
                "videos_trans": t.get("videos", []),
                "videos_diar": d.get("videos", []),
            }
        )

    # --- graficos ---
    wer_chart = bar_chart(
        versions,
        [
            ("WER", [r["wer"] * 100 if r["wer"] is not None else 0 for r in rows], COLOR_RED),
            ("CER", [r["cer"] * 100 if r["cer"] is not None else 0 for r in rows], COLOR_BLUE),
        ],
        "Erro (%)",
        "Qualidade de transcricao (menor = melhor)",
    )
    diar_chart = bar_chart(
        versions,
        [
            (
                "Erro de sequencia",
                [r["ser"] * 100 if r["ser"] is not None else 0 for r in rows],
                COLOR_RED,
            ),
            (
                "Diff n. falantes",
                [r["speaker_diff"] if r["speaker_diff"] is not None else 0 for r in rows],
                COLOR_GREEN,
            ),
        ],
        "Erro (% ou contagem)",
        "Qualidade de diarizacao (menor = melhor)",
    )
    speed_chart = bar_chart(
        versions,
        [
            ("Diarizacao", [(r["diar_ms"] or 0) / 1000 for r in rows], COLOR_BLUE),
            ("Transcricao", [(r["trans_ms"] or 0) / 1000 for r in rows], COLOR_RED),
        ],
        "Tempo medio (s)",
        "Velocidade media por versao",
    )

    # --- html ---
    def fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    def fmt_s(v):
        return f"{v/1000:.1f}s" if v is not None else "—"

    def video_rows_html(r):
        by_video = {}
        for v in r["videos_trans"]:
            by_video.setdefault(v["video"], {}).update({"wer": v.get("wer"), "cer": v.get("cer")})
        for v in r["videos_diar"]:
            by_video.setdefault(v["video"], {}).update(
                {
                    "ser": v.get("sequence_error_rate"),
                    "falantes": f"{v.get('num_speakers_hyp', '?')} / {v.get('num_speakers_ref', '?')}",
                }
            )
        lines = []
        for name, vals in sorted(by_video.items()):
            lines.append(
                "<tr>"
                f"<td>{name}</td>"
                f"<td>{fmt_pct(vals.get('wer'))}</td>"
                f"<td>{fmt_pct(vals.get('cer'))}</td>"
                f"<td>{fmt_pct(vals.get('ser'))}</td>"
                f"<td>{vals.get('falantes', '—')}</td>"
                "</tr>"
            )
        return "\n".join(lines)

    cards_html = []
    for r in rows:
        cards_html.append(
            f"""
        <details class="card">
          <summary>
            <span class="v-badge">v{r['version']}</span>
            <span class="v-combo">{r['stt']} &middot; {r['embedding']}</span>
            <span class="v-metrics">
              <span class="metric"><b>WER</b> {fmt_pct(r['wer'])}</span>
              <span class="metric"><b>Erro seq.</b> {fmt_pct(r['ser'])}</span>
              <span class="metric"><b>Transcricao</b> {fmt_s(r['trans_ms'])}</span>
            </span>
          </summary>
          <table class="video-table">
            <thead><tr><th>Video</th><th>WER</th><th>CER</th><th>Erro seq.</th><th>Falantes (hip/ref)</th></tr></thead>
            <tbody>{video_rows_html(r)}</tbody>
          </table>
        </details>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Relatorio DevCall — comparacao de versoes</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 64px;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #1c1c1e; color: #f2f2f2;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 26px; font-weight: 700; margin: 0 0 4px; }}
  .subtitle {{ color: #9b9b9f; font-size: 14px; margin: 0 0 32px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 40px 0 14px; }}
  .charts {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 8px; }}
  .chart-box {{
    background: #202022; border: 1px solid #3a3a3c; border-radius: 12px;
    padding: 12px; text-align: center;
  }}
  .chart-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .card {{
    background: #202022; border: 1px solid #3a3a3c; border-radius: 12px;
    margin-bottom: 12px; overflow: hidden;
  }}
  .card summary {{
    cursor: pointer; list-style: none; padding: 14px 18px;
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  }}
  .card summary::-webkit-details-marker {{ display: none; }}
  .card[open] summary {{ border-bottom: 1px solid #3a3a3c; }}
  .v-badge {{
    background: #4a90e2; color: #fff; font-weight: 700; font-size: 13px;
    padding: 3px 10px; border-radius: 999px; flex: none;
  }}
  .v-combo {{ font-size: 14px; color: #d0d0d3; flex: 1 1 220px; }}
  .v-metrics {{ display: flex; gap: 16px; font-size: 12px; color: #9b9b9f; flex-wrap: wrap; }}
  .v-metrics b {{ color: #f2f2f2; font-weight: 600; }}
  .video-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .video-table th, .video-table td {{
    padding: 8px 18px; text-align: left; border-bottom: 1px solid #2a2a2c;
  }}
  .video-table th {{ color: #9b9b9f; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  .note {{
    font-size: 12px; color: #9b9b9f; border-left: 3px solid #4a90e2;
    padding: 10px 14px; background: #202022; border-radius: 0 8px 8px 0; margin-top: 32px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Relatorio DevCall — comparacao de versoes</h1>
  <p class="subtitle">Qualidade e velocidade de v1-v5 contra a referencia em calls/transcriptions</p>

  <div class="charts">
    <div class="chart-box"><img src="data:image/png;base64,{wer_chart}"></div>
    <div class="chart-box"><img src="data:image/png;base64,{diar_chart}"></div>
    <div class="chart-box"><img src="data:image/png;base64,{speed_chart}"></div>
  </div>

  <h2>Detalhe por versao (clique pra expandir)</h2>
  {''.join(cards_html)}

  <p class="note">
    WER/CER medem a transcricao (menor = melhor). Erro de sequencia mede a diarizacao
    comparando a ordem de troca de falantes (sem timestamp na referencia, o rotulo da
    hipotese e realinhado ao rotulo de referencia mais provavel via algoritmo hungaro
    antes de comparar — nao e um DER de verdade). v1/v2/v3 compartilham o mesmo
    embedding de diarizacao (wespeaker) e por isso tem erro de sequencia identico;
    o mesmo vale pra v4/v5 (CAM++).
  </p>
</div>
</body>
</html>"""

    out_path = CALLS_DIR / "relatorio.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Relatorio salvo em {out_path}")


if __name__ == "__main__":
    main()
