# Transcritor de Áudio de Aba (DevCall)

Extensão Chrome (Manifest V3) que **captura o áudio de uma aba, transcreve
localmente com Whisper e baixa a transcrição em `.txt`** — tudo no browser,
sem backend e sem que nenhum áudio saia da máquina.

Este MVP é a primeira fatia de um projeto maior: uma ferramenta para reuniões
no Google Meet com a pipeline **captura → diarização → STT → resumo com LLM**.
Hoje todos os estágios estão implementados: captura, diarização (sherpa-onnx),
STT (Whisper) e resumo (Gemma 3 270M).

### Vídeos de demonstração

**Call de exemplo gerada com IA** (áudio usado para testar a extensão):

<video src="data/video_call.mp4" controls width="600"></video>

**Extensão em execução durante a call**, com o pipeline completo de ponta a
ponta (captura → diarização → STT → resumo) chamando todos os modelos:

<video src="data/video_extensao.mp4" controls width="600"></video>

---

## 1. O que foi implementado

### Arquitetura

Tudo roda no browser ("Arquitetura A"). A extensão captura um stream de áudio
da aba, transcreve com o Whisper via `transformers.js` (ONNX Runtime em
WASM/WebGPU) e entrega um arquivo de texto.

```
[Aba do navegador]
      │  áudio da aba (tabCapture)
      ▼
[Offscreen document] ── grava (MediaRecorder) ── decodifica p/ 16 kHz mono
      │                                                   │
      │                                                   ▼
      │                                  sherpa-onnx: pyannote-segmentation-3.0
      │                                  + embedding wespeaker (diarização)
      │                                                   │ turns (falante x tempo)
      │                                                   ▼
      │                                          Whisper (transformers.js)
      │                                                   │ chunks com timestamp
      │                                                   ▼
      │                                  alinhamento texto × falante
      │                                                   │ texto diarizado
      │                                                   ▼
      │                                          Gemma 3 270M (resumo)
      │                                                   │ resumo
      ▼                                                   ▼
[Service worker] ◄───────────────────────── envia transcrição + resumo
      │  chrome.downloads
      ▼
transcricao-<data>.txt + resumo-<data>.txt
```

### Componentes

| Arquivo | Papel |
|---|---|
| `manifest.json` | Declara permissões (`tabCapture`, `offscreen`, `tabs`, `downloads`, `storage`) e a CSP que libera WASM e a Hugging Face. |
| `popup.html` / `popup.js` | Modal: lista de abas, timer de gravação e estado ao vivo (gravando / transcrevendo). |
| `background.js` | Service worker. Orquestra captura, offscreen, estado e o download. |
| `offscreen.html` / `offscreen.js` | Documento invisível que grava o áudio, roda a diarização, o Whisper e o resumo. |
| `lib/` | `transformers.min.js` + binários WASM do ONNX Runtime, **embutidos** (a CSP do MV3 proíbe carregar script de CDN). |
| `lib/sherpa/` | WASM do sherpa-onnx (`sherpa-onnx-wasm-main-speaker-diarization.*`), o glue `sherpa-onnx-speaker-diarization.js`, o modelo de embedding `wespeaker_en_voxceleb_resnet34_LM.onnx` e `diarization.js` (wrapper ESM usado pelo offscreen). O modelo de segmentação (`pyannote-segmentation-3.0`) já vem embutido no `.data` do sherpa-onnx. |

### Fluxo de dados

1. Clique no ícone → o popup lista as abas `http/https`.
2. Selecionar a aba + **Gravar** → o service worker chama
   `tabCapture.getMediaStreamId`, cria o offscreen e manda o `streamId`.
   O offscreen faz `getUserMedia` e começa a gravar.
3. **Finalizar e transcrever** → o offscreen decodifica o WebM para Float32
   mono a 16 kHz.
4. **Diarização** → o sherpa-onnx (pyannote-segmentation-3.0 + embedding
   wespeaker) roda sobre esse mesmo áudio e devolve `turns = [{ start, end,
   speaker }]`.
5. **Transcrição** → o Whisper roda com `return_timestamps: true`, devolvendo
   `chunks` com `[inicio, fim]` por trecho.
6. **Alinhamento** → cada chunk do Whisper é atribuído ao falante de maior
   sobreposição em `turns`, gerando o texto diarizado
   (`[mm:ss] Falante N: ...`).
7. O offscreen passa esse texto para o Gemma 3 270M (instruct), que gera um
   resumo em tópicos ("Principais pontos", "Decisões", "Ações").
8. O service worker baixa `transcricao-<data>.txt` e `resumo-<data>.txt`.

### Decisões técnicas e armadilhas resolvidas

- **Captura no MV3 exige offscreen document.** O service worker não acessa
  `MediaStream` e pode ser suspenso, então a captura vive no offscreen.
- **Reconexão de áudio.** Ao capturar, a aba fica muda para o usuário; uma
  ponte `AudioContext → destination` devolve o som durante a gravação.
- **Permissão `storage`.** O estado (gravando/transcrevendo) fica em
  `chrome.storage.local` para o popup refletir mesmo após ser fechado.
- **Corrida "Receiving end does not exist".** O listener do offscreen é
  registrado de forma síncrona e a lib pesada é carregada por `import()`
  sob demanda; o service worker reenvia o `offscreen-start` com retry.
- **CSP do MV3.** `wasm-unsafe-eval` para compilar WASM e `connect-src`
  liberando a Hugging Face. A biblioteca é embutida, não vem de CDN.
- **Bundle certo.** Usamos `transformers.min.js` (autossuficiente); o build
  `*.web.min.js` externaliza o `onnxruntime` e quebra sem bundler.
- **Modelo.** Baixado da Hugging Face na 1ª execução e mantido em cache;
  o device é WebGPU se disponível, senão WASM.
- **sherpa-onnx é script classico, não ESM.** `sherpa-onnx-wasm-main-speaker-diarization.js`
  espera um global `Module` (estilo Emscripten); `lib/sherpa/diarization.js`
  injeta `<script>` no documento do offscreen e usa `Module.locateFile` para
  resolver `.wasm`/`.data` via `chrome.runtime.getURL`.
- **Embedding trocado em runtime.** O `.data` do sherpa-onnx já traz o
  `pyannote-segmentation-3.0`, mas o embedding padrão é um modelo 3D-Speaker
  (zh). Carregamos `wespeaker_en_voxceleb_resnet34_LM.onnx` à parte e o
  injetamos no FS virtual via `Module.FS_createDataFile` antes de criar o
  `OfflineSpeakerDiarization`.

### Limitações atuais

- **Diarização aproximada** — o embedding wespeaker é treinado em inglês
  (VoxCeleb); funciona como heurística de "quem falou quando" em pt-BR, mas a
  qualidade fica abaixo do pyannote.audio completo (Caminho B no histórico do
  projeto).
- **Tamanho do pacote** — `lib/sherpa/` adiciona ~80 MB (WASM + `.data` do
  sherpa-onnx + modelo wespeaker). O `.data` ainda inclui um embedding
  3D-Speaker que não é usado.
- **Velocidade** — diarização, Whisper (`whisper-base` em WASM puro é lento;
  WebGPU resolve) e resumo rodam em **sequência**, para não disputar
  CPU/VRAM.
- **Resumo** — o Gemma 3 270M é pequeno; para transcrições muito longas a
  qualidade do resumo cai (modelo sem suporte a contexto extenso).
- **Abas** — só `http/https`; `tabCapture` é mais estável na aba ativa.
- **Primeira execução** precisa de internet para baixar os modelos do Whisper
  e do Gemma (os modelos de diarização já vêm embutidos em `lib/sherpa/`).

---

## 2. Como instalar e usar

1. `chrome://extensions` → ative o **Modo de desenvolvedor**.
2. **Carregar sem compactação** → selecione esta pasta.
3. Clique no ícone → escolha a aba → **Gravar** → **Finalizar e transcrever**.
4. Na 1ª vez, aguarde o download do modelo (`Baixando modelo… %`).

**Configuração** (`offscreen.js`):

- `MODEL` → `"Xenova/whisper-base"` (padrão, melhor pt-BR) ou
  `"Xenova/whisper-tiny"` (mais rápido).
- `SUMMARY_MODEL` → `"onnx-community/gemma-3-270m-it-ONNX"` (modelo do
  resumo).

### Estrutura

```
gravador-aba/
├── manifest.json
├── popup.html
├── popup.js
├── background.js
├── offscreen.html
├── offscreen.js
└── lib/
    ├── transformers.min.js
    ├── ort-wasm-simd-threaded.jsep.wasm
    ├── ort-wasm-simd-threaded.jsep.mjs
    └── sherpa/
        ├── diarization.js
        ├── sherpa-onnx-speaker-diarization.js
        ├── sherpa-onnx-wasm-main-speaker-diarization.js
        ├── sherpa-onnx-wasm-main-speaker-diarization.wasm
        ├── sherpa-onnx-wasm-main-speaker-diarization.data
        └── wespeaker_en_voxceleb_resnet34_LM.onnx
```

---

## 3. Diarização (quem falou) — implementado

Pipeline: **captura → diarização → STT → alinhamento → resumo**, tudo no
browser ("Caminho A": 100% local com sherpa-onnx).

### Diarização com sherpa-onnx

- `lib/sherpa/sherpa-onnx-wasm-main-speaker-diarization.{js,wasm,data}` é o
  build WASM oficial do sherpa-onnx para `OfflineSpeakerDiarization`. O `.data`
  já embute o modelo de **segmentação `pyannote-segmentation-3.0`**.
- O **embedding de locutor** é o `wespeaker_en_voxceleb_resnet34_LM.onnx`
  (~25 MB), carregado à parte e gravado no FS virtual do WASM
  (`Module.FS_createDataFile`) antes de criar o `OfflineSpeakerDiarization`.
- `lib/sherpa/diarization.js` (ESM) injeta os scripts do sherpa-onnx no
  documento do offscreen, configura `Module.locateFile` (via
  `chrome.runtime.getURL`) e expõe `diarize(audioFloat32) -> turns`, com
  `turns = [{ start, end, speaker: "Falante N" }]`.
- Roda **sobre o mesmo `audio` Float32 16 kHz** decodificado para o Whisper,
  e **antes** da transcrição.

### Alinhamento texto × falante

Com `turns` em mãos, o Whisper roda com `return_timestamps: true`
(`out.chunks: [{ timestamp: [ini, fim], text }]`), e cada chunk é atribuído ao
falante de maior sobreposição (`speakerFor` em `offscreen.js`):

```
[00:00] Falante 1: bom dia pessoal, vamos começar
[00:07] Falante 2: bom dia, só um instante
[00:11] Falante 1: claro
```

Esse texto diarizado é o que vira `transcricao-<data>.txt` e também a entrada
do resumo (seção 4). Se a diarização falhar ou não retornar turnos, o texto
cai de volta para a saída simples do Whisper (sem prefixo de falante).

### Alternativa não usada: backend com pyannote.audio 4.0

Um servidor local (FastAPI + WebSocket) rodando **pyannote.audio 4.0** daria
diarização de qualidade superior, mas exige instalação/serviço externo —
abandonado em favor do Caminho A (zero instalação).

---

## 4. Resumo com LLM (implementado)

Após a transcrição e o alinhamento, o offscreen roda o **Gemma 3 270M instruct**
(`onnx-community/gemma-3-270m-it-ONNX`) via `transformers.js`, no mesmo
device (WebGPU ou WASM) usado pelo Whisper, **em sequência** para não disputar
recursos.

- O texto diarizado (com `Falante N`) é enviado como mensagem de usuário, com
  um prompt em pt-BR pedindo um resumo em tópicos: "Principais pontos",
  "Decisões" e "Ações".
- O resultado é baixado como `resumo-<data>.txt`, ao lado da
  `transcricao-<data>.txt`.
