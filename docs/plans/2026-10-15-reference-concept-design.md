# Funzione 2: dal riferimento al concept (progetto)

Data: 15 ottobre 2026. Stato: **scaletta della gara, in attesa dell'approvazione dei download**. Niente scaricato,
niente installato, nessun codice.
Piano delle funzioni: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 2. La
funzione è generica: vale per qualunque riferimento e qualunque progetto.

In breve: si vuole un'immagine **nuova** che si riconosca dalla forma di un riferimento (un'immagine o un modello),
oppure che rispetti esattamente una posa o un corpo dato. Prima di costruire si fa una **gara** fra i due modi di
modifica con immagine di riferimento che `sd-cli` sa già fare, su quattro casi reali. Vince chi l'utente giudica
migliore a occhio. Il ControlNet si aggiunge solo se perdono tutti e due sulla precisione.

## 1. Vincoli (decisione dell'utente del 15/10/2026)

- AssetFurnace è un programma a sé e se ne rispetta la struttura: i motori stanno in `vendor/` (qui
  `vendor/sdcpp/sd-cli`), i pesi nel catalogo (`viewer/backend_catalog.py`, `WeightSet`) e si scaricano solo dopo una
  conferma. **Niente ComfyUI.**
- **Niente controlli deterministici** su luminosità, contorni, saturazione o colori: "non funzionano bene". Restano
  fuori il controllo della palette in ΔE e quello di copia sui bordi Canny della versione precedente. Resta l'IoU della
  sagoma su maschere ritagliate (lo stesso della funzione 7).
- Un segnale di "copia" si usa solo se viene da un modello imparato (per esempio la somiglianza DINOv2) **e** regge
  contro i giudizi dell'utente. Il giudizio finale è dell'utente, a occhio.
- "O facciamo qualcosa di fatto bene o non lo facciamo per niente."

## 2. Le strade in gara (verificate il 15/10/2026)

Motore per entrambe: `vendor/sdcpp/sd-cli.exe`, stable-diffusion.cpp commit `60423ab` del 13/10/2026 (codice MIT).
Nessun cambio al motore.

### A. Qwen-Image 2.1 in modifica

- Lo stesso modello che AssetFurnace usa già per generare. `docs/qwen_image_2.1.md` di stable-diffusion.cpp: gli
  stessi pesi generano e modificano; i riferimenti si passano con `-r` (ripetibile, nell'ordine dato) e la visione del
  codificatore con `--llm_vision`.
- Cosa manca: solo il file della visione di Qwen3-VL-8B (`mmproj`).
- Licenza: come oggi (Qwen Research License, vedi rischi).
- Controllo dichiarato: l'esempio ufficiale è "da uno schizzo a una foto con la struttura dello schizzo". Mappe di
  profondità o di posa non sono dichiarate: lo dirà la gara.

### B. Qwen-Image-Edit-2511

- Un modello di modifica a sé (20B), con codificatore Qwen2.5-VL-7B e la VAE di Qwen-Image 1.
  `docs/qwen_image_edit.md`: va attivato `--model-args qwen_image_zero_cond_t=true`, altrimenti "la qualità della
  modifica cala molto"; impostazioni dell'esempio `--cfg-scale 2.5 --flow-shift 3`, `-r` ripetibile, `--llm_vision`.
- Controllo dichiarato: la 2509 dichiara il supporto nativo a "mappe di profondità, mappe di contorni, mappe di punti
  chiave" e agli schizzi; la 2511 è la sua evoluzione (meno deriva, personaggi più coerenti, ragionamento geometrico più
  forte) e non lo ripete, ma non lo toglie.
- Licenza: **Apache-2.0** per tutti i file, verificati uno per uno (sezione 5).

### C. Fun ControlNet Union aggiunto da noi a `sd-cli` (solo se A e B falliscono)

- `sd-cli` oggi ha il ControlNet solo per SD1, SD2, SDXL e SVD (`src/model/diffusion/control.hpp`); nessuna issue o
  PR a monte per Qwen-Image.
- Il ramo di controllo (alibaba-pai, `Qwen-Image-2.1-Fun-Controlnet-Union`, 7,55 GB, Qwen Research License identica a
  quella di Qwen-Image 2.1, SHA-256 del `LICENSE` `8dc973f0…b28d`) sono 16 blocchi uguali a quelli del trasformatore,
  innestati ogni due blocchi. Si porterebbe in C++ nel nostro `sd-cli` e si confronterebbe con l'implementazione
  ufficiale (VideoX-Fun, Apache-2.0) a parità di seme, prima di fidarsene.
- Ha un progetto e un elenco di download suoi, che preparo solo se serve.

## 3. I casi della gara

Gli stessi ingressi, gli stessi semi (2 per variante) e la stessa dimensione (1024 × 1024) per A e per B. Il prompt di
stile viene dalla `wording` di `image_to_3dlab/prompt_recipes.json` (letto, non modificato: è della funzione 3).

### Caso 1: un riferimento ridisegnato nel nostro stile

Riconoscibile dalla forma, non copiato. I 5 riferimenti già scelti (in `output/`, locali):

| Riferimento | Tipo |
|---|---|
| `compare-20261015/knight/knight.glb` | Modello (umanoide) |
| `compare-20261015/werewolf/werewolf.glb` | Modello (creatura) |
| `compare-20261015/chest/chest.glb` | Modello (oggetto) |
| `images/a-single-friendly-orange-toy-robot-full__20261013-183055/…png` | Immagine |
| `images/full-body-original-northern-fantasy-huma__20261013-184920/…png` | Immagine |

Due varianti d'ingresso: (a) il riferimento così com'è (render frontale per i modelli); (b) una mappa di struttura:
profondità disegnata dal modello per i GLB, lineart per le immagini. 5 × 2 × 2 semi = **20 immagini per strada**.

### Caso 2: un personaggio nella T-pose esatta del corpo base

Corpo base della prova della funzione 4 (sessione af-parti-slot, letto senza modificarlo):
`journal/spike-slot-parts/common/body.png` e `body3d/body_rigged.glb` nel suo worktree. Due varianti: (a) l'immagine del
corpo come riferimento; (b) una **mappa di posa** (scheletro a punti chiave nel formato OpenPose, disegnato da noi
proiettando le ossa del corpo rigato: nessun modello). 2 personaggi × 2 varianti × 2 semi = **8 per strada**.
A occhio si guarda la posa con lo scheletro sovrapposto; si misura l'IoU della sagoma ritagliata contro quella del
corpo.

### Caso 3: un pezzo d'armatura disegnato sopra il corpo base

Riferimento: `body.png`; il prompt chiede un pezzo (busto, poi elmo e spalle) senza cambiare corpo e posa. Si confronta
anche con il concept che la funzione 4 ha già fatto con l'inpainting di Qwen-Image 2.1 (`common/concept.png`), come
termine di paragone. 2 armature × 2 semi = **4 per strada**. Coordinamento con la funzione 4 nelle note condivise:
se la gara trova un modo migliore, lo può usare anche lei.

### Caso 4: viste coerenti della stessa creatura

Il lupo mannaro: dalla vista frontale del caso 1 si chiedono lato e retro. Due varianti: (a) solo la vista frontale come
riferimento; (b) la vista frontale più il render di profondità del modello da quel lato (secondo `-r`).
2 viste × 2 varianti × 2 semi = **8 per strada**.

In tutto **40 immagini per strada**, 80 nella gara.

## 4. Cosa si misura e chi decide

- **Tempo per immagine** (dal log di `sd-cli`: `generate_image completed in`) e **VRAM di picco** (`nvidia-smi` ogni
  mezzo secondo durante la corsa), sulla RTX 5070 Ti da 16 GB.
- **IoU della sagoma** su maschere ritagliate (casi 1 e 2): solo un numero accanto all'immagine, non un filtro.
- **Foglio di confronto** per ogni caso: ingresso, A e B affiancati, con lo stesso seme sulla stessa riga. Accanto a
  ogni immagine un giudizio dell'utente su tre valori: *riconoscibile e nuovo* / *copiato* / *irriconoscibile* (caso 1),
  *posa giusta* / *posa sbagliata* (casi 2 e 3), *stessa creatura* / *altra creatura* (caso 4).
- **Regola**: per ogni caso l'utente sceglie A, B o "nessuna delle due". Vince la strada scelta in almeno 3 casi su 4.
  A parità decidono tempo, VRAM e licenza (B è tutta Apache-2.0). Se i casi 2 o 3, quelli di precisione, finiscono
  entrambi con "nessuna", si passa alla strada C.
- **Segnale di copia imparato**: i giudizi del caso 1 sono anche il campione per provare DINOv2. Si calcola la
  somiglianza fra riferimento e risultato e si guarda se separa "copiato" dagli altri (con `fit_band` della funzione 7).
  Entra nella funzione solo se separa; altrimenti non c'è.

Lavoro sulla GPU sempre con il file di blocco condiviso (`%TEMP%\assetfurnace-gpu.lock`, pid e nome della sessione).
Gli script della gara sono usa e getta e restano in `journal/` (regola 2 di `AGENTS.md`); entrano nella repo solo i
pezzi generali (mappa di profondità da un GLB, mappa di posa da un rig), con i loro test.

## 5. Download (ognuno solo dopo un sì esplicito)

Dimensioni esatte in byte dall'API di Hugging Face del 15/10/2026. Disco libero su `C:`: 138 GB.

| Strada | File | Repo | Byte | Licenza dei pesi | Licenza del codice |
|---|---|---|---:|---|---|
| A | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | `Qwen/Qwen3-VL-8B-Instruct-GGUF` | 1.159.029.824 | Apache-2.0 | `sd-cli` MIT |
| B | `qwen-image-edit-2511-Q4_K_M.gguf` | `unsloth/Qwen-Image-Edit-2511-GGUF` (quantizzazione di `Qwen/Qwen-Image-Edit-2511`) | 13.244.758.624 | Apache-2.0 (originale e repo) | `sd-cli` MIT |
| B | `Qwen2.5-VL-7B-Instruct-Q8_0.gguf` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` (l'organizzazione di llama.cpp) | 8.098.523.680 | Apache-2.0 (originale e repo) | `sd-cli` MIT |
| B | `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` | 853.119.712 | Apache-2.0 | `sd-cli` MIT |
| B | `vae/diffusion_pytorch_model.safetensors` | `Qwen/Qwen-Image-Edit-2511` (ufficiale; stesso SHA-256 `0c8bc8b7…` della VAE di `Qwen/Qwen-Image`) | 253.806.966 | Apache-2.0 | `sd-cli` MIT |

- Strada A: **1,16 GB**. Strada B: **22,45 GB**. In tutto 23,61 GB.
- Scelte e scarti, per la licenza: `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF`, citato dalla documentazione di
  `sd-cli`, **non dichiara una licenza**: preferito `ggml-org`, che dichiara Apache-2.0 (stessi byte del Q8_0). La VAE
  ufficiale in formato diffusers è preferita al repackage di Comfy-Org (Apache-2.0, 253.806.246 byte, contenuto
  diverso): se `sd-cli` non legge i nomi diffusers, ripiego su quello, con conferma.
- Quantizzazione Q4_K_M per il 20B: è quella dell'esempio di `sd-cli` e sta nei 16 GB con lo scaricamento automatico.
  Q8_0 (21,76 GB) solo se la Q4 mostra difetti attribuibili alla quantizzazione, con una domanda a parte.
- Nessun file carica altri pesi: modello, codificatore, visione e VAE sono tutti in tabella.
- Più avanti, solo se serve il segnale di copia: `facebook/dinov2-small`, `model.safetensors`, 88.249.960 byte,
  Apache-2.0 (più la libreria `transformers`, Apache-2.0, che oggi non è nel `.venv`).

**Nel catalogo** (`viewer/backend_catalog.py`): A diventa un `WeightSet` in più del backend `qwen-image`; B un backend
nuovo `qwen-image-edit` con i suoi quattro file. Ogni `WeightSet` punta al **file**, non alla cartella della cache:
`_dir_state` conta i file singoli, mentre la cartella di Qwen3-VL contiene già il codificatore e falserebbe il
conteggio. Licenza di B in `image_to_3dlab/provenance.py` come profilo nuovo, Apache-2.0. Lo scaricamento passa da
Setup & Status come per gli altri, con nome, dimensione e conferma.

## 6. Cosa si costruisce dopo la gara (se una strada vince)

Dipende dal vincitore; lo schema, da confermare dopo la gara:
- `image_to_3dlab/concept/`: ingressi (riferimento, mappa di profondità da un GLB, mappa di posa da un rig, lineart),
  comando di `sd-cli` per la strada vincente come funzione pura, sidecar `<immagine>.concept.json` con riferimenti
  (nome e SHA-256), strada, modelli con licenza, IoU.
- `viewer/image_api.py`: poche righe d'aggancio (`extra_args` ed `extra_provenance` facoltativi nel lavoro) per riusare
  coda e avanzamento.
- `scripts/reference_concept.py` (elencato in `scripts/README.md`).
- Studio, `viewer/studio/reference-concept.js`: nel foglio "Make a picture", carica uno o più riferimenti, sceglie
  l'ingresso (riferimento così com'è, profondità, posa, lineart), vede riferimento e risultato affiancati con l'IoU, e
  dà il giudizio che finisce nel registro di revisione della funzione 7. Una "forza" c'è solo se la strada vincente ne
  ha una vera: nessuna delle due ha una manopola, e non la inventiamo.

## 7. Rischi

1. **Licenza di Qwen-Image 2.1 (vale già oggi, strada A)**. La Qwen Research License concede l'uso "FOR NON-COMMERCIAL
   PURPOSES ONLY" (§2a) e chiede una licenza a parte per l'uso commerciale (§2b). La dichiarazione di Qwen del 21/09/2026
   dice di chi sono le immagini, non se si può usare il modello per un prodotto commerciale. La strada B non ha questo
   problema.
2. **20B su 16 GB**: la strada B scarica parte dei pesi in RAM (31 GB nel PC). Potrebbe essere lenta; il tempo misurato
   è parte del giudizio.
3. **Le mappe come riferimento** potrebbero essere lette come immagini da copiare (una mappa di profondità grigia nel
   risultato). La gara lo mostra nel foglio.
4. **Riferimenti di terzi**: il riferimento guida solo la forma; nel sidecar resta il suo SHA-256.
5. **GPU condivisa** con le altre sessioni: file di blocco a ogni lavoro.

## 8. Consegna

Dopo l'approvazione dei download: la gara (fogli e tempi in questo documento, risultati nelle note condivise), poi la
scelta dell'utente, poi PR piccole verso `main` con test e riga in `CHANGELOG.md`: catalogo e licenza; ingressi e
comando; API e studio. Review di Codex con la regola di stop (P1 sempre; P2 solo se errori veri visibili; massimo 3
giri). Non unisco io.

## 9. Domande per l'approvazione

1. Download della strada A (1,16 GB) e della strada B (22,45 GB) come in tabella?
2. Va bene la gara con 40 immagini per strada, o meno semi per accorciarla?

## 10. Seguiti

(Vuoto finché non arrivano le review.)
