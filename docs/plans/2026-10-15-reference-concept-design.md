# Funzione 2: dal riferimento al concept (progetto)

Data: 15 ottobre 2026. Stato: **strada scelta dall'utente (Qwen-Image 2.1 in modifica), in attesa dell'approvazione
del download**. Niente scaricato, niente installato, nessun codice.
Piano delle funzioni: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 2. La
funzione è generica: vale per qualunque riferimento e qualunque progetto.

In breve: si vuole un'immagine **nuova** che si riconosca dalla forma di un riferimento (un'immagine o un modello),
oppure che rispetti esattamente una posa o un corpo dato. Lo fa Qwen-Image 2.1, il modello che AssetFurnace usa già,
nella sua modalità di modifica con immagini di riferimento, dentro `vendor/sdcpp/sd-cli`. Prima si prova su quattro
casi reali, giudicati dall'utente a occhio; poi si costruisce.

## 1. Vincoli (decisioni dell'utente del 15/10/2026)

- **Una sola strada: Qwen-Image 2.1 in modifica.** Qwen-Image-Edit-2511 e il ControlNet sono fuori dal progetto. Se la
  modifica fallisce sulla precisione in qualche caso, mi fermo e riporto con le immagini; non passo da solo a un'altra
  strada.
- AssetFurnace è un programma a sé e se ne rispetta la struttura: il motore sta in `vendor/sdcpp/`, i pesi nel catalogo
  (`viewer/backend_catalog.py`, `WeightSet`) e si scaricano solo dopo una conferma. **Niente ComfyUI.**
- **Niente controlli deterministici** su luminosità, contorni, saturazione o colori. Resta l'IoU della sagoma su
  maschere ritagliate (lo stesso della funzione 7), come numero accanto all'immagine.
- Un segnale di "copia" si usa solo se viene da un modello imparato (per esempio la somiglianza DINOv2) **e** regge
  contro i giudizi dell'utente. Il giudizio finale è dell'utente, a occhio.
- "O facciamo qualcosa di fatto bene o non lo facciamo per niente."

## 2. La strada (verificata il 15/10/2026)

- Motore: `vendor/sdcpp/sd-cli.exe`, stable-diffusion.cpp commit `60423ab` del 13/10/2026 (codice MIT). Nessun cambio.
- `docs/qwen_image_2.1.md` di stable-diffusion.cpp: gli stessi pesi di Qwen-Image 2.1 generano e modificano. I
  riferimenti si passano con `-r` (ripetibile, nell'ordine dato: `-r primo.png -r secondo.png`); la modifica richiede la
  visione del codificatore, che con un codificatore GGUF si passa con `--llm_vision`.
- Cosa abbiamo già (in cache e nel catalogo): modello `qwen_image_2.1-Q8_0.gguf`, codificatore
  `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, VAE `qwen_image_2.1_vae_bf16.safetensors`. **Manca solo il file della visione.**
- Controllo dichiarato: l'esempio ufficiale di Qwen-Image 2.1 è "da uno schizzo a una foto, con la struttura dello
  schizzo". Che segua anche mappe di profondità o di posa non è dichiarato: lo dicono i casi qui sotto.
- Impostazioni di partenza: quelle di oggi in `image_api.py` (cfg 1.0, euler, 8 passi), a 1024 × 1024. Se i primi
  risultati sono deboli, provo i passi e la cfg dell'esempio ufficiale (cfg 6.0) e annoto tempi e differenze.

## 3. I quattro casi

Due semi per variante. Il prompt di stile viene dalla `wording` di `image_to_3dlab/prompt_recipes.json` (letto, non
modificato: è della funzione 3).

### Caso 1: un riferimento ridisegnato nel nostro stile

Riconoscibile dalla forma, non copiato. I 5 riferimenti (in `output/`, locali):

| Riferimento | Tipo |
|---|---|
| `compare-20261015/knight/knight.glb` | Modello (umanoide) |
| `compare-20261015/werewolf/werewolf.glb` | Modello (creatura) |
| `compare-20261015/chest/chest.glb` | Modello (oggetto) |
| `images/a-single-friendly-orange-toy-robot-full__20261013-183055/…png` | Immagine |
| `images/full-body-original-northern-fantasy-huma__20261013-184920/…png` | Immagine |

Due varianti d'ingresso: (a) il riferimento così com'è (render frontale per i modelli); (b) una mappa di struttura:
profondità disegnata dal modello per i GLB, lineart per le immagini. 5 × 2 × 2 semi = **20 immagini**.

### Caso 2: un personaggio nella T-pose esatta del corpo base

Corpo base della prova della funzione 4 (sessione af-parti-slot, letto senza modificarlo):
`journal/spike-slot-parts/common/body.png` e `body3d/body_rigged.glb` nel suo worktree. Due varianti: (a) l'immagine del
corpo come riferimento; (b) una **mappa di posa** (scheletro a punti chiave nel formato OpenPose, disegnato da noi
proiettando le ossa del corpo rigato: nessun modello). 2 personaggi × 2 varianti × 2 semi = **8 immagini**. A occhio si
guarda la posa con lo scheletro sovrapposto; accanto, l'IoU della sagoma ritagliata contro quella del corpo.

### Caso 3: un pezzo d'armatura disegnato sopra il corpo base

Riferimento: `body.png`; il prompt chiede un pezzo (busto, poi elmo e spalle) senza cambiare corpo e posa. Termine di
paragone: il concept che la funzione 4 ha già fatto con l'inpainting di Qwen-Image 2.1 (`common/concept.png`).
2 armature × 2 semi = **4 immagini**. Se la modifica funziona meglio dell'inpainting, lo scrivo nelle note condivise per
la funzione 4.

### Caso 4: viste coerenti della stessa creatura

Il lupo mannaro: dalla vista frontale del caso 1 si chiedono lato e retro. Due varianti: (a) solo la vista frontale come
riferimento; (b) la vista frontale più il render di profondità del modello da quel lato (secondo `-r`).
2 viste × 2 varianti × 2 semi = **8 immagini**.

In tutto **40 immagini**.

## 4. Cosa si misura e chi decide

- **Tempo per immagine** (dal log di `sd-cli`: `generate_image completed in`) e **VRAM di picco** (`nvidia-smi` ogni
  mezzo secondo durante la corsa), sulla RTX 5070 Ti da 16 GB.
- **IoU della sagoma** su maschere ritagliate (casi 1 e 2): un numero accanto all'immagine, non un filtro.
- **Un foglio per caso**: ingresso e risultati affiancati, un seme per riga. Accanto a ogni immagine il giudizio
  dell'utente: *riconoscibile e nuovo* / *copiato* / *irriconoscibile* (caso 1), *posa giusta* / *posa sbagliata*
  (casi 2 e 3), *stessa creatura* / *altra creatura* (caso 4).
- **Esito**: un caso passa se l'utente lo approva. Se un caso fallisce sulla precisione (posa, corpo, coerenza fra le
  viste), **mi fermo** e riporto al coordinatore con i fogli; la decisione su come andare avanti è dell'utente.
- **Segnale di copia imparato**: i giudizi del caso 1 sono anche il campione per provare DINOv2. Si calcola la
  somiglianza fra riferimento e risultato e si guarda se separa "copiato" dagli altri (con `fit_band` della funzione 7).
  Entra nella funzione solo se separa; altrimenti non c'è. Richiede un download a parte (sezione 5), chiesto solo
  quando ci sono i giudizi.

Lavoro sulla GPU sempre con il file di blocco condiviso (`%TEMP%\assetfurnace-gpu.lock`, pid e nome della sessione).
Gli script della prova sono usa e getta e restano in `journal/` (regola 2 di `AGENTS.md`); entrano nella repo solo i
pezzi generali (mappa di profondità da un GLB, mappa di posa da un rig), con i loro test.

## 5. Download (solo dopo un sì esplicito)

Dimensione esatta in byte dall'API di Hugging Face del 15/10/2026.

| File | Repo | Byte | Licenza dei pesi | Licenza del codice che lo usa |
|---|---|---:|---|---|
| `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | `Qwen/Qwen3-VL-8B-Instruct-GGUF` (ufficiale di Qwen; lo stesso repo del codificatore che abbiamo già) | **1.159.029.824** (1,16 GB) | Apache-2.0 (Qwen3-VL-8B-Instruct) | `sd-cli` MIT |

- È la parte visiva di Qwen3-VL-8B: legge le immagini di riferimento. Non carica altri file.
- F16 perché è il formato indicato dalla documentazione di `sd-cli` per la modifica con Qwen-Image 2.1. Esiste anche un
  Q8_0 (752.289.728 byte): non lo prendo, la differenza di spazio non vale il rischio di leggere peggio i riferimenti.
- Disco libero su `C:`: 138 GB.
- **Nel catalogo**: un `WeightSet` in più nel backend `qwen-image` di `viewer/backend_catalog.py`, che punta al **file**
  e non alla cartella della cache: la cartella contiene già il codificatore e `_dir_state` sommerebbe i due. Lo
  scaricamento passa da Setup & Status come per gli altri, con nome, dimensione e conferma. Lo scarico con lo stesso
  meccanismo, non a mano.
- Più avanti, solo se serve il segnale di copia: `facebook/dinov2-small`, `model.safetensors`, 88.249.960 byte,
  Apache-2.0, più la libreria `transformers` (Apache-2.0), che oggi non è nel `.venv`.

### Per la funzione 3 (bibbia dello stile)

La modifica accetta più `-r`, quindi oltre ai riferimenti di forma può ricevere **1–3 riferimenti di stile** scelti dalla
bibbia (`image_to_3dlab.style.refs.pick`, della funzione 3). Ordine fisso: prima la forma, poi lo stile, e il prompt li
nomina in quell'ordine ("the first image…", "the style of the other images…"). Offro una funzione pura
`generate(prompt, references, seed) -> Path` sopra il comando di `sd-cli`; la prova "solo prompt / prompt + 1 / prompt + 3
riferimenti di stile" la conduce e la misura la funzione 3 con il suo giudice, sugli stessi soggetti e semi. Il file
della visione serve anche al suo giudice: un solo `WeightSet`, lo aggiungo io.

## 6. Cosa si costruisce dopo la prova (se i casi passano)

- `image_to_3dlab/concept/`: ingressi (riferimento, mappa di profondità da un GLB, mappa di posa da un rig, lineart),
  comando di `sd-cli` per la modifica come funzione pura, sidecar `<immagine>.concept.json` con i riferimenti (nome e
  SHA-256), gli ingressi usati, i modelli con licenza e l'IoU.
- `viewer/image_api.py`: poche righe d'aggancio (`extra_args` ed `extra_provenance` facoltativi nel lavoro) per riusare
  coda e avanzamento, e `--llm_vision` quando ci sono riferimenti.
- `scripts/reference_concept.py` (elencato in `scripts/README.md`).
- Studio, `viewer/studio/reference-concept.js`: nel foglio "Make a picture", carica uno o più riferimenti, sceglie
  l'ingresso (riferimento così com'è, profondità, posa, lineart), vede riferimento e risultato affiancati con l'IoU, e
  dà il giudizio che finisce nel registro di revisione della funzione 7. Nessuna "forza": la modifica non ha una
  manopola, e non la inventiamo.

## 7. Rischi

1. **Licenza di Qwen-Image 2.1 (vale già oggi)**. La Qwen Research License concede l'uso "FOR NON-COMMERCIAL PURPOSES
   ONLY" (§2a) e chiede una licenza a parte per l'uso commerciale (§2b). La dichiarazione di Qwen del 21/09/2026 dice di
   chi sono le immagini, non se si può usare il modello per un prodotto commerciale. Il file della visione è Apache-2.0
   e non cambia nulla.
2. **Precisione**: la modifica potrebbe non tenere la T-pose esatta o il corpo sotto l'armatura. È proprio ciò che
   misurano i casi 2 e 3; se fallisce, mi fermo.
3. **Le mappe come riferimento** potrebbero essere lette come immagini da copiare (una mappa di profondità grigia nel
   risultato). Si vede nei fogli.
4. **Riferimenti di terzi**: il riferimento guida solo la forma; nel sidecar resta il suo SHA-256.
5. **GPU condivisa** con le altre sessioni: file di blocco a ogni lavoro.

## 8. Consegna

Dopo l'approvazione del download: il `WeightSet` nel catalogo (PR piccola, con test), poi la prova (fogli e tempi in
questo documento, riassunto nelle note condivise), poi il giudizio dell'utente, poi PR piccole verso `main` con test e
riga in `CHANGELOG.md`: ingressi e comando; API e studio. Review di Codex con la regola di stop (P1 sempre; P2 solo se
errori veri visibili; massimo 3 giri). Non unisco io.

## 9. Domanda per l'approvazione

Si può scaricare `mmproj-Qwen3VL-8B-Instruct-F16.gguf` (1.159.029.824 byte, Apache-2.0) dal catalogo?

## 10. Seguiti

Dalla review di Codex della PR #9 (secondo giro, P2 rimandati: non sono errori visibili nei casi normali).
- Rimuovere Qwen-Image cancella anche i file parziali (`*.incomplete`) di altre quantizzazioni scaricate a metà nella
  stessa cartella della cache. Serve sapere quale blob parziale appartiene a quale file.
- Un blob condiviso con un'altra voce della cache (stesso contenuto in un altro repo o snapshot fuori dalla cartella)
  viene cancellato e lascia quel collegamento rotto. Meglio passare dall'API di cancellazione di `huggingface_hub`.
- L'avanzamento del download conta tutta la cartella del file con nome, quindi anche file non dichiarati (altre
  quantizzazioni): parte alto e sovrastima il totale. Era già così prima della PR, per il codificatore.
