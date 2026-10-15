# Funzione 2: dal riferimento al concept (progetto)

Data: 15 ottobre 2026. Stato: **proposta, in attesa di approvazione**. Niente scaricato, niente installato.
Piano delle funzioni: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 2. La
funzione è generica: vale per qualunque riferimento e qualunque progetto.

In breve: si parte da un riferimento (un'immagine o un GLB), se ne ricava una **mappa di controllo** (sagoma, bordi,
lineart o profondità) e una **palette**, e Qwen-Image 2.1 disegna un oggetto nuovo che rispetta la forma. Il risultato
riceve due punteggi: **sagoma** (dentro una fascia: simile, non identica) e **palette** (entro una distanza). Lo studio
mostra riferimento, mappa e risultato affiancati con i punteggi.

La sorpresa della verifica: **`sd-cli` non supporta il ControlNet di Qwen-Image**, quindi il "riuso" previsto dal piano
non è gratis. Propongo una prova breve su due strade che `sd-cli` sa già fare, e il ControlNet vero (con ComfyUI) solo se
la prova fallisce.

## 1. Cosa si riusa (verificato il 15/10/2026)

### `sd-cli` (stable-diffusion.cpp, commit `60423ab` del 13/10/2026, MIT)

| Cosa | Supporto | Prova |
|---|---|---|
| ControlNet (`--control-net`, `--control-image`, `--control-strength`) | **Solo SD1, SD2, SDXL, SVD** (reti UNet) | `src/model/diffusion/control.hpp` a quel commit costruisce solo `ControlNetBlock` UNet; `qwen_image_2_1.hpp` non ha nessun ramo di controllo. Nessuna issue o PR aperta per il ControlNet di Qwen-Image |
| img2img (`-i` immagine iniziale, `--strength`) | Sì, per tutti i modelli | Opzioni generiche del campionatore |
| **Modifica con immagine di riferimento** (`-r`, più `--llm_vision`) | **Sì, per Qwen-Image 2.1** | `docs/qwen_image_2.1.md`: stessi pesi per generare e modificare; serve in più il file della visione di Qwen3-VL (`mmproj`). L'esempio ufficiale di Qwen-Image 2.1 è proprio "da uno schizzo a una foto, con la struttura dello schizzo" |

Quindi oggi, senza installare altro, ci sono due modi di guidare la forma: partire da un'immagine iniziale (img2img) o
dare la mappa come riferimento nella modalità di modifica.

### I due ControlNet del piano

| Modello | Licenza dei pesi | Dimensione | Base | Gira con | Per noi |
|---|---|---|---|---|---|
| **Qwen-Image-2.1-Fun-Controlnet-Union** (alibaba-pai, 23/09/2026) | **Qwen Research License**: il file `LICENSE` ha lo stesso SHA-256 di quello di Qwen-Image 2.1 (`8dc973f0…b28d`) | 7,55 GB bf16; **3,78 GB** in int8 nel repackage di Comfy-Org (`model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors`) | Qwen-Image 2.1 (lo stesso che usiamo) | VideoX-Fun (Apache-2.0, Python) o **ComfyUI** (GPL-3.0; supporto unito il 28/09/2026, [Comfy-Org/ComfyUI#16519](https://github.com/Comfy-Org/ComfyUI/pull/16519)). **Non** `sd-cli` | Compatibile, stessa licenza di ciò che già usiamo; 8 controlli (Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose, Scribble) più inpainting; forza regolabile (`control_context_scale`) |
| InstantX/Qwen-Image-ControlNet-Union | Apache-2.0 | 3,54 GB | **Qwen-Image 1** (20B, Apache-2.0), non la 2.1 | diffusers, ComfyUI | Architettura diversa: per usarlo serve un secondo modello base (12–13 GB in GGUF Q4) e il suo codificatore. È però l'unica strada **con licenze tutte permissive** (vedi rischi) |

**Dipendenze dei pesi** (la lezione di OmniPart):
- Il Fun ControlNet contiene solo il ramo di controllo (`control_img_in` e 16 `control_blocks`) e si carica sopra il
  trasformatore di Qwen-Image 2.1, che già usiamo con la stessa licenza. Non carica altri file.
- I preprocessori dei controlli nei nodi di ComfyUI e in VideoX-Fun scaricano modelli propri (HED, lineart, DWPose,
  Depth Anything) con licenze varie. **Non li usiamo**: le mappe le facciamo noi (sezione 2).
- Depth Anything V2: solo la **Small** è Apache-2.0 (99 MB); **Base e Large sono CC BY-NC 4.0**. Se un giorno serve la
  profondità da una foto, solo la Small.

### Già in AssetFurnace

- `viewer/image_api.py`: comando, avanzamento, coda (una immagine alla volta), sidecar di provenienza.
- `image_to_3dlab/checks/silhouette.py` (funzione 7): `subject_of_image`, `fit_mask`, `iou`, `overlay`; da
  `registry.py`: `result` e `grade_band`. Il punteggio di sagoma è **lo stesso IoU** della funzione 7, fra due immagini
  invece che fra un modello e un riferimento.
- `image_to_3dlab/matte.py`: ritaglio del soggetto (BiRefNet-lite MIT o u2net). Serve perché `sd-cli` scrive un alfa
  rumoroso (nota in `matte.py`): la maschera del risultato si ricava ritagliandolo.
- OpenCV (già nelle dipendenze) per bordi e lineart.

## 2. Cosa costruisco

Un pacchetto `image_to_3dlab/concept/`, piccolo e testabile senza GPU:

| File | Fa |
|---|---|
| `control_maps.py` | Dal riferimento: maschera del soggetto (alfa o ritaglio), e le mappe **sagoma** (maschera piena), **canny** (OpenCV), **lineart** (XDoG: differenza di gaussiane, puro calcolo, nessun peso), **profondità** solo da un GLB (z-buffer disegnato dal modello con la stessa proiezione ortografica della funzione 7). Da un GLB si sceglie la vista (fronte, lato, tre quarti) |
| `palette.py` | Palette del soggetto: k-means in CIELAB sui pixel della maschera (k configurabile, 6 di base), con il peso di ogni colore. Distanza fra due palette: per ogni colore il più vicino dell'altra in CIEDE2000, media pesata, simmetrica. Esporta anche i colori come testo per il prompt |
| `score.py` | I punteggi, nel formato `result(...)` della funzione 7: `concept.silhouette` (IoU, fascia `min`–`max`), `concept.palette` (ΔE, `max`), `concept.copy` (sovrapposizione dei bordi interni, F-score dei canny entro 2 px, con un `max`: ferma un risultato che ricalca il riferimento anche dentro la sagoma, il caso img2img a forza bassa) |
| `routes.py` | Come si guida la generazione: argomenti in più per `sd-cli` per la strada **init** (`-i`, `--strength`) e per la strada **edit** (`-r`, `--llm_vision`); funzioni pure, testabili leggendo il comando |
| `run.py` | Un concept: mappa, palette, comando, generazione, ritaglio, punteggi, sidecar `<immagine>.concept.json` con riferimento (nome e SHA-256), controllo, forza, strada, modelli in più con licenza, punteggi |

Più:
- `scripts/reference_concept.py`: un concept dalla riga di comando, `--score-only` per dare il punteggio a un'immagine
  già fatta, `--batch cartella/` per la prova d'accettazione (tabella e foglio di confronto). Elencato in
  `scripts/README.md`.
- `viewer/concept_api.py`: `POST /api/concept` (riferimento, controllo, forza, prompt), `GET` dello stato con i
  punteggi. Riusa la coda di `image_api.MANAGER`, così resta una immagine alla volta.
- `viewer/studio/reference-concept.js` (e il suo CSS): nel foglio "Make a picture", un interruttore "From a reference":
  carica il riferimento, sceglie controllo e forza, vede la palette ricavata; alla fine riferimento, mappa e risultato
  affiancati con sovrapposizione delle sagome (stessi colori della funzione 7), IoU, palette a confronto e ΔE.

**Righe di aggancio nei file condivisi** (minime):
- `viewer/image_api.py`: `ImageJob` accetta `extra_args` e `extra_provenance` facoltativi; `build_command` li aggiunge e
  `provenance` li unisce. Circa 8 righe, senza cambiare il comportamento di oggi.
- `viewer/generate_api.py`: 3–4 righe per instradare `/api/concept`.
- `viewer/studio/create.js` e `studio.html`: 2–3 righe ciascuno per montare il pannello.

**Cosa vuol dire "forza"** per strada:
- init: `--strength` (quanto rumore sopra l'immagine iniziale). L'immagine iniziale non è il riferimento, ma la sua
  **sagoma riempita con la palette** (colori piatti per zona), così forma e colori entrano senza copiare i dettagli.
- edit: il modello non ha una manopola di forza; la si approssima con la mappa (sagoma sola = libera, lineart = stretta)
  e con il prompt. Nello studio il cursore diventa la scelta della mappa, e lo si dice.
- ControlNet (solo se si arriva alla fase B): `control_context_scale`.

**Prompt e funzione 3**: leggo `prompt_recipes.json` (la `wording` del goal) senza modificarlo e aggiungo la palette
come testo. Per ora non mi serve nessun campo nuovo; se la prova mostra che serve una formula fissa per "concept da
riferimento", la chiedo al coordinatore.

## 3. La prova prima della funzione

**Fase A** (quasi senza download): sui 5 riferimenti, per ognuna delle strade init ed edit, 3 livelli di forza × 2 semi
a 768 px, 8 passi. Si misurano IoU, ΔE, copia, tempo per immagine; più un foglio di confronto da guardare a occhio
("disegnato da capo o ricalcato?"). Sulla GPU con il file di blocco condiviso.

**Regola di decisione**: una strada passa se su almeno 4 riferimenti su 5 c'è una forza in cui sagoma, palette e copia
sono nei limiti **e** l'utente, guardando il foglio, la giudica un disegno nuovo. Se passa, si costruisce su quella. Se
nessuna passa, **fase B**: ComfyUI con il Fun ControlNet (download sotto, da approvare a parte).

**I 5 riferimenti**: modelli e immagini già in `output/` (locali, non nella repo), scelti per coprire entrambi i tipi
di riferimento e forme diverse:

| Riferimento | Tipo | Perché |
|---|---|---|
| `compare-20261015/knight/knight.glb` | GLB (umanoide) | Sagoma articolata; profondità dal modello |
| `compare-20261015/werewolf/werewolf.glb` | GLB (creatura) | Sagoma irregolare, pelo |
| `compare-20261015/chest/chest.glb` | GLB (oggetto) | Forma squadrata: la copia si vede subito |
| `images/a-single-friendly-orange-toy-robot-full__20261013-183055/` | Immagine | Palette forte e semplice |
| `images/full-body-original-northern-fantasy-huma__20261013-184920/` | Immagine | Figura intera con dettagli interni |

Dai GLB si usa la vista frontale e quella a tre quarti; dalle immagini l'unica vista.

## 4. Download (ognuno solo dopo un sì esplicito)

| Fase | File | Dimensione | Licenza | Serve per |
|---|---|---|---|---|
| A | `Qwen/Qwen3-VL-8B-Instruct-GGUF` → `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,16 GB | Apache-2.0 | La strada edit (la visione del codificatore). La strada init non scarica niente |
| B | `Comfy-Org/Qwen-Image-2.1` → `model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors` | 3,78 GB | Qwen Research License | Il ControlNet |
| B | `Comfy-Org/Qwen-Image-2.1` → `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7,26 GB | Qwen Research License | Lo stesso modello che abbiamo in GGUF, nel formato di ComfyUI |
| B | `Comfy-Org/Qwen-Image-2.1` → `text_encoders/qwen3vl_8b_w4a8.safetensors` | 6,31 GB | Apache-2.0 (Qwen3-VL) | Il codificatore nel formato di ComfyUI |
| B | ComfyUI (codice) e PyTorch CUDA nel suo ambiente | circa 5–6 GB | GPL-3.0 (gira in locale, non lo ridistribuiamo) | Il motore |

Fase B in tutto: circa 17,4 GB di pesi più l'ambiente. La VAE è già in cache. Se ComfyUI-GGUF (Apache-2.0) caricasse i
GGUF che abbiamo già, si risparmierebbero 13,6 GB: da verificare all'inizio della fase B, non prima.

## 5. Alternative scartate (per ora)

- **Portare il Fun ControlNet dentro `sd-cli`** (C++): il ramo sono 16 blocchi uguali a quelli del trasformatore,
  innestati ogni 2 blocchi; fattibile ma lungo, e il binario è precompilato da `vendor/`. Meglio come contributo a
  monte se la fase B lo dimostra utile.
- **VideoX-Fun in Python**: vuole i pesi completi in formato diffusers (trasformatore 13 GB più codificatore 17 GB) e su
  16 GB gira solo con scaricamento in RAM. ComfyUI fa lo stesso con file più piccoli.
- **InstantX su Qwen-Image 1**: un secondo modello base da 12–13 GB e un altro codificatore, con uno stile diverso da
  quello degli altri concept. Resta la scorta se la licenza di Qwen Research diventa un problema (sotto).
- **Correggere la palette dopo** (trasferimento di colore sul risultato): rimandato. Prima si vede se basta il prompt
  più l'immagine iniziale; correggere dopo nasconde il problema invece di risolverlo.

## 6. Rischi

1. **Licenza di Qwen-Image 2.1 (vale già oggi, non la introduce questa funzione)**. La Qwen Research License concede
   l'uso "FOR NON-COMMERCIAL PURPOSES ONLY" (§2a) e chiede una licenza a parte per l'uso commerciale (§2b). La repo la
   tratta come "commerciale con condizioni" perché Qwen ha dichiarato il 21/09/2026 che le immagini prodotte non sono
   Materials. Quella dichiarazione riguarda **di chi sono le immagini**, non se si possa **usare il modello** per un
   prodotto commerciale. Il Fun ControlNet ha la stessa identica licenza, quindi non cambia l'esposizione, ma prima di
   produrre in grande per un prodotto commerciale va deciso dall'utente (eventualmente scrivendo a Qwen, o passando alla
   strada tutta Apache-2.0: Qwen-Image 1 con InstantX).
2. **La strada edit potrebbe non seguire bene una mappa di bordi**: Qwen-Image-Edit 2509 lo faceva per addestramento,
   per la 2.1 c'è solo l'esempio dello schizzo. Lo dice la prova.
3. **Copia invece di disegno nuovo**: con img2img a forza bassa il risultato ricalca il riferimento. Lo ferma
   `concept.copy`, e il limite superiore della fascia di sagoma.
4. **Riferimenti di terzi** (render o immagini di modelli altrui): il riferimento guida solo la forma; nel sidecar resta il suo
   SHA-256, così si sa sempre da cosa è nato un concept. La soglia di copia serve anche a questo.
5. **La palette nel prompt è imprecisa**: Qwen capisce i nomi dei colori meglio dei codici esadecimali. La strada init
   porta i colori nell'immagine iniziale; la prova misura entrambe.
6. **GPU condivisa** con le altre sessioni: file di blocco in `%TEMP%\assetfurnace-gpu.lock` per ogni lavoro sulla GPU.
7. **ComfyUI su Blackwell (RTX 5070 Ti) con i file int8 convrot**: non provato. Solo in fase B.

## 7. Come provo l'accettazione

- **Test unitari** (prima del codice, file piccoli): mappe da forme sintetiche (un quadrato, una L); palette di immagini
  a colori piatti noti; ΔE CIEDE2000 sui valori di riferimento pubblicati (Sharma 2005); IoU e fascia; `concept.copy` alta
  per un'immagine con sé stessa, bassa per una forma uguale con interno diverso; comandi di `sd-cli` per le due strade;
  sidecar con riferimento e licenze; API e modulo dello studio come gli altri.
- **Accettazione del piano**: `scripts/reference_concept.py --batch` sui 5 riferimenti con la strada scelta. Passa se
  tutti e 5 hanno IoU dentro la fascia (proposta di partenza 0,70–0,93, configurabile) e ΔE della palette sotto la
  soglia (proposta 15, configurabile), con `concept.copy` sotto il suo massimo. Le soglie di partenza si correggono con
  `fit_band` (funzione 7) sui giudizi dell'utente. Tabella e foglio di confronto in questo documento.

## 8. Consegna

PR piccole verso `main`, ognuna con test e riga in `CHANGELOG.md`:
1. Punteggi e mappe (`concept/control_maps.py`, `palette.py`, `score.py`) e `--score-only`. Niente GPU.
2. Strade di `sd-cli`, aggancio in `image_api.py`, risultati della prova in questo documento.
3. API e pannello dello studio.
4. Solo se serve: la strada ComfyUI.

Review di Codex con la regola di stop (P1 sempre; P2 solo se errori veri visibili; massimo 3 giri). Non unisco io.

## 9. Domande per l'approvazione

1. Va bene la prova in fase A prima del ControlNet (e il download di 1,16 GB per la strada edit)?
2. La licenza di Qwen-Image 2.1 per un uso commerciale: si accetta come oggi, o si tiene pronta la strada Apache-2.0?

## 10. Seguiti

(Vuoto finché non arrivano le review.)
