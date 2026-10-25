# Prova rapida della funzione 4: parti per slot e adattamento a un corpo base (progetto)

Data: 15 ottobre 2026. Stato: **prova fatta il 15–16/10/2026**: risultati in `2026-10-15-slot-parts-spike.md`. Pesi del gruppo 2 (OmniPart) cancellati il 16/10.

## Decisione (15/10/2026)

L'utente, tramite il coordinatore, approva il progetto e i download dei gruppi 1 (HoloPart) e 2 (OmniPart). Sceglie di
eseguire anche la strada A e **si assume la responsabilità della licenza NVIDIA di PartField**, alle condizioni:

- A gira **solo come riferimento interno di qualità** e nel rapporto è marcata **non utilizzabile per gli asset del
  gioco**, per la licenza;
- **nessuna uscita di A** va in `output/` condiviso, nella Libreria o nella coda di revisione (resta in
  `journal/spike-slot-parts/`);
- **i pesi del gruppo 2 si cancellano a fine prova**, comunque vada;
- se A non parte su questa GPU entro 2 ore, si segna "non installabile qui" e si va avanti con B e C.

I gruppi 3 e 4 si richiedono solo se servono.

È una prova, non la funzione: gli script stanno in `journal/spike-slot-parts/` (fuori dalla repo consegnata). Nella PR
finale entrano solo il rapporto e gli script che generalizzano (regola 2 di `AGENTS.md`).

## Domanda

Quale delle tre strade dà un set di armatura modulare (elmo, busto, spalle, guanti, gambe, stivali + un'arma rigida),
adattato a un corpo base, **a qualità piena e con meno ritocchi a mano**?

## Materiale comune (una volta sola, tutte e tre le strade lo usano)

| Passo | Come | Tempo atteso |
|---|---|---|
| 1. Immagine del corpo base | `viewer/image_api.py` (Qwen-Image 2.1, già installato): umanoide stilizzato, T-pose, calzamaglia aderente, stesso prompt di stile del cavaliere di `output/compare-20261015/` | ~1 min |
| 2. Concept dell'armatura **sullo stesso corpo e nella stessa posa** | `sd-cli` con `--init-img` = immagine del corpo e `--mask` sulle zone degli slot (inpainting), stesso seme. Così sagoma e posa restano quelle del corpo. Se l'inpainting di Qwen-Image 2.1 non regge, ripiego: Qwen-Image-Edit-2511 (vedi download, **solo con approvazione**) | ~2–5 min con qualche tentativo |
| 3. Corpo 3D | `scripts/pixal3d_generate.py` → `scripts/retopo_repaint.py --skip-paint --views` → SkinTokens `demo.py --use_transfer` → `scripts/clean_skin_weights.py` (la catena del cavaliere) | ~10 min |
| 4. Maschere 2D degli slot | Script: proietto lo scheletro del corpo rigato nella vista frontale (la camera di Pixal3D è nota: fov 20°) e assegno ogni pixel del concept all'osso più vicino; gli slot sono gruppi di ossa (testa → elmo, petto/spina → busto, clavicola/braccio alto → spalle, mano → guanti, coscia/stinco → gambe, piede → stivali). L'arma si genera a parte (Pixal3D su un'immagine dell'arma), perché nel concept in T-pose le mani sono vuote | secondi, più eventuali correzioni a mano (contate come ritocco) |

Il corpo e le maschere sono gli stessi per le tre strade, così il confronto misura solo la generazione delle parti.

## Le tre strade

### A. OmniPart con le maschere degli slot — **bloccata dalla licenza, serve una decisione**

- Ingresso: concept + maschera a etichette (`.exr`, un id per slot). OmniPart genera da solo i riquadri 3D delle parti
  e poi le parti.
- **Problema di licenza trovato verificando i pesi.** Il codice e la scheda dei pesi dicono MIT, ma
  `omnipart/OmniPart_modules/partfield_encoder.ckpt` ha **lo stesso SHA-256** (`463efc8a…877ee4`, letto dall'API di
  Hugging Face senza scaricare) di `model_objaverse.ckpt` di **PartField**, che è sotto la NVIDIA License: §3.3,
  "solo per ricerca ed educazione non commerciali", e vale anche per i lavori derivati. Lo script di inferenza lo carica
  a ogni esecuzione (serve al generatore dei riquadri, `bbox_gen.ckpt`, addestrato su quelle caratteristiche). Non si
  toglie senza riaddestrare. In più la demo Gradio carica BRIA RMBG-2.0 (CC BY-NC); lo script da riga di comando no.
- Conseguenza: **anche se vincesse, la strada A non si può usare per produrre gli asset del gioco.** Proposta:
  **non eseguirla** e confrontare B e C (risparmio ~9 GB e ~2 ore). In alternativa, se l'utente ritiene che una prova
  interna rientri nella "ricerca non commerciale", la eseguo come solo riferimento di qualità, marcata come non
  utilizzabile, e cancello i pesi alla fine. Decisione dell'utente, non mia.
- Dove girerebbe: **WSL2** (Linux con CUDA; ruote precompilate solo Linux, detectron2). Dipendenze fissate a
  torch 2.4 cu121: per la RTX 5070 Ti (sm_120) servono torch 2.7+ cu128 e ricompilare flash-attn, torch-scatter,
  nvdiffrast; spconv non ha kernel precompilati oltre sm_90 (compila a runtime: da provare). La issue #4 del progetto
  ha una correzione della comunità per torch 2.8 cu128. Limite: 2 ore per farla partire, poi "non installabile qui".

### B. Pixal3D sul personaggio intero + segmentazione + HoloPart

- Pixal3D (nativo Windows, già installato) sul concept intero → mesh unica.
- Segmentazione per slot: **maschere proiettate** dalla vista frontale (la stessa del passo 4), con propagazione alle
  facce nascoste per vicinanza lungo la superficie. Nessun peso nuovo. SAMPart3D resta una seconda scelta, provata solo
  se la proiezione dà confini sbagliati: è pesante (ottimizzazione per oggetto, ~6 min; RAPIDS cuML, Blender Linux,
  pointops e spconv da compilare). PartField è non commerciale e SAMesh non ha licenza: esclusi.
- HoloPart completa ogni parte dove era nascosta (interno dei guanti, sotto le spalle, collo dell'elmo). Ingresso:
  mesh `.glb` + `mask.npy` con l'etichetta di ogni faccia, che la proiezione produce direttamente.
- Dove: Pixal3D su Windows; HoloPart in WSL2 (estensione CUDA `diso` da compilare dal sorgente, issue #5; torch libero,
  quindi cu128 va bene; numpy fissato a 1.22.3). Progetto fermo da aprile 2025: se si rompe, lo ripariamo noi.

### C. Un'immagine per parte → Pixal3D per parte (come AssetHub)

- Per ogni slot un'immagine isolata: il concept con tutto il resto riempito di sfondo (dalla maschera del passo 4).
  Se una parte isolata è troppo piccola o tagliata, ripiego: ritaglio ingrandito a 1024 px.
- Pixal3D per parte (nativo Windows) → 7 mesh, poi finitura per parte (`retopo_repaint.py --skip-paint`).
- Posizionamento: dalla posizione e dimensione della maschera nella vista frontale, poi allineamento alla zona
  dell'osso sul corpo.
- Dove: tutto su Windows, nessun peso nuovo.

## Adattamento al corpo (uguale per tutte e tre)

Uno script Blender headless (`journal/spike-slot-parts/fit_part.py`), stessi parametri per tutte le strade:

1. **Allineamento** della parte alla zona del suo osso (riquadro delle facce del corpo pesate su quelle ossa).
2. **Gabbia low-poly** della parte (voxel remesh grosso + decimazione) con **Shrinkwrap** sul corpo, scostamento 4 mm
   verso l'esterno (fuori dalla superficie).
3. **Surface Deform** della parte sulla gabbia (Mesh Deform come ripiego se Surface Deform rifiuta il bind).
4. **Pesi dal corpo** con Data Transfer (faccia più vicina, interpolata), poi normalizzati e limitati a 4 influenze.
5. **Oggetti rigidi** (arma; elmo se è un guscio chiuso) pesati al 100% su un osso (mano destra, testa).
6. **Maschera del corpo coperto**: colore dei vertici del corpo = 1 dove un vertice sta entro 8 mm dentro un guscio
   indossato (per il materiale in Unreal, non si taglia la mesh).

Pose di prova: riposo, **walk** (clip Kimodo già in `image_to_3dlab/motions/walk.npz`, suonata con
`scripts/kimodo_retarget.py`) e **sword_swing** come posa estrema (stessa cartella). Nessun modello Kimodo da scaricare.

## Cosa misuro, e come

| Misura | Come |
|---|---|
| Tempo per set | Orologio per passo, da script (generazione, finitura, adattamento), escluso il materiale comune |
| Qualità visiva | Render frontale e a 3/4 di ogni parte e del set indossato (`scripts/blender_render_asset.py`), immagini piccole nel rapporto; giudizio su 3 voci: forma fedele al concept, superficie pulita, texture leggibile |
| Compenetrazioni | Per ogni fotogramma della camminata e della posa estrema: percentuale di vertici del corpo **visibili** fuori dalla maschera che stanno fuori dal guscio, e vertici del guscio dentro il corpo (raycast in Blender). Riporto il massimo e la media sui fotogrammi |
| Ritocchi a mano | Lista concreta dei ritocchi che servirebbero, con una stima in minuti per ciascuno (correzione maschera, parte mancante, pesi da ritoccare, compenetrazione da scolpire). Il numero da confrontare è il totale in minuti |
| Licenze | Codice e pesi di ogni componente, letti dai file di licenza e dalle schede dei modelli |
| Dove gira | Windows nativo / solo WSL2 / non installabile qui, con il motivo |

Criterio di scelta: vince la strada con meno minuti di ritocco a parità di qualità accettabile; il tempo macchina
conta solo se la differenza è grande (oltre 3× fra le strade).

## Installazioni e download

**Già presenti, nessun download**: Qwen-Image 2.1 (sd-cli), Pixal3D (pixal3d-cpp, nativo), SkinTokens, Blender 5.2,
clip Kimodo, WSL2 Ubuntu con CUDA 13.3. (Promemoria: l'output di Pixal3D resta "commerciale con condizioni" per
l'encoder DINOv3 incluso, come nel piano; vale per B e C allo stesso modo.)

**Da installare** (solo codice, in WSL2, ognuno nel suo ambiente `uv` in `~/spike-slot-parts/`, Python 3.10, torch
2.8 cu128, circa 3–5 GB di pacchetti per ambiente): HoloPart; OmniPart solo se l'utente sceglie di eseguire A;
SAMPart3D solo se la proiezione fallisce.

**Pesi** (dimensioni lette dall'API di Hugging Face il 15/10/2026; ciascun gruppo si scarica solo dopo un sì esplicito):

| Gruppo | Per | File | Dimensione | Licenza |
|---|---|---|---|---|
| **1. HoloPart** (necessario per B) | B | `VAST-AI/HoloPart`: `transformer/` 3,23 GB, `vae/` 0,49 GB, `part_encoder/` 0,11 GB | **3,83 GB** | MIT |
| 2. OmniPart (solo se si esegue A) | A | `omnipart/OmniPart`: 6 modelli di flusso e decoder stile TRELLIS | 3,01 GB | MIT (dichiarata) |
| | | `omnipart/OmniPart_modules/bbox_gen.ckpt` | 1,87 GB | MIT dichiarata, ma addestrato sulle caratteristiche di PartField |
| | | `omnipart/OmniPart_modules/partfield_encoder.ckpt` (= PartField `model_objaverse.ckpt`) | 1,24 GB | **NVIDIA License, solo non commerciale** |
| | | `omnipart/OmniPart_modules/sam_vit_h_4b8939.pth` (solo se lo script lo chiede: le maschere le diamo noi) | 2,56 GB | Apache-2.0 |
| | | DINOv2 `dinov2_vitl14_reg` via torch.hub | ~1,2 GB (non verificata) | Apache-2.0 |
| | | **Totale A** | **~7,3–9,9 GB** | |
| 3. SAMPart3D (solo se la proiezione fallisce) | B | `yhyang-myron/SAMPart3D/ptv3-object.pth` | 0,45 GB | MIT |
| | | `facebook/sam-vit-huge/model.safetensors` | 2,56 GB | Apache-2.0 |
| | | **Totale** | **3,0 GB** | |
| 4. Concept con modifica (solo se l'inpainting di Qwen-Image 2.1 non regge) | tutte | `unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf` | 13,24 GB | Apache-2.0 |
| | | `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`: `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + `mmproj-F16.gguf` | 4,68 + 1,35 GB | Apache-2.0 |
| | | `Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors` | 0,25 GB | Apache-2.0 |
| | | **Totale** | **19,5 GB** | |

Richiesta di approvazione ora: **solo il gruppo 1 (3,83 GB, MIT)** e la scelta su A (gruppo 2). I gruppi 3 e 4 li
chiedo di nuovo, con il motivo, solo se arrivo a doverli usare.

Disco: C: ha 154 GB liberi; il disco virtuale di WSL vive su C:, quindi i download sopra stanno nel margine ma vanno
cancellati a fine prova se la strada perde.

## GPU condivisa

Prima di ogni lavoro sulla GPU (Qwen, Pixal3D, SkinTokens, OmniPart, HoloPart) lo script crea in modo esclusivo
`%TEMP%\assetfurnace-gpu.lock` con pid e nome della sessione (`spike-slot-parts`), e lo
cancella alla fine anche in caso d'errore. Se esiste e il processo è vivo, aspetta. Per i lavori in WSL il blocco lo
prende lo script Windows che lancia `wsl`, così vale per tutte le sessioni.

## Scaletta

1. Materiale comune (corpo, concept, corpo rigato, maschere). ~1 ora.
2. Strada C (tutto già installato). ~1 ora.
3. Strada B: Pixal3D intero + proiezione, poi HoloPart in WSL2. ~2 ore con l'installazione.
4. Strada A: OmniPart in WSL2, **solo se approvata nonostante la licenza**. ~2 ore con l'installazione.
5. Adattamento e pose per tutte le strade eseguite. ~1–2 ore.
6. Rapporto `docs/plans/2026-10-15-slot-parts-spike.md`, PR piccola, review di Codex (al massimo 2 giri). Non unisco.

Totale stimato: una giornata di lavoro (meno 2 ore senza A), di cui metà installazioni e attese.

## Decisioni chieste al coordinatore

1. **Strada A**: non eseguirla (consigliato), oppure eseguirla come solo riferimento di qualità, sapendo che PartField è
   non commerciale.
2. **Download del gruppo 1** (HoloPart, 3,83 GB, MIT): sì / no.
3. Va bene un solo corpo base per la prova, con il riadattamento agli altri 5 solo stimato?

## Cosa resta fuori

- Gli altri 5 corpi base (3 razze × 2 sessi): la prova usa un corpo solo. Il riadattamento fra corpi si stima, non si
  misura, a meno che resti tempo.
- Import in Unreal: la funzione 5 lo copre già; qui ci si ferma a Blender.
- Varianti di colore e maschere di tinta (funzione 8).
