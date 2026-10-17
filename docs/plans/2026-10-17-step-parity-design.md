# Ogni passo saltabile, correggibile e rilanciabile, da studio, riga di comando e agenti (progetto)

Data: 17 ottobre 2026. Stato: **fatto**. Le mancanze 1-11 sono chiuse; la parità sulle tre strade è nella
sezione 12, le PR nella sezione 13, quello che resta fuori nei Seguiti.

Richiesta dell'utente: "dobbiamo dare la possibilità a agenti e UI di fare queste cose". Ogni passo della pipeline si
deve poter **saltare**, il suo risultato **correggere o sostituire**, e lo si deve poter **rilanciare da solo** senza
rifare quelli prima. Tre strade, stesse possibilità: lo studio (`viewer/studio` con `viewer/*_api.py`), la riga di
comando (`scripts/`, `pipeline.py`) e gli agenti (server MCP `image_to_3dlab/mcp_server.py` e lotti JSON/CSV
`image_to_3dlab/batch.py`).

In breve: **una sola descrizione per passo** (`image_to_3dlab/step_catalog.py`: nome, ingressi, parametri, uscite e le
quattro operazioni) e **una sola funzione Python per passo** in `image_to_3dlab/`. Le tre strade sono adattatori sottili
che leggono il catalogo. Per gli agenti la catena di passi è la riga di lotto, che ha già ripresa, licenze, provenienza e
coda di revisione: un agente che vuole un passo solo lancia un lotto di una riga. Un test di parità confronta catalogo,
flag della riga di comando, chiavi delle API dello studio e opzioni dei lotti, e fallisce quando una strada resta
indietro.

## 1. Cosa manca oggi (verificato su `main`, d6f7057)

| # | Mancanza | Dove |
|---|---|---|
| 1 | Gli agenti arrivano solo a `generate` e `finish`. `materials`, `export_unreal` e `checks` sono nomi senza modulo, saltati con un avviso. Il server MCP ha solo gli strumenti dei lotti | `batch.py:59`, `batch.py:1188`, `batch_steps/`, `mcp_server.py:218` |
| 2 | Cambiare un campo di una riga la rifà dall'inizio: l'impronta è di tutta la riga | `batch.py:126`, `batch.py:1071` |
| 3 | Nello studio solo Animate si rifà; cambiare triangoli o seme vuol dire un nuovo Create. Il seme di Pixal3D è sempre casuale | `studio/app.js:99`, `studio/steps.js:108`, `studio/create.js:125`; il Finish della catena manda `{}` (`activity_api.py:267`) |
| 4 | Esportazione Unreal: lo studio non manda LOD, collisione, maschera di tinta, direzione né nome (l'API accetta il nome, `unreal-export.js:66` non lo manda). Le soglie dei controlli (`--set`) solo da riga di comando | `unreal_export_api.py:55`, `review_api.py:109`, `scripts/check_asset.py:36` |
| 5 | Famiglie di creature: clip, retarget e varianti GenMM sempre dopo il rig, senza interruttore. `chain_retarget.py` senza riga di comando | `autorig_api.py:209`, `clip_library.py:205`, `scripts/chain_retarget.py` |
| 6 | Bibbia dello stile: solo `style_bible.py prompt|refs` la legge. Nessuna strada la applica | `style/prompts.py:37`, `style/refs.py:24` |
| 7 | Pezzi, modo 3: il concept del vestito è sempre il primo seme (`slot_parts.py:329`); `--stop-at-pictures` ferma dopo i pezzi. Arma e scudo senza scelta. `refit`, `fit`, `unreal` solo da riga di comando | `scripts/slot_parts.py:280`, `slot_parts_api.py:143` |
| 8 | Materiali (8): solo riga di comando, fuori da ogni catena, nessun file proprio al posto delle zone | `scripts/game_materials.py`, `materials.py:624` |
| 9 | Una versione fatta in Blender non va avanti a Finish, Rig, Animate: lo studio nasconde i passi, i selettori per nome non la elencano | `app.js:151`, `autorig_api.py:115`, `props_api.py:858` |
| 10 | Nessuna riga di comando per il testo-immagine: il codice sta in `viewer/image_api.py` | `image_api.py:178` |
| 11 | La coda di revisione non ha strumenti MCP; le righe di lotto ci arrivano senza controlli e non si approvano finché qualcuno non lancia `check_asset.py` | `review.py:218`, `batch.py:1259` |

Altri fatti che il progetto usa: lo studio ricostruisce gli asset dai file a ogni richiesta e collega i passi per sha256
(`assets_api.py:209`), non per "fatto da"; i record di rig e di versione Blender hanno già lo sha256 del loro ingresso
(`autorig_api.py:183`, `blender_versions.py:280`); la catena dello studio accetta già un `made` precompilato
(`activity_api.py:383`), cioè "salta questo passo, usa questo file"; il lucchetto della GPU è preso solo dai modelli
dei materiali (`material_zones.py`, `delight.py`, `baked_light.py`).

## 2. Il catalogo: una descrizione per passo

`image_to_3dlab/step_catalog.py` contiene un `StepSpec` per passo. Nessuna esecuzione: solo descrizione e validazione.

```python
Param(name, type, default, choices=None, range=None, help="")      # int, float, bool, text, choice, file, files, json
StepSpec(
    name="finish", feature=None, label="Finish",
    takes=("asset", "picture"),       # cosa legge della catena: asset (GLB), picture (immagine), rig, ...
    makes="asset",                    # cosa lascia: asset, picture, rig, clips, report, export, materials
    params=(Param("faces", "int", 40000, range=(1000, 200000)), ...),
    row_fields=("faces",),            # campi della riga di lotto che contano per questo passo (oltre ai params)
    skip="asset passes through unchanged",   # None: non si salta (es. generate senza un file al suo posto)
    replace=("glb",),                 # file proprio accettato al posto dell'uscita; () se non ha senso
    candidates=None,                  # o il nome del parametro che ne fa più d'uno (es. "count")
    gpu=True,
    cli="scripts/retopo_repaint.py",  # dove sta la riga di comando
    run="image_to_3dlab.batch_steps.finish:run",
)
```

I passi, nell'ordine di una catena:

| Passo | Funz. | Prende | Lascia | Parametri principali | Salto | File proprio | Candidati |
|---|---|---|---|---|---|---|---|
| `picture` | — | prompt | picture | width, height, seed, count, cfg_scale, sampler, negative_prompt, bible | sì: si dà `images` | PNG/JPG | `count` semi |
| `concept` | 2 | picture, riferimenti | picture | mode, shape, style_refs, subject, colours, parts, outfit, piece, view, seed, count, bible | sì | PNG/JPG | `count` semi |
| `generate` | — | picture | asset | backend, seed, steps, gss, gsh, fov, matte | solo con un file | GLB (`asset`) | — |
| `finish` | — | asset, picture | asset | faces, photo, atlas, texture_size, metallic, roughness, ior, voxel, paint, skip_bake, skip_compress | sì | GLB | — |
| `rig` | — | asset | rig | (nessuno oggi) | sì | GLB con scheletro | — |
| `clips` | 6 | rig | clips | family, library, variations, variation_factor, variation_seed | sì | — | — |
| `animate` | — | rig | clips | presets, speed, arm_spread | sì | GLB animati | — |
| `materials` | 8 | asset | asset | zones, materials, ao, profile, zones_file | sì | GLB, maschera PNG o JSON zona→materiale | — |
| `export_unreal` | 5 | asset (+rig, clips) | export | size_cm, measure, name, collision, lods, tint, forward, body, body_name | sì | — | — |
| `checks` | 7 | asset | report | profile, set, reference, height, width, depth | sì | — | — |

Fuori dalla catena di una riga, perché non sono un passo su un asset:

- **Pezzi (4)**: un insieme di pezzi da un corpo, con le sue fasi `concept → cuts → pictures → models → fit` più
  `refit` e `unreal`. Il catalogo li descrive come `pieces.*` con gli stessi campi; l'unica implementazione è
  `scripts/slot_parts.py` (le sue funzioni `run_set`, `refit_job`, `run_fit`, `unreal_set`), che studio e MCP
  lanciano come oggi fa `slot_parts_api.py`.
- **Revisione (7)**: approvare o rimandare è una decisione, non un passo. Una funzione sola, `checks.review.decide`
  più il verdetto di `scripts/review_queue.py:decide`, per tutte e tre le strade.
- **Bibbia dello stile (3)**: non è un passo ma un parametro, `bible`, di `picture`, `concept` e `pieces`. Spento per
  difetto. `"-"` è la bibbia attiva, un nome una bibbia precisa. Applica `style.prompts.compose` al testo e aggiunge
  fino a 3 riferimenti da `style.refs.pick` dove il passo ne accetta. La provenienza registra bibbia e versione.

## 3. Come si evita di scrivere tre volte la stessa cosa

1. **Una funzione per passo, in `image_to_3dlab/`.** Dove oggi la logica sta solo nello studio, si sposta nel pacchetto
   e lo studio la importa: il testo-immagine (`viewer/image_api.py` → `image_to_3dlab/text_image.py`), i comandi del
   rig (`autorig_api.rig_commands`, `animate_command` → `image_to_3dlab/rigging.py`). Lo studio tiene i suoi gestori di
   lavori e la sua interfaccia, non il "come".
2. **I parametri vivono nel catalogo.** Nomi, valori predefiniti, intervalli e scelte. `step_catalog.check(step,
   options)` restituisce i problemi; lo usano il `preflight` dei passi di lotto, le API dello studio e gli strumenti
   MCP. Oggi il Finish ha due elenchi di opzioni (`finish_api.py:56` e `batch_steps/finish.py:30`): diventano uno.
3. **Gli agenti usano la riga di lotto.** Un passo da solo è un lotto di una riga: `run_steps` (MCP) scrive la specifica
   e la lancia con `submit_batch`. Così ripresa, licenze, provenienza, coda di revisione, un lotto alla volta e lo stato
   leggibile valgono anche per un passo singolo, senza un secondo motore di lavori.
4. **La riga di comando è lo script del passo.** Ogni voce del catalogo nomina il suo script; dove non c'è si aggiunge
   (`scripts/text_image.py`, `scripts/chain_retarget.py` come riga di comando, `scripts/game_materials.py
   --zones-file`). Le catene da riga di comando sono `pipeline.py --run-batch`.
5. **Lo studio legge il catalogo.** `GET /api/steps` restituisce i passi con i loro parametri; i moduli "Rifai con altri
   valori" li costruiscono da lì, quindi un parametro nuovo appare nello studio senza un elenco a mano.
6. **Il test di parità.** `tests/test_step_parity.py` prende ogni passo del catalogo e verifica che: lo script accetti
   un flag per ogni parametro; l'API dello studio accetti ogni parametro; il passo di lotto esista e il suo
   `preflight` rifiuti un'opzione sconosciuta; l'MCP lo raggiunga. Una strada che resta indietro fa fallire il test.
   Le eccezioni sono scritte nel catalogo con il motivo, così si vedono.

## 4. Le quattro operazioni, uguali ovunque

| Operazione | Lotti e agenti | Riga di comando | Studio |
|---|---|---|---|
| **Saltare** | Non mettere il passo in `steps` | Non lanciare lo script | "Salta" sul passo; il passo dopo prende il risultato di quello prima |
| **Rilanciare dal passo N** | Cambiare i suoi valori nella specifica: riparte dal primo passo cambiato (sezione 5). `rerun_step(batch, row, step)` lo forza a valori uguali | Rilanciare lo script del passo sul file del passo prima | "Rifai con altri valori" su un passo fatto; i passi prima restano |
| **Sostituire l'uscita con un proprio file** | `"replace": {"<passo>": "file"}` nella riga: il passo non gira, il file (validato) diventa la sua uscita | Dare il file al passo dopo | "Usa un mio file" sul passo (anche una versione Blender) |
| **Scegliere un candidato** | Un passo con `count > 1` lascia la riga in `waiting` con i candidati; `pick_candidate` (MCP) o `"picks": {"<passo>": <seme>}` la fa continuare | `--count N --stop` poi `--pick` | Le immagini in fila, un clic per scegliere |

Un file proprio entra con un record di provenienza `source: user-provided`: licenza e classe vengono dal suo
`.provenance.json` se c'è (una versione Blender ce l'ha), altrimenti sono "non dichiarate" e il controllo
`license.components` lo dice. Il cancello di licenza del lotto lo tratta come oggi tratta un `asset` senza record.

Un passo con candidati fa le immagini di tutti i semi, le registra nello stato e si ferma. La scelta non fa parte
dell'impronta del passo (rifare le immagini per cambiare scelta sarebbe GPU sprecata), ma fa parte dell'impronta dei
passi dopo: cambiare scelta rifà da lì in avanti. Uno stato nuovo della riga, `waiting`, con codice di uscita 3 del
lotto (nessuna riga fallita, qualcuna aspetta una scelta).

## 5. Lotti: si riparte dal primo passo cambiato (PR a)

Oggi `BatchRow.fingerprint()` copre tutta la riga. Diventa un'impronta per passo, a catena:

    impronta[0] = H(sorgente)                          immagini, asset e suoi record, prompt (contenuto, non percorso)
    impronta[i] = H(impronta[i-1], nome_i, campi_i)    campi_i: opzioni del passo + i row_fields del catalogo

con `campi_i` per passo: `generate` backend e parameters; `finish` faces e opzioni; `export_unreal` kind, size_cm,
measure e opzioni; `checks` kind, size_cm, measure e opzioni. Ogni passo fatto tiene la sua impronta nello stato. Alla
ripresa una riga cambiata non si azzera: i passi con impronta uguale e file intatti restano, si riparte dal primo
diverso. Rimuovere un passo lascia gli altri; aggiungerne uno in fondo fa girare solo quello. La riga ha una nota
"cambiata nella specifica: riparte da `finish`". Un'impronta registrata in un formato vecchio vale come diversa (si
rifà tutto, come oggi). Il blocco `batch` del manifest di `generate` tiene solo id e riferimenti: il resto non conta per
generare, e un manifest con `faces` vecchie sarebbe falso.

Due passi nuovi, file adattatori come prevede il contratto (`batch_steps/__init__.py`):

- **`export_unreal`** chiama `scripts/export_unreal.export_unreal` con `kind`, `size_cm`, `measure` della riga e le
  opzioni `name` (predefinito l'id), `collision`, `lods`, `tint`, `forward`, `body`, `body_name`, `animations`. Scrive
  in `rows/<id>/export_unreal/`. Non cambia l'asset corrente: il suo risultato è un'uscita in più (`outputs.json` la
  elenca sotto `unreal`). Il `preflight` vuole `size_cm` e Blender.
- **`checks`** chiama `checks.registry.run_checks` con il profilo del tipo (`equipment` → `equipment_part`), le soglie
  in `"checks": {"set": {"budget.faces.max": 20000}}` (stesso formato di `--set`), l'immagine della riga come
  riferimento della sagoma, `size_cm` come altezza o lato. Scrive `<id>.checks.json` nella sua cartella. Il rapporto
  viaggia con l'asset nella coda di revisione, quindi la riga si può approvare subito (mancanza 11). Un controllo
  `fail` non ferma la riga: decide chi revisiona. `"checks": {"stop_on_fail": true}` la ferma.
- **`checks` entra nei passi predefiniti** (`generate`, `finish`, `checks`). Con le impronte per passo, un lotto finito
  ripreso fa girare solo i controlli.

## 6. Agenti: il server MCP (PR b)

Strumenti nuovi, funzioni semplici come gli altri, testabili senza MCP:

| Strumento | Cosa fa |
|---|---|
| `describe_steps()` | Il catalogo: passi, parametri, valori predefiniti, cosa si salta o si sostituisce (sostituisce `list_steps`, che resta come alias) |
| `run_steps(name, steps, images|asset|prompt, options, kind, faces, size_cm, ...)` | Scrive una specifica di una riga in `output/batches/.agent/<name>.json` e la lancia. Un passo solo o una catena |
| `rerun_step(batch, row, step)` | Segna il passo da rifare e riprende il lotto |
| `pick_candidate(batch, row, step, choice)` | Registra la scelta per una riga in `waiting` e riprende |
| `review_queue()`, `review_item(key)` | La coda e una voce, come `review_queue.py --list` |
| `check_asset(glb, key, profile, set, reference, height, width, depth)` | Come `check_asset.py`: rapporto e coda |
| `review_decide(key, decision, note, asset_sha256)` | Approva o rimanda, con il verdetto, come lo studio |
| `pieces_run`, `pieces_pick`, `pieces_resume`, `pieces_refit`, `pieces_fit`, `pieces_unreal`, `pieces_status`, `pieces_cancel` | `scripts/slot_parts.py` in sottofondo, un insieme alla volta, come lo lancia lo studio |

Passi di lotto nuovi in questa PR, per arrivare a testo-immagine, concept e rig/animazione: `picture`, `concept`,
`rig`, `animate` (con `clips` separato in PR d). `StepContext` prende `picture` (l'immagine corrente: la prima di
`images`, o quella lasciata da `picture`/`concept`), e `generate` legge quella. Una riga può partire da `prompt`
(per `picture`) oltre che da `images` o `asset`. I candidati e `replace` (sezione 4) arrivano con questi passi.

## 7. Studio (PR c)

- **Rifai con altri valori**: ogni passo fatto ha "Rifai…" che apre il suo modulo con i valori usati l'ultima volta
  (letti dal record del passo) e lancia solo quel passo sul risultato del passo prima. I risultati vecchi restano su
  disco. I passi dopo, fatti da un ingresso che non è più quello corrente (sha256 nel loro record), si vedono come
  "fatto da un Finish precedente" con "Rifai" accanto, invece di sembrare aggiornati.
- **Seme di Pixal3D**: campo seme con "nuovo a ogni prova" (spuntato per difetto, come oggi). In Create e nel passo
  Modello. Il seme usato si legge nel riepilogo del passo.
- **Triangoli in Create**: Create manda le facce scelte al Finish della catena (oggi manda `{}`).
- **Esportazione Unreal**: nome, collisione (`box`, `convex`, `unreal`, nessuna), LOD (automatici o nessuno), maschera
  di tinta (automatica, nessuna, un file), direzione (dalla famiglia o a mano). L'API li accetta e li passa a
  `export_unreal`.
- **Soglie dei controlli**: nel foglio di revisione, "Rifai i controlli" con le soglie del profilo modificabili; vanno
  a `run_checks` come `overrides`.
- **Coda di revisione e lotti**: le righe in `waiting` mostrano i candidati sulla pagina Batches, un clic per scegliere.

## 8. Funzioni 3, 4, 6 e le versioni Blender (PR d)

- **Famiglie (6)**: `clips` diventa un passo a sé. Nello studio, sotto Rig, "Clip della famiglia" (spuntato per
  difetto, come oggi) e "varianti più lunghe (GenMM)"; `creature_clips.py --no-variations`, `--variation-factor`,
  `--variation-seed`; `scripts/chain_retarget.py RIG SOURCE --out` come riga di comando.
- **Bibbia (3)**: il parametro `bible` su tutte e tre le strade: casella "Usa la bibbia attiva (nome)" in Create, nel
  pannello del concept e nei pezzi; `--bible NAME|-` su `text_image.py`, `reference_concept.py make`, `slot_parts.py
  run`; opzione `bible` nei lotti e in `run_steps`.
- **Pezzi (4)**: `--stop-at-concept` (modo 3) fa i concept dei semi dati e si ferma; `--pick concept=<seme>` riprende.
  Arma e scudo entrano in `candidates.json` come gli altri pezzi, con la stessa scelta. Lo studio mostra i concept del
  modo 3 da scegliere e aggiunge `refit`, `fit`, `unreal` al pannello dei pezzi.
- **Versioni Blender (mancanza 9)**: una versione è un "file proprio" al posto dell'uscita del passo da cui viene
  (modello o finito). Lo studio mostra i passi anche su una versione scelta e li lancia su `blender-edits/versions/
  <id>/model.glb`; Rig e Animate la accettano (`pickable_models` include le versioni). Nei lotti e per gli agenti è una
  riga con `asset` uguale al file della versione, che porta già la sua provenienza (`source_record`).

## 9. Materiali (PR e), come deciso dall'utente

L'utente ha chiuso la funzione 8 con **un materiale per pezzo o per asset**, proposto da una regola e scelto da una
persona (`image_to_3dlab/piece_materials.py`, PR #61, #64, #65): niente zone automatiche, niente maschere di zone,
niente strumento per dipingerle ("quello che ha fatto ha fatto"). Il passo `materials` (#81) è questo materiale:

- nei lotti e in `run_steps`: `"materials": {"material": "leather"}`, oppure la proposta della regola dalle parole da
  cui il modello è nato (la sua provenienza, o quella del modello della riga) e dal `kind`; scritto in
  `rows/<id>/materials/material.json` e passato a `export_unreal`, che lo mette nell'ORM (la sua opzione `material`
  vince); senza il passo il modello tiene le sue mappe; deve venire prima di `export_unreal`;
- i pezzi di un set: `slot_parts.py materials SET_DIR --set helmet=leather`, i menu della pagina Pieces e, per gli
  agenti, `pieces_materials(body, set, {pezzo: materiale | null})`; l'export del set li legge;
- un asset singolo nello studio: il menu "Material" di Export for Unreal (Auto, i materiali, "Keep its own maps");
  da riga di comando `export_unreal.py --material`.

Una prima versione con zone e maschere proprie (#79) è stata chiusa senza unirla: andava contro questa decisione.

## 10. GPU condivisa

I passi che calcolano sulla GPU (`picture`, `concept`, `generate`, `finish` con ridipintura, `rig`, i pezzi) prendono
`gpu_lock.gpu(<nome>)` solo intorno al processo che calcola, mai durante il controllo dell'installazione o
l'attesa di una scelta. Le funzioni che lo prendono già da sé (zone, luce tolta) non vengono avvolte una seconda volta.

## 11. Ordine delle PR

| PR | Contenuto | Mancanze chiuse |
|---|---|---|
| a | Impronte per passo; passi `export_unreal` e `checks`; `checks` tra i predefiniti; catalogo con questi quattro passi; test di parità per i passi del catalogo | 1 (in parte), 2, 11 (in parte) |
| b | Testo-immagine nel pacchetto e `scripts/text_image.py`; passi `picture`, `concept`, `rig`, `animate`; `replace` e candidati; strumenti MCP della sezione 6 | 1, 10, 11 |
| c | Studio: rifai con altri valori, seme, triangoli in Create, opzioni Unreal, soglie, scelta dei candidati nei lotti | 3, 4 |
| d | Passo `clips` e interruttori, bibbia su tre strade, pausa dopo il concept, arma e scudo, versioni Blender nella catena | 5, 6, 7, 9 |
| e | Materiali: passo, `--zones-file`, pannello | 8 |

Ogni PR aggiorna il catalogo, la tabella della sezione 12 e il test di parità.

## 12. Tabella di parità

S = saltare, C = correggere o sostituire l'uscita, R = rilanciare da solo. "Sì" ovunque: ogni casella dice come.
Nei lotti e per gli agenti saltare è togliere il passo dalla riga, rilanciare è `rerun_step` (o cambiare la riga:
riparte dal primo passo la cui impronta cambia), sostituire è `"replace": {"<passo>": "file"}` o la scelta fra i
candidati (`pick_candidate`, `picks.json`). Gli agenti usano gli stessi lotti (`run_steps`, una riga) più gli
strumenti MCP dei pezzi e della revisione.

| Funzione | Studio S/C/R | Riga di comando S/C/R | Agenti S/C/R |
|---|---|---|---|
| Testo-immagine | sì/sì/sì: Create parte anche da un'immagine propria; più immagini da scegliere; si rifà da Create | sì/sì/sì: `text_image.py` (`--count`, `--seed`) | sì/sì/sì: passo `picture` con candidati e scelta, `replace` |
| 2 Concept da riferimento | sì/sì/sì: pannello "Draw from a reference", facoltativo, si rifà | sì/sì/sì: `reference_concept.py make` | sì/sì/sì: passo `concept` con candidati, `replace` |
| 3 Bibbia dello stile (opzionale) | sì/sì/sì: casella in Create, nel pannello del concept e nei pezzi, spenta di base | sì/sì/sì: `--bible` su `text_image.py`, `reference_concept.py`, `slot_parts.py` | sì/sì/sì: `bible` in `picture`, `concept`, `pieces_run` |
| Modello 3D | sì/sì/sì: "Open GLB"; "Redo…" con il seme | sì/sì/sì: `pipeline.py` | sì/sì/sì: passo `generate`, `replace` |
| Finish | sì/sì/sì: "Redo…" con i triangoli; una versione Blender al suo posto | sì/sì/sì: `retopo_repaint.py` | sì/sì/sì: passo `finish`, `replace` |
| Versione Blender nella catena | sì/sì/sì: i passi dopo partono dalla versione scelta (#77) | sì/sì/sì: il suo `model.glb` è un file come gli altri | sì/sì/sì: `replace` con il `model.glb` e la sua provenienza |
| Rig | sì/sì/sì: "Redo…"; un rig da Blender si anima senza rifarlo | sì/sì/sì: `autorig.py rig` | sì/sì/sì: passo `rig`, `replace` |
| 6 Clip della famiglia | sì/sì/sì: "Family test clips" e "Longer takes" nel Rig | sì/sì/sì: `creature_clips.py --no-clips/--no-library/--no-variations`, `chain_retarget.py` | sì/sì/sì: passo `clips`, `replace` |
| Animazione | sì/sì/sì: le mosse si aggiungono, una per volta | sì/sì/sì: `autorig.py animate` | sì/sì/sì: passo `animate`, `replace` con una lista di GLB |
| 4 Pezzi (concept, pezzi, arma/scudo, fit, refit, Unreal) | sì/sì/sì: scelta del vestito, dei pezzi, di arma e scudo; "Fit again", "Carry to body", "Export for Unreal"; materiale per pezzo | sì/sì/sì: `slot_parts.py run --stop-at-concept/--stop-at-pictures/--resume --pick`, `fit`, `refit`, `unreal`, `materials` | sì/sì/sì: `pieces_run`, `pieces_pick`, `pieces_fit`, `pieces_refit`, `pieces_unreal` |
| 8 Materiali (uno per pezzo o asset) | sì/sì/sì: menu "Material" dell'export (Auto, un materiale, "Keep its own maps") e menu dei pezzi | sì/sì/sì: `export_unreal.py --material`, `slot_parts.py materials --set` | sì/sì/sì: passo `materials` (`material`), `pieces_materials` |
| 5 Esportazione Unreal | sì/sì/sì: "More options" (nome, collisione, LOD, maschera di tinta, direzione) | sì/sì/sì: `export_unreal.py` | sì/sì/sì: passo `export_unreal` |
| 7 Controlli e revisione | sì/sì/sì: soglie per esecuzione nel foglio di revisione, decisione della persona | sì/sì/sì: `check_asset.py --set` | sì/sì/sì: passo `checks`, `review_queue`, `review_item`, `review_decide` in MCP |
| 1 Lotti | sì/sì/sì: pagina Batches con scelta dei candidati e "rifai un passo" | sì/sì/sì: `pipeline.py --run-batch` con ripresa e `--retry-failed` | sì/sì/sì: `submit_batch`, `run_steps`, `pick_candidate`, `rerun_step` |

Il test di parità (`tests/test_step_catalog.py` e i test di ogni passo) confronta il catalogo con i flag delle righe
di comando nei due sensi.

## 13. Stato

Tutte le PR hanno avuto una revisione indipendente con un agente; i P1 trovati sono stati corretti prima del merge
(quelli trovati dopo, in PR di seguito).

| PR | Stato |
|---|---|
| a | Unita (#60): impronte per passo, passi `export_unreal` e `checks`, catalogo |
| a-fix | Unita (#63): i lotti vecchi non rigenerano più quando `checks` entra fra i predefiniti; file nelle opzioni per contenuto; controlli ripetuti ri-accodati |
| b1 | Unita (#62): strumenti MCP per la coda di revisione e per i pezzi (`slot_parts/runs.py` condiviso con lo studio) |
| b2 | Unita (#66): testo-immagine nel pacchetto e `scripts/text_image.py`; passo `picture` con candidati, attesa (`waiting`, uscita 3) e scelta (`pick`, `picks.json`); MCP `describe_steps`, `run_steps`, `pick_candidate`, `rerun_step`; test di parità nei due sensi |
| b3a | Unita (#67): `image_to_3dlab/rigging.py` condiviso con lo studio, `scripts/autorig.py`, passi `rig` e `animate`; le clip arrivano all'export Unreal |
| b3b | Unita (#68): passo `concept` (funzione 2) con candidati; `concept/requests.py` condiviso con lo studio; chiave dei candidati = impronta del passo senza la scelta; flag della riga di comando nel catalogo |
| b3c | Unita (#69): `replace` nei lotti e in `run_steps` (sostituire l'uscita di un passo con un proprio file, versione Blender compresa) |
| c1 | Unita (#70): "Redo…" su 3D, Finish e Rig con i valori usati; passi "fatti da una versione precedente"; seme del 3D in Create e nel passo; triangoli del Finish in Create |
| c2 | Unita (#71): opzioni dell'export Unreal nello studio; soglie dei controlli nel foglio di revisione (le condizioni di licenza no) |
| c3 | Unita (#72, seguito #76): pagina Batches, scelta dei candidati e "rifai un passo"; righe che disegnano soltanto |
| d1 | Unita (#73, seguito #78): passo `clips`, interruttori nel Rig, `creature_clips.py --no-clips`, riga di comando di `chain_retarget.py` |
| d2 | Unita (#74): bibbia dello stile su tre strade, spenta di base |
| d3 | Unita (#75): scelta del vestito (`--stop-at-concept`), arma e scudo fra i candidati, "Fit again", "Carry to body", "Export for Unreal" nei pezzi |
| d4 | Unita (#77, seguito #82): una versione Blender va avanti a Finish, Rig e Animate, e quello che se ne fa resta nel suo asset con il suo id |
| e | Unita (#81): passo `materials`, un materiale per asset o pezzo (scelto o proposto), `pieces_materials`, "Keep its own maps" nello studio. #79 (zone proprie) chiusa: contro la decisione dell'utente |

## 14. Backlog (aggiornato a ogni PR)

Tutto il lavoro chiesto a questa sessione, una riga per voce, con lo stato: aperto / PR #N / unito. Va confrontato con
questa lista prima del resoconto finale.

| Voce | Stato |
|---|---|
| Parità: impronte, `export_unreal`, `checks`, catalogo (a) | unito (#60, #63) |
| Parità: revisione e pezzi via MCP (b1) | unito (#62) |
| Parità: testo-immagine, `picture`, candidati, MCP dei passi (b2) | unito (#66) |
| Parità: `rig`, `animate` (b3a) | unito (#67) |
| Parità: passo `concept` (b3b) | unito (#68) |
| Parità: `replace` (b3c) | unito (#69) |
| Parità: studio, Redo, seme, triangoli (c1) | unito (#70) |
| Parità: opzioni Unreal e soglie nello studio (c2) | unito (#71) |
| Parità: pagina Batches, scelte e "rifai un passo" (c3) | unito (#72, #76) |
| Parità: clip della famiglia (d1) | unito (#73, #78) |
| Parità: bibbia dello stile (d2) | unito (#74) |
| Parità: pezzi, scelta del vestito, arma e scudo, azioni sui set (d3) | unito (#75) |
| Parità: versioni Blender nella catena (d4) | unito (#77, #82) |
| Parità: materiali, un materiale per asset o pezzo (e) | unito (#81); #79 chiusa |
| Documento finale e tabella di parità | PR #80, da unire per ultima |
| ~~Set vero 1: spada presa per la lama~~ | superato: le armi non si legano più al personaggio (decisione dell'utente, 17/10 sera); resta il buono, sotto "Armi e scudi come asset autonomi". #83 chiusa |
| Set vero 2: concept senza i pezzi scelti non nominati | unito (#84) |
| Set vero 3: UV sovrapposte del Finish | unito (#85) |
| Set vero 4: lucchetto GPU (Pixal3D, giudice, sd-cli) | unito (#86) |
| Set vero 5: arma senza candidati | unito (#75) |
| Set vero 6: frammenti staccati | unito (#87) |
| Set vero 7: giudice mancante senza avviso | unito (#88) |
| Decisione A: come è vestito il corpo base (tutina, abiti semplici, intimo, testo libero), scelto alla registrazione su tre strade, in `body.json` | unito (#93) |
| Decisione B: nello studio un set si ferma dopo le immagini, scelta automatica evidenziata, un clic per il 3D; CLI e agenti con un flag | era già così; fissato con un test: unito (#89) |
| ~~Decisione C: arma e scudo su un socket della mano del corpo in Unreal~~ | superata dalla decisione dell'utente: "non dobbiamo collegare il personaggio alla sua arma, non è lo scopo di AssetFurnace; dobbiamo però generarle bene". Quello che #108 ha messo in main per il corpo (socket sulle mani, "Held by", fit alla mano, controllo della presa) si toglie nella PR delle armi autonome |
| ~~C, collegato: l'arma segue l'osso del dito `middle_02_r`~~ | superato: l'arma non si adatta più al corpo |
| Decisione D: giudice (llama.cpp + Qwen3-VL) installato nel checkout principale con il bootstrap; Setup lo mostra; il lavoro avvisa | installato il 17/10 (bootstrap_style.py --only llama, provato: risponde in 13 s); avviso e Setup: unito (#88) |
| Decisione E: mantello (nuovo slot con fisica della stoffa) rimandato | scritto nei seguiti del progetto dei pezzi (#80) |
| MCP 1: annotazioni degli strumenti (readOnly, destructive, idempotent, title) | unito (#109) |
| MCP 2: risultati tipizzati (outputSchema, structuredContent) | unito (#109) |
| MCP 3: file prodotti come resource link con mimeType | unito (#109; leggibili con `resources/read`, solo sotto `output/` o le roots) |
| MCP 4: notifiche di avanzamento; valutare i "tasks" dell'SDK | unito (#109): `wait_batch` con avanzamento; i tasks restano fuori (l'SDK 2.3 li esclude di proposito, i lavori in background fanno già il loro lavoro) |
| MCP 5: solo percorsi sotto il repo, `output/` o le roots del client | unito (#109; anche i percorsi dentro uno spec) |
| MCP 6: nome del server "assetfurnace", istruzioni, documentazione degli strumenti per funzione | unito (#109) |
| MCP 7: Streamable HTTP facoltativo su 127.0.0.1 con token, uso via tunnel SSH | unito (#109; provato davvero: 401 senza token, 200 con) |
| Pixal3D secondo la scheda ufficiale (12 passi, campo visivo stimato con MoGe-2) | unito (#104; MoGe-2 installato e provato: 23 s su GPU) |
| Review finale con Codex di tutto il lavoro, correzioni, resoconto | unito (#110): review per aree (lotti, MCP, studio, pezzi, Unreal, motori), 19 difetti veri corretti con un test ciascuno; due di #103 (altra sessione) segnalati, non toccati |
| Armi e scudi come asset autonomi, generati bene: posa canonica nel prompt (dritta, impugnatura in basso, punta in alto, fondo piatto) | unito (#111) |
| Armi: controllo dell'orientamento con il giudice Qwen3-VL, girata se l'impugnatura è in alto | unito (#111; sul set vero una spada su due girata) |
| Armi: candidati con punteggio come gli altri pezzi, pausa per la scelta nello studio | unito (#111) |
| Armi: socket "grip" sull'SM_ dell'arma, trovato con il grounding di Qwen3-VL sull'immagine canonica e proiettato sul 3D; correggibile su tre strade (studio, CLI, MCP) | unito (#111; grounding provato con llama.cpp: 0,5 s, riquadri giusti; sul manico lungo la classe dice dove si chiude la mano) |
| Armi: export Unreal come SM_ con il socket grip, scala reale, pivot sull'impugnatura, collisione sensata; in un set fatte accanto, come asset separati, non agganciate | unito (#111; il socket lo fa il controllo in UE: l'import glTF non legge SOCKET_grip) |
| Armi: togliere il legame al corpo messo da #108 (socket SOCKET_Weapon/SOCKET_Shield sul corpo, "Held by", altezza del portatore, fit alla mano, GRIP_REACH_CM) | unito (#111) |
| Armi: prova in UE di spada, ascia, arco e scudo come asset singoli (socket presente sull'impugnatura, pivot giusto, scala plausibile) | unito (#111): tutte e quattro PASS in UE 5.8 (100, 80, 140, 75 cm) |
| Audit 1: `batch.class_problem` rifiuta research-only anche senza intento commerciale; allinearlo a validate_run_policy (rifiuta solo se commerciale, altrimenti avvisa e registra) | unito (#112) |
| Audit 2: `slot_parts/sidecar.py` scrive la classe di licenza fissa (commercial-conditional); prenderla dal catalogo | unito (#112; research-only) |
| Audit 3: catalogo dei passi: aiuto di negative_prompt ("solo con cfg sopra 1"), dimensioni del passo picture fino a 2048 con i preset ufficiali (2048 senza riferimenti, 1024 con), parametri transparent e rewrite di #103, opzione two_sided dell'export Unreal | unito (#113) |
| Audit 4: Create nello studio offre i preset della scheda, 2048 predefinito per il solo testo | unito (#113) |
| Audit 5: le immagini dei pezzi senza riferimenti usano T2I_DEFAULTS | unito (#113) |
| Audit 6: `pixal3d_generate.py` cancella anche .ply e __matted.png una volta usati (811 MB inutili nel set vero) | unito (#114): .ply tolto; il ritaglio resta come solo contorno, perché la camera di corpo e arma lo legge |
| Audit 7: export Unreal senza avvisi del validatore glTF: tangenti scritte (MESH_PRIMITIVE_GENERATED_TANGENT_SPACE), mesh skinnata come nodo radice (NODE_SKINNED_MESH_NON_ROOT) | unito (#115; corpo ed elmo del cavaliere: 0 errori, 0 avvisi; in UE tutto PASS) |
| Lint F (ruff) entrato in main con #84, #87, #88 | unito (#112) |
| Review finale 2 (coordinatore, d6f7057..106fd90), lotti: `validate_run_policy` senza `commercial_intent` (un lotto commerciale passa il cancello di Qwen) | unito (#116) |
| Review finale 2, lotti: i controlli misurano la sorgente non scalata contro la misura dell'export | unito (#116) |
| Review finale 2, lotti: lo staging del concept sovrascritto da un riferimento (`drawn_` che collide) | unito (#116) |
| Review finale 2, lotti: concept → generate → finish rifiutato dal preflight | unito (#116) |
| Review finale 2, lotti: i LOD trovati da soli fuori dall'impronta (un LOD cambiato non riesporta) | unito (#116) |
| Review finale 2, GPU: annullare il rig su macOS/Linux lascia vivo SkinTokens | unito (#117) |
| Review finale 2, GPU: Pixal3D e il giudice lasciano il lucchetto con il figlio ancora vivo | unito (#117) |
| Review finale 2, GPU: il recupero di un lucchetto stantio può togliere quello di un lavoro vivo | unito (#117) |
| Review finale 2, studio: un'immagine annullata aspetta e riscrive comunque il prompt | unito (#117) |
| Review finale 2, pezzi: due avvii insieme senza prenotazione (studio e MCP) | unito (#118) |
| Review finale 2, pezzi: il concept in cache ignora l'immagine del corpo | unito (#118) |
| Review finale 2, pezzi: solo armi in modo 2 chiede un concept (API dello studio) | unito (#118) |
| Review finale 2, MCP: i link di una cartella di rete non si leggono con `resources/read` | unito (#118) |
| Review finale 2, installer: torch 2.14.1 chiesto all'indice cu128, che arriva a 2.11.0 su Windows | unito (#119): `requirements-cu128.lock` con torch 2.11.0 (un `setuptools<82` impediva di togliere solo torch dal lock) |
| Review finale 2, installer: il ripiego su `requirements.txt` va in errore | unito (#119) |

## Seguiti

- Dalla prova delle armi autonome (17/10, #111):
  - Pixal3D ha dato all'arco lungo una curvatura profonda 62 cm, più di un arco vero. La lunghezza è giusta (140 cm), la forma da una sola vista no: è un limite del modello sugli oggetti sottili, non dell'impugnatura.
  - L'import glTF di UE 5.8 non trasforma un nodo `SOCKET_grip` in un socket della static mesh: lo fa il controllo in Unreal, e chi importa a mano lo aggiunge (il pivot è già sull'impugnatura).
- Codex ha revisionato #108 e #109 (un giro ciascuna) e poi tutto il lavoro, per aree, su main (#110). Le PR di
  prima avevano avuto una revisione indipendente con un agente.
- Da #103 (testo-immagine Qwen-Image 2.1, altra sessione), trovati dalla review finale e lasciati a chi l'ha scritta:
  `negative_applies` scarta il prompt negativo anche sotto cfg 1, mentre il suo commento dice che
  stable-diffusion.cpp lo usa quando la guida non è 1; nello studio un'immagine che sta riscrivendo il prompt non si
  può annullare mentre aspetta la GPU.
- Lo studio non usa il lucchetto della GPU e non vede i lotti in corso (`docs/batches.md`): resta così finché lo
  studio non ha un gestore di lavori unico. Fuori da questo lavoro.
- Piccole cose trovate dalle revisioni e lasciate (P2, nessuna cambia un risultato):
  - con le clip spente il Rig lancia comunque `creature_clips.py --no-clips` per il file della famiglia: se quel
    passo si rompe, il rig risulta fallito anche se il GLB è scritto;
  - con le clip spente il messaggio finale di un rig non umanoide dice "No test clips for this family yet";
  - una bibbia senza parole per quel tipo disegna di nuovo i candidati di una riga invece di riusarli.
- Un set portato su un altro corpo ("Carry to body", `refit`) non ha controlli su quel corpo: i controlli del primo
  corpo non valgono lì e non vengono copiati. Rifarli sul nuovo corpo è un passo da aggiungere ai pezzi.
- Il Finish dello studio non scrive la licenza del modello che riceve (anche per una versione Blender): è così da
  prima di questo lavoro, e la licenza resta nella versione e nel rig.
- La prova di un set vero di pezzi (`red-gold-knight`, 17/10) ha trovato sette difetti, corretti in PR a parte: la
  spada presa per la lama (#83), il concept che non disegnava i pezzi scelti ma non nominati dal vestito (#84), le UV
  sovrapposte del Finish (#85), il lucchetto della GPU non preso da Pixal3D, dal giudice e dai disegni dei pezzi (#86),
  i frammenti staccati (#87), il giudice mancante senza avviso (#88); l'arma senza candidati era già coperta da #75.
