# Ogni passo saltabile, correggibile e rilanciabile, da studio, riga di comando e agenti (progetto)

Data: 17 ottobre 2026. Stato: **proposto**, in costruzione a piccole PR (sezione 9).

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

Niente strumento per dipingere le zone ("quello che ha fatto ha fatto"). Il passo `materials` si può saltare,
lanciare e sostituire con un proprio file sulle tre strade:

- passo di lotto `materials` che chiama `materials.make_game_materials` con le zone di `material_zones.find_zones`,
  oppure con `zones_file`;
- `zones_file` accetta un JSON `{zona: materiale}` con una maschera PNG di indici di zona nello spazio UV, oppure il
  `.materials.json` di un'esecuzione precedente corretto a mano; viene letto in `ZoneMap` e salta `find_zones`;
- `scripts/game_materials.py --zones-file`; pannello Materiali nello studio con "Usa un mio file"; opzione nei lotti e
  in `run_steps`.

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

## 12. Tabella di parità (obiettivo)

S = saltare, C = correggere o sostituire l'uscita, R = rilanciare da solo. Si aggiorna PR dopo PR; "sì" ovunque è la
fine del lavoro.

| Funzione | Studio S/C/R | Riga di comando S/C/R | Agenti S/C/R |
|---|---|---|---|
| Testo-immagine | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 2 Concept da riferimento | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 3 Bibbia dello stile (opzionale) | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| Modello 3D | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| Finish | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| Versione Blender nella catena | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| Rig | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 6 Clip della famiglia | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| Animazione | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 4 Pezzi (concept, pezzi, arma/scudo, fit, refit, Unreal) | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 8 Materiali | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 5 Esportazione Unreal | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 7 Controlli e revisione | sì/sì/sì | sì/sì/sì | sì/sì/sì |
| 1 Lotti | sì/sì/sì | sì/sì/sì | sì/sì/sì |

Stato reale, aggiornato a ogni PR: [sezione 13](#13-stato).

## 13. Stato

| PR | Stato |
|---|---|
| a | Unita (#60): impronte per passo, passi `export_unreal` e `checks`, catalogo |
| b1 | In revisione: strumenti MCP per la coda di revisione e per i pezzi (`slot_parts/runs.py` condiviso con lo studio) |

## Seguiti

- Codex non ha revisionato #60 (limite d'uso raggiunto); al suo posto una revisione indipendente con un agente.

- Lo studio non usa il lucchetto della GPU e non vede i lotti in corso (`docs/batches.md`): resta così finché lo
  studio non ha un gestore di lavori unico. Fuori da questo lavoro.
