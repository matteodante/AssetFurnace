# Funzione 7: controlli e coda di revisione (progetto)

Data: 15 ottobre 2026. Stato: **approvato e implementato** sul ramo `claude/feat-checks-review-744c44` (in attesa di
merge). Le compenetrazioni nelle pose restano per un secondo passo. Le differenze rispetto alla proposta sono segnate
con *Implementato*.
Piano di riferimento: il piano di produzione (privato), funzione 7.

In breve: un registro di controlli che danno tutti lo stesso JSON, un file di esito accanto a ogni GLB, e una coda
nello studio dove l'utente approva o rimanda con una nota. Tutto gira sul processore (CPU): niente GPU, niente modelli.

## 1. Il contratto

### Un controllo

Un controllo è una funzione Python `check(asset: Asset, config: dict) -> dict`. Restituisce sempre:

```json
{
  "check": "mesh.holes",
  "status": "fail",
  "value": 4.2,
  "threshold": {"warn": 1.0, "fail": 3.0},
  "message": "4.2% of faces touch an open edge (fail above 3.0%)",
  "details": {"boundary_edges": 1840}
}
```

| Campo | Obbligatorio | Cosa contiene |
|---|---|---|
| `check` | sì | Nome a punti: famiglia e controllo (`mesh.holes`, `budget.faces`, `material.baked_light`) |
| `status` | sì | `"pass"`, `"warn"` o `"fail"`. Nient'altro |
| `value` | sì | Il numero misurato (o un oggetto piccolo, per esempio `{"height": 1.82}`) |
| `threshold` | sì | Le soglie usate: `{"warn": x, "fail": y}` per un limite, `{"min": a, "max": b}` per una fascia |
| `message` | sì | Una frase in inglese che dice cosa è andato male e di quanto |
| `details` | no | Dati in più (valori per vista, percorsi delle immagini di confronto). Chi legge ignora le chiavi che non conosce |

`Asset` è un dataclass immutabile: percorso del GLB, riferimento facoltativo (GLB, immagine o misure in numeri),
percorso della provenienza se trovato, chiave dell'asset (id esterno del lotto o id dello studio).

Un controllo che non ha ciò che gli serve (niente riferimento, niente rig, validatore non installato) **non viene
eseguito** e finisce in `skipped` con il motivo. Così gli stati restano tre e un "pass" vuol dire sempre "misurato e
buono", mai "non misurato".

### Il registro

```python
from image_to_3dlab.checks import register, run_checks

@register("mesh.holes", needs=("glb",))
def holes(asset, config): ...

report = run_checks(asset, profile="prop", only=None)
```

`needs` dice cosa serve (`glb`, `reference_glb`, `reference_image`, `reference_numbers`, `rig`, `provenance`,
`gltf_validator`). La sessione della funzione 8 scrive i suoi controlli `material.*` in un suo modulo con la stessa
firma; dopo il merge li importo nel registro (una riga).

### L'esito su disco: `<nome>.checks.json`

Accanto al GLB, come il `.provenance.json`:

```json
{
  "schema_version": 1,
  "asset": "output/.../SK_Warrior_18k.glb",
  "asset_sha256": "…",
  "asset_key": "mob_0412",
  "reference": {"kind": "glb", "path": "…", "sha256": "…"},
  "profile": "creature",
  "ran_at": "2026-10-15T18:00:00+00:00",
  "status": "fail",
  "results": [ {"check": "…", "status": "…", "value": "…", "threshold": "…", "message": "…"} ],
  "skipped": [ {"check": "gltf.valid", "reason": "glTF-Validator is not installed"} ]
}
```

`status` complessivo = il peggiore dei risultati (`fail` > `warn` > `pass`).

### Profili e soglie

Le soglie stanno in file JSON per tipo di asset: `image_to_3dlab/checks/profiles/prop.json`, `creature.json`,
`equipment_part.json`. Una riga di lotto può sovrascriverne alcune (per esempio il budget di facce dello slot). Tutti i
numeri iniziali qui sotto sono **provvisori**: si tarano nel pilota.

## 2. I controlli

Prima di contare qualsiasi cosa la mesh si **salda per posizione** (`scripts/mesh_health.py::weld_by_position`): il
glTF divide i vertici a ogni cucitura delle UV, e senza saldatura si contano le isole delle UV invece della geometria.
È la trappola documentata in `mesh_health.py` e `ribbon_metric.py`.

### Pulizia della mesh

| Controllo | Metodo | Soglie iniziali |
|---|---|---|
| `mesh.flipped_faces` | Su ogni spigolo condiviso da due facce, le facce devono percorrerlo in versi opposti; conto gli spigoli dove non succede. Più il segno del volume: negativo = mesh rovesciata (stessa regola di `fix_winding.needs_inverting`) | warn > 0,1% degli spigoli, fail > 1% o volume negativo |
| `mesh.holes` | Percentuale di facce che toccano uno spigolo aperto (`ribbon_metric.boundary_stats`) | warn > 1%, fail > 3% (da `ribbon_metric`: chiuso 0%, qualche strappo 1-3%). Nei profili con pezzi aperti per scelta (foglie, stoffa) si alza |
| `mesh.loose_parts` | Componenti connesse sulla mesh saldata; "volante" = sotto lo 0,5% delle facce del pezzo più grande (stessa logica di `remove_loose_parts.py`) | warn ≥ 1 pezzo volante, fail se i pezzi volanti superano l'1% delle facce |
| `mesh.nonmanifold` | Spigoli con 3 o più facce (`mesh_health.measure`) | warn > 0, fail > 0,5% |
| `uv.overlap` | Disegno i triangoli delle UV su una griglia di 1024 × 1024 con OpenCV e conto i pixel coperti più di una volta. Anche: UV mancanti con una texture (fail), UV fuori da 0-1 (warn) | warn > 4%, fail > 10% (*Implementato*: conta ogni uso di un pixel condiviso, quindi i bordi delle isole che si toccano pesano il doppio; da qui warn 4 invece di 2). Solo warn nei profili che usano UV specchiate di proposito |

### Budget

| Controllo | Metodo | Soglie iniziali |
|---|---|---|
| `budget.faces` | Triangoli contati dagli accessor del glTF (lettura diretta, come `glb_forensics.read_gltf_json`) | Dal profilo o dalla riga del lotto: warn oltre il budget, fail oltre il budget + 25% |
| `budget.materials` | Numero di materiali e di primitive (una primitive = una chiamata di disegno) | prop ≤ 2, creatura ≤ 3, parte ≤ 1 |
| `budget.textures` | Lato massimo di ogni immagine, letto dall'intestazione PNG/JPEG dentro il GLB | warn oltre 2048, fail oltre 4096 |
| `budget.texel_density` | Pixel per metro = √(pixel della texture × area UV usata ÷ area della superficie in m²) | Fascia dal profilo (da tarare sulla scala del gioco); solo warn finché non è tarata |

Facoltativo, se `vendor/gltfpack/gltfpack.exe` c'è: `gltfpack -v` stampa triangoli, vertici, chiamate di disegno e
peso in byte. Lo uso come seconda lettura in `details` (provato oggi su un GLB di `output/`: 8.984 triangoli,
10.949 vertici, 1 chiamata di disegno, 1,7 MB).

### Misure rispetto al riferimento

`measure.height` e `measure.footprint` (ingombro in X e Z), dal riquadro che contiene la mesh, in metri (unità del
glTF, Y in alto). Il riferimento è un GLB (il modello originale convertito) oppure numeri nella riga del lotto.
Tolleranza relativa dal profilo: warn oltre ±5%, fail oltre ±15%.

### Somiglianza della sagoma

Il punto delicato: deve essere **simile ma non identica**.

**Metodo.**
1. Asset e riferimento vengono messi alla **stessa scala**: in ogni vista la sagoma viene adattata al riquadro (il lato
   più lungo riempie il 90%) e centrata, con una scala uguale sui due assi. La dimensione assoluta la giudica già
   `measure.height`: qui conta solo la forma, quindi lo stesso modello grande il doppio dà 1,00.
2. Proiezione **ortografica** (senza prospettiva) da 4 viste: fronte, lato, tre quarti (45°), alto. Non uso fronte,
   retro, sinistra e destra: in ortografica il retro è lo specchio del fronte e la sinistra della destra, quindi
   darebbero solo 2 viste vere.
3. Ogni vista diventa una maschera di 256 × 256. *Implementato*: non con `cv2.fillPoly`, come proposto. Con molti
   poligoni usa la regola pari-dispari, quindi dove due triangoli si sovrappongono si annullano, e una mesh chiusa
   (fronte e retro su ogni pixel) esce quasi vuota. Lo fa un rasterizzatore vettoriale in numpy
   (`geometry.raster_hits`), che fa l'unione dei triangoli.
4. Per ogni vista: **IoU** (intersezione diviso unione: quanta parte delle due sagome si sovrappone; 1 = identiche,
   0 = nessuna sovrapposizione). `value` = media delle 4 viste; `details` ha i valori per vista e 4 immagini di
   confronto (riferimento e asset in due colori).
5. Esito: fail sotto `min` (non si riconosce), fail sopra `max` (praticamente una copia), pass dentro la fascia. Anche
   una sola vista sotto `min_view` dà warn: un braccio sbagliato si vede di lato anche se il fronte torna.

Se il riferimento è solo un'immagine (un concept), si confronta solo la vista di fronte, con la maschera ricavata dallo
sfondo dell'immagine come in `compare_to_source.py`.

**Taratura, prima versione: prova esplorativa su 7 GLB già in `output/`** (codice usa e getta, fuori dalla repo;
normalizzazione per altezza 3D):

| Coppia | IoU medio | Vista peggiore |
|---|---|---|
| Stesso guerriero, originale contro versione finita a 9k facce | 0,97 | 0,96 |
| Stesso guerriero allargato del 15-25% su un asse | 0,85-0,86 | 0,79 |
| Due allargamenti diversi dello stesso guerriero | 0,79 | 0,71 |
| Due generazioni dallo stesso prompt | 0,45 | 0,41 |
| Umanoidi diversi fra loro | 0,23-0,56 | 0,16-0,48 |
| Umanoide contro tartaruga | 0,18-0,42 | 0,08-0,37 |

Cosa dice: l'IoU separa bene "stesso oggetto, ritoccato" (sopra 0,78) da "oggetto diverso" (sotto 0,56), ma **non
ordina bene le cose diverse tra loro** (una ragazza e una tartaruga a 0,42, due umanoidi a 0,23). Quindi è adatto a
"si riconosce come lo stesso oggetto", non a misurare "quanto è lontano".

**Taratura ripetuta dopo la normalizzazione al riquadro** (richiesta del coordinatore), sugli stessi 8 GLB con il codice
consegnato:

| Coppia | IoU medio | Vista peggiore |
|---|---|---|
| Stesso guerriero ingrandito ×2 (scala uniforme) | **1,00** | 1,00 |
| Stesso guerriero, originale contro versione finita a 9k facce | 0,96 | 0,96 |
| Stesso guerriero allargato del 15% (X) o del 25% (Z) | 0,87 | 0,77-0,79 |
| Due allargamenti diversi dello stesso guerriero | 0,78 | 0,69 |
| Due generazioni dallo stesso prompt | 0,52 | 0,47 |
| Umanoidi diversi fra loro | 0,27-0,53 | 0,16-0,48 |
| Umanoide contro tartaruga | 0,19-0,44 | 0,10-0,41 |

L'allargamento su un solo asse resta a 0,87: non è una questione di scala, ma di proporzioni cambiate, ed è giusto che
la sagoma lo veda. Con una scala uniforme il punteggio è 1,00.

**Fascia di partenza**: `min` 0,65, `max` 0,95, `min_view` **0,60** (prima 0,50: nei modelli ritoccati la vista peggiore
non scende sotto 0,69, in quelli diversi non sale sopra 0,48). Sopra 0,95 c'è solo lo stesso modello a un'altra
risoluzione. **Parte come solo warn**: uno stile più marcato (testa e mani più grandi) può scendere sotto 0,65 e restare
riconoscibile. Si tara nel pilota sulle approvazioni e i rimandi dell'utente: si sceglie la fascia che li separa,
lo stesso metodo previsto per il punteggio di stile della funzione 3. Il calcolo della fascia dai dati etichettati è
una funzione piccola e testata, non un numero a mano.

### Validità del glTF

`gltf.valid`: il **glTF-Validator** di Khronos, versione 2.0.0-dev.3.10, Apache-2.0. Uso il programma nativo per Windows
(`gltf_validator-2.0.0-dev.3.10-win64.zip` nella release GitHub), non il pacchetto npm: niente Node e niente codice di
collegamento. Errori → fail, avvisi → warn, note informative ignorate; il conteggio e i primi 10 messaggi vanno in
`details`.

Si installa con `scripts/bootstrap_gltf_validator.py`, copiato da `bootstrap_gltfpack.py`: dice cosa scarica, da dove e
quanto pesa, verifica una somma di controllo fissata e chiede conferma (`--yes` per gli agenti). Non sono pesi di un
modello, ma la regola "niente download senza un sì" vale lo stesso. Senza validatore il controllo va in `skipped`.

### Licenze dei componenti

`license.components` legge la provenienza e raccoglie ogni licenza che trova, anche annidata. Oggi in `output/` ci
sono **tre forme** di provenienza, e il controllo le legge tutte:

1. `<nome>.provenance.json` di `provenance.finalize_output` (licenza + `components`);
2. `<nome>.json` di Pixal3D (licenza + `components`);
3. la provenienza del rig, con `models` e una catena di `source_record` annidati.

E un buco: **i modelli finiti (`output/finish/...`) non hanno una provenienza propria**. Il legame c'è solo attraverso
`input/source.glb`, che ha lo stesso sha256 del GLB generato (è così che lo studio li collega in `assets_api.py`). Il
controllo segue quel legame; se la catena non si chiude, fail.

Regole:
- nessuna provenienza raggiungibile → **fail** (la provenienza è la regola della repo);
- Hunyuan/Tencent, classe `territory-restricted`, o un componente `blocked` caricato → **fail**;
- una licenza che non è nella tabella nota (`provenance.LICENSES` più le licenze dei componenti) → **warn**;
- licenze con condizioni (DINOv3, la citazione "Built with Qwen") → pass, con le condizioni elencate in `value`, così
  arrivano fino ai crediti del gioco. Un profilo può renderle warn.

### Compenetrazioni nelle pose di prova: secondo passo

Le basi ci sono solo in parte: `rig_check.py` (lo scheletro è umanoide?), le clip in `image_to_3dlab/motions/`, le
capsule del corpo di `arm_clearance.py`, Blender 5.2 senza interfaccia con `mathutils.bvhtree.BVHTree.overlap` per le
intersezioni fra triangoli. Ma il caso che conta davvero, l'armatura dentro il corpo, nasce con la funzione 4, che non
c'è ancora. **Propongo di non farlo ora** e di scriverlo quando la funzione 4 avrà i primi pezzi adattati. Il registro
lo prevede già (`needs=("rig",)`).

## 3. Cosa si riusa (verificato il 15/10/2026)

| Strumento | Stato su questo PC | Uso |
|---|---|---|
| glTF-Validator 2.0.0-dev.3.10 (Apache-2.0) | Release con zip per Windows verificata su GitHub, licenza verificata su npm. **Non scaricato** | `gltf.valid`, via bootstrap con conferma |
| gltfpack 1.3 (meshoptimizer, MIT) | Già in `vendor/gltfpack/`; `-v` provato | Seconda lettura del budget (chiamate di disegno, vertici, byte) |
| Analisi di meshoptimizer (cache dei vertici, overdraw) | Solo libreria C++, nessun collegamento Python nel venv | **Scartata per ora**: utile per le prestazioni, non per bloccare un asset difettoso |
| Blender 5.2 | Installato, trovato da `image_to_3dlab/blender.py` | Solo dove numpy non basta: intersezioni in posa (secondo passo). Il "3D-Print Toolbox" non è più incluso in Blender (è un'estensione da scaricare): non lo uso, uso direttamente le sue fondamenta (`bmesh`, `BVHTree`) |
| `scripts/mesh_health.py` | Letto | Saldatura per posizione, spigoli non-manifold, volume |
| `scripts/ribbon_metric.py` | Letto | `boundary_stats` per i buchi |
| `scripts/remove_loose_parts.py` | Letto | Logica delle componenti connesse |
| `scripts/fix_winding.py` | Letto | Regola del volume negativo |
| `scripts/glb_forensics.py` | Letto | Lettura del JSON del glTF, materiali, texture |
| `scripts/compare_to_source.py` | Letto | Maschera del soggetto da un'immagine, idea del confronto delle sagome |
| `scripts/mark_asset.py` | Letto | Modello del registro: una riga per evento, mai riscritto, chiave sha256, misure fotografate al momento del giudizio |
| Librerie nel venv | trimesh 5.1.1, numpy 1.26, OpenCV 4.11, scikit-image 0.26, Pillow 12.3, scipy 1.17 | Tutto il resto. Nessuna dipendenza nuova |

Nota: le funzioni di `scripts/` si importano aggiungendo `scripts/` al percorso, come già fanno `mark_asset.py` e
`viewer/assets_api.py`. Spostarle nel pacchetto sarebbe più pulito, ma tocca file non miei: lo propongo come lavoro a
parte.

## 4. La coda di revisione nello studio

### Il registro dell'asset: `output/review-log.jsonl`

Una riga per evento, mai riscritta (come `mark_asset.py`). Chiave: `asset_key` (id del lotto o id dello studio) più lo
sha256 del GLB, così una nuova versione dello stesso asset si riconosce.

| Evento | Quando | Contenuto |
|---|---|---|
| `queued` | I controlli hanno girato su una versione | sha256, esito complessivo, riassunto dei controlli |
| `approved` | L'utente approva | nota facoltativa |
| `sent_back` | L'utente rimanda | **nota obbligatoria** |

Lo stato di un asset è il suo ultimo evento. La coda mostra:
- **Da rivedere**: ultimo evento `queued`;
- **Rimandati**: ultimo evento `sent_back`, con la nota bene in vista. Restano in coda finché non arriva una nuova
  versione;
- una nuova versione (sha256 diverso) che passa i controlli torna in **Da rivedere** portandosi dietro tutte le note
  precedenti.

Così "un rimando torna in coda con la sua nota". La funzione 1 legge i `sent_back` per sapere quali righe del lotto
rifare e perché. Facoltativo: all'approvazione e al rimando scrivo anche un giudizio `good`/`bad` in
`output/verdicts.jsonl` con `mark_asset.build_entry`, così cresce il dataset previsto in `mark_asset.py`.

### Server

`viewer/review_api.py` con `handle(handler, output, mutation=...)`, lo stesso schema di `blender_edit_api.py`:

- `GET /api/review/queue` → elenco con stato, esito dei controlli e note;
- `GET /api/review/<chiave>` → storia completa e ultimo `.checks.json`;
- `POST /api/review/<chiave>/checks` → esegue i controlli (CPU, qualche secondo);
- `POST /api/review/<chiave>/decision` con `{"decision": "approved"|"sent_back", "note": "…"}`.

Per agganciarlo servono **due righe in `viewer/generate_api.py`** (in `do_GET` e `do_POST`, accanto a
`blender_edit_api.handle`). È un file condiviso: lo segnalo al coordinatore.

### Interfaccia

- Nell'intestazione un pulsante **Review (n)** accanto ad **Activity**, che apre un pannello come quello di Activity
  (`pages.js`).
- Selezionato un asset, il GLB va nella scena principale con la **rotazione automatica a 360°** (`OrbitControls.autoRotate`
  di three.js, già presente in `viewer/vendor/`).
- **Confronto**: una seconda vista piccola col riferimento (GLB o immagine), con la telecamera legata a quella
  principale, e sotto le 4 immagini delle sagome sovrapposte prodotte dal controllo.
- L'elenco dei controlli con un'etichetta colorata per esito e il messaggio.
- **Approva** e **Rimanda**; Rimanda non si invia senza nota.

File nuovi: `viewer/review_api.py`, `viewer/studio/review.js`. Ritocchi piccoli: `studio.html` (il pulsante),
`app.js` (l'aggancio), `studio.css` (un blocco di stile in fondo).

### Python e riga di comando

- `from image_to_3dlab.checks import run_checks, write_report` per la funzione 1, che aggancerà il passo `checks` in
  `cli.py` dopo il merge (io `cli.py` non lo tocco).
- `scripts/check_asset.py asset.glb [--reference ref.glb|ref.png] [--profile prop] [--key mob_0412] [--json]` scrive
  `<nome>.checks.json` e mette l'asset in coda. Codice d'uscita: 0 se pass o warn, 1 se fail.
- `scripts/review_queue.py --list | --approve KEY | --send-back KEY --note "…"`: la stessa coda, per agenti e terminale.

## 5. Alternative scartate

| Alternativa | Perché no |
|---|---|
| Sagome renderizzate con Blender | Secondi di avvio per ogni asset e Blender obbligatorio; la proiezione con OpenCV costa millisecondi e si testa senza nulla. Blender resta per ciò che serve davvero (pose) |
| Sagome con prospettiva dall'angolo del concept | Serve solo con un'immagine di riferimento, e `compare_to_source.py` lo fa già per l'occhio umano. Fra due GLB l'ortografica è più stabile |
| Confronto con embedding (CLIP, DINO) | Pesi da scaricare e un numero meno spiegabile; è il mestiere del punteggio di stile della funzione 3 |
| Momenti di Hu o distanza fra contorni | I momenti di Hu ignorano rotazione e scala, quindi troppo indulgenti; la distanza fra contorni la tengo come possibile seconda misura se l'IoU non basta nel pilota |
| glTF-Validator da npm | Richiede Node e un piccolo programma di collegamento; il binario nativo no |
| Data Validation di Unreal | Gira dentro Unreal: utile come ultimo controllo nel gioco, non qui |
| Stato della revisione in `.assets-meta.json` | Si riscrive e perde la storia; il registro a righe la tiene |
| Usare `verdicts.jsonl` come coda | Ha un altro scopo (gusto per il dataset) e non conosce gli stati di una coda |

## 6. Rischi

- **Soglie sbagliate all'inizio.** Per questo sono nei profili e la sagoma e la densità dei texel partono come solo warn.
- **Velocità sui GLB grezzi** (circa un milione di facce). *Misurato*: tutti i controlli tranne la sagoma impiegano
  15 s sul guerriero grezzo da 957k facce (erano 173 s finché le UV si disegnavano un'isola alla volta) e 0,7 s sulla
  versione finita da 9k facce. La sagoma aggiunge 2-7 s sul grezzo e 0,2 s sul finito.
- **Mesh aperte di proposito** (foglie, mantelli) e **UV specchiate di proposito**: gestite nei profili, non nel codice.
- **Provenienza eterogenea** (tre forme, più i modelli finiti senza file proprio): un test per ogni forma.
- **Conflitti di merge** su `generate_api.py`, `studio.html`, `app.js`, `CHANGELOG.md`, `scripts/README.md`: aggiunte
  piccole e in fondo.
- **Download del validatore**: serve un sì dell'utente prima di provarlo su un GLB vero.

## 7. Come provo l'accettazione

**"Un asset difettoso costruito apposta viene fermato dal controllo giusto."** Nei test costruisco con trimesh un asset
pulito (una capsula con UV e una texture piccola) che passa tutto, e poi una copia difettosa per ogni controllo:

| Difetto costruito | Deve fallire |
|---|---|
| Metà facce rovesciate | `mesh.flipped_faces` |
| Una zona di facce tolta | `mesh.holes` |
| Un granello staccato dal corpo | `mesh.loose_parts` |
| UV di due metà sovrapposte | `uv.overlap` |
| 200.000 facce in un profilo da 10.000 | `budget.faces` |
| 8 materiali | `budget.materials` |
| Texture di 8192 px | `budget.textures` |
| Scalato di 1,3 | `measure.height` |
| Copia identica del riferimento | `silhouette` (lato alto) |
| Una forma diversa (un cubo) | `silhouette` (lato basso) |
| GLB con un accessor rotto | `gltf.valid` (parser del rapporto testato su un rapporto vero salvato come fixture; prova dal vivo solo se il validatore è installato) |
| Provenienza con Hunyuan, o nessuna provenienza | `license.components` |

Il test controlla che fallisca **quel** controllo e, dove il difetto è isolato, solo quello. In più giro i controlli sui
GLB veri di `output/` (nessuna generazione) e riporto numeri e tempi.

**"Un rimando torna in coda con la sua nota."** Test sul registro: `queued` → `sent_back` con nota → l'asset è fra i
Rimandati con la nota → una nuova versione viene controllata → torna in Da rivedere con la nota precedente nella storia
→ `approved` → esce dalla coda. Lo stesso percorso attraverso l'API del server, come gli altri test di `viewer/`, e a
mano nel browser dello studio.

## 8. Cosa resta fuori da questa consegna

- Compenetrazioni nelle pose di prova (dopo la funzione 4).
- Analisi di meshoptimizer per le prestazioni.
- Taratura definitiva delle soglie (nel pilota).
- Registrazione dei controlli `material.*` della funzione 8 (dopo il suo merge: una riga).
- Provenienza propria per i modelli finiti (è un buco del passo di finitura, non di questa funzione: va proposto a parte).

## 9. Come è stato consegnato

| Pezzo | Dove |
|---|---|
| Registro, contratto, profili | `image_to_3dlab/checks/registry.py`, `profiles/*.json` |
| Asset e misure condivise | `image_to_3dlab/checks/asset.py`, `geometry.py` |
| Controlli | `mesh_checks.py`, `budget_checks.py`, `measure_checks.py`, `silhouette.py`, `validator.py`, `licenses.py` |
| Coda di revisione | `image_to_3dlab/checks/review.py` (`check_and_enqueue` è la chiamata per la funzione 1) |
| Riga di comando | `scripts/check_asset.py`, `scripts/review_queue.py`, `scripts/bootstrap_gltf_validator.py` |
| Studio | `viewer/review_api.py` (due righe in `generate_api.py`), `viewer/studio/review.js`, `review-model.js`, pulsante in `studio.html`, girata in `viewer.js` |
| Test | `tests/test_checks_*.py`, `tests/_check_fixtures.py`, `test_check_asset.py`, `test_viewer_review_api.py`, `test_studio_review.py`, `test_bootstrap_gltf_validator.py` |

Il glTF-Validator nativo è stato provato dal vivo: 0 errori sul guerriero finito, 2 avvisi sul rigato (spazio delle
tangenti generato a runtime), un errore `ACCESSOR_TOO_LONG` sul GLB rotto apposta.

*Implementato, in più*: le misure lette dai numeri di una riga di lotto hanno la precedenza su quelle del modello di
riferimento, e un controllo di misura senza il dato che gli serve (per esempio larghezza e profondità) viene saltato
invece di avvisare.

## 10. Seguiti

Punti della review di Codex rimandati per la regola del 15/10/2026 (i P2 si correggono solo se sono errori che un
utente vedrebbe), più il lavoro che spetta ad altre parti della repo.

- Decidere su un asset il cui GLB è stato cancellato o spostato riesce ancora; il giudizio in `verdicts.jsonl` viene
  comunque saltato e segnalato. Rimandato: con il file sparito l'asset non si può spedire, e oggi i test della coda
  usano percorsi finti. [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287008029)
- Le soglie passate come override non sono validate nel valore: `--set mesh.holes.fail=NaN` fa passare tutto.
  Rimandato: serve un valore scritto apposta; i nomi sbagliati sono già rifiutati.
  [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287008052)
- I prop ereditano la licenza dal `props.provenance.json` del run, che non ha un hash per prop: oggi passano al massimo
  come avviso. La correzione vera è nel passo dei prop (`viewer/props_api.py`, `scripts/finish_props.py`): scrivere
  accanto a ogni `<prop>_LOD0.glb` un sidecar con il suo `output.sha256`, che il controllo delle licenze legge già.
  [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287008015)
- Il controllo delle UV legge solo il primo set (TEXCOORD_0, quello che trimesh carica); una texture con `texCoord: 1`
  non viene controllata sul suo set. Rimandato: raro nei modelli generati.
  [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287209897)
- Il riferimento non è protetto da modifiche durante i controlli, a differenza del GLB controllato. Rimandato: serve una
  sostituzione nei pochi secondi della corsa. [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287209926)
- Due controlli contemporanei dello stesso GLB scrivono lo stesso `<nome>.checks.json`; uno dei due eventi può puntare
  al report dell'altro. Rimandato: serve un percorso del report per corsa, o il lock esteso alla scrittura.
  [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287209947)
- I modelli finiti prima del sidecar derivato della funzione 1 ereditano la licenza dal run generato senza un hash che
  leghi il file finito: passano al massimo come avviso. Quelli con il sidecar della funzione 1 sono legati.
  [Thread](https://github.com/matteodante/AssetFurnace/pull/4#discussion_r4287209881)
- Compenetrazioni nelle pose di prova: dopo la funzione 4.
