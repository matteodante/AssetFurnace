# Funzione 4: personaggi in pezzi e equipaggiamento su un corpo base (progetto)

Data: 16 ottobre 2026, aggiornato la sera dello stesso giorno con i tre modi e l'esito della seconda prova della strada C.
Stato: **progetto approvato (PR #20); questa revisione aspetta il via libera**.
Piano: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 4.
Prove che lo precedono: [`2026-10-15-slot-parts-spike.md`](2026-10-15-slot-parts-spike.md), con l'appendice sulla strada C.

In una riga: nello studio si sceglie **come** fare un personaggio, fra tre modi.
1. Un pezzo solo.
2. Vestito, diviso in pezzi che combaciano.
3. Corpo nudo più set di equipaggiamento da indossare, come in 4Story e in Auroconts.

I modi 2 e 3 usano gli stessi slot, lo stesso adattamento al corpo in Blender, gli stessi controlli in tre pose e la
stessa uscita per Unreal.

## 1. Decisioni prese (16/10/2026)

Approvate dall'utente, riferite dal coordinatore:

- **Tre modi separati**, scelti nello studio (sezione 1b).
- **L'utente sceglie quali pezzi e quanti**, slot per slot.
- **Il concept sul corpo si fa con `-r corpo.png`**, come ha trovato la funzione 2, al posto dell'inpainting in due passate.
- **Immagini con Qwen-Image 2.1 Turbo** (ricetta predefinita dalla PR #21: `--sigmas qwen-image-2.1-turbo`, cfg 1.0,
  euler, 1024 px). AssetFurnace non è commerciale: la licenza di Qwen va bene. FLUX.2 klein 4B (Apache-2.0) resta
  registrato come alternativa.
- **I pezzi (modi 2 e 3) si fanno con la strada C**: un'immagine pulita per pezzo, ricavata dal ritaglio del concept
  con Qwen Turbo, poi Pixal3D per pezzo. Lo dice la seconda prova (sezione 10). B0 resta come riserva per slot.
  HoloPart e OmniPart restano esclusi.

## 1b. I tre modi

| | 1. Un pezzo solo | 2. Vestito, diviso in pezzi | 3. Corpo nudo + equipaggiamento |
|---|---|---|---|
| Cosa si genera | Il personaggio vestito | Il personaggio vestito | Il corpo base nudo (calzamaglia), poi uno o più set di pezzi su quel corpo |
| Corpo sotto | No | Sì: il corpo base su cui è nato il concept; se il concept arriva da fuori, il corpo si ricava togliendo l'armatura con una modifica `-r` ("stessa persona, stessa posa, in calzamaglia"), poi Pixal3D e rig | Sì: è il punto di partenza |
| Pezzi | Nessuno | Gli slot scelti, ricavati dal concept del vestito | Gli slot scelti, set per set; i set si cambiano pezzo per pezzo |
| Adattamento al corpo, maschera del corpo coperto | No | Sì | Sì |
| Controlli | Quelli di oggi (funzione 7) | Più `fit.clipping` e `fit.holes` | Come il 2 |
| Uscita per Unreal | Un modello rigato | Corpo più pezzi sullo stesso scheletro | Come il 2, con più set per corpo |
| Codice nuovo | Nessuno: è il flusso attuale, con un nome nello studio | Tutto quello di questo progetto | Come il 2, più la libreria dei corpi e dei set |

Il modo 1 esiste già. **Il 2 e il 3 sono lo stesso motore**. Nel 2 il concept del vestito nasce sul corpo base e si
butta dopo l'uso; nel 3 il corpo resta e si generano più set sopra. La differenza sta nello studio e nella libreria,
non nella pipeline.

## 2. Cosa c'è già

| Cosa | Dove | Stato |
|---|---|---|
| Profilo dei controlli per un pezzo | `image_to_3dlab/checks/profiles/equipment_part.json` | Su `main`: facce ≤ 8000, 1 materiale, buchi più tolleranti |
| Uscita Unreal di un pezzo | `scripts/export_unreal.py --kind equipment --body RIGGED.glb` | Su `main`: misura, verso e scheletro presi dal corpo |
| Deformazione in posa | controllo `rig.deformation` (funzione 6) | Su `main` |
| Clip per umanoidi | `image_to_3dlab/motions/` + `scripts/kimodo_retarget.py` (`walk`, `sword_swing`…) | Su `main` |
| Immagini | `sd-cli` con Qwen-Image 2.1 Turbo (`viewer/image_api.py`, ricetta della PR #21); modifica con riferimenti in `image_to_3dlab/concept/` (funzione 2) | Su `main` |
| Normali girate | `scripts/fix_winding.py` | Su `main`: serve per i pezzi che Pixal3D gira tutti all'interno |
| 3D, rig | Pixal3D, finitura, SkinTokens, `clean_skin_weights.py` | Su `main` |
| Script della prova | `journal/spike-slot-parts/` (locali) | Da ripulire: `roles.py`, `camera.py`, `slot_masks.py`, `part_images.py`, `flatten_cuts.py`, `make_config_c2.py`, `fit_set.py`, `piece_quality.py`; `segment_project.py` per la riserva B0 |

Niente pesi nuovi: tutto ciò che serve è già installato.

## 3. Gli ingressi

### Il corpo base

Un corpo base è un personaggio fatto con i passi che esistono:
- immagine frontale in T-pose con calzamaglia;
- Pixal3D a vista singola (serve la sua camera per proiettare);
- SkinTokens con i pesi puliti.

Nello studio un bottone **"Usa come corpo base"** lo registra in `output/bodies/<nome>/body.json`, con:
- l'immagine;
- il GLB rigato;
- la camera di Pixal3D;
- i ruoli delle ossa (testa, mani, piedi… trovati dalla forma dell'albero, come nella prova);
- due etichette libere (per Auroconts: razza e sesso).

Un corpo senza camera di Pixal3D o non umanoide viene rifiutato con il motivo.
Nel modo 2, con un concept arrivato da fuori, il corpo si ricava come detto nella sezione 1b e si registra allo stesso modo.

### La lista dei pezzi

Il catalogo degli slot è fisso. La scelta per slot è dell'utente.

| Slot | Come si fa il pezzo | Scelte |
|---|---|---|
| Elmo | C (immagine dedicata) | sì/no; rigido sulla testa dopo l'adattamento |
| Busto | C | sì/no |
| Spallacci | C | sì/no; **un pezzo o due** (sinistro e destro) |
| Guanti | C | sì/no; sempre due pezzi |
| Gambe | C | sì/no |
| Stivali | C | sì/no; sempre due pezzi |
| Arma | C, dal concept o da un prompt a sé | sì/no; mano destra o sinistra |
| Scudo | C, dal concept o da un prompt a sé | sì/no; avambraccio sinistro |

Per ogni slot si può passare a **B0** (il pezzo ritagliato dal 3D del personaggio intero): è la riserva, più veloce,
da usare per un'anteprima o quando l'immagine dedicata di uno slot non viene.

Uno slot non scelto resta corpo: i suoi pixel non diventano un pezzo e la maschera del corpo lì non nasconde niente.

**Mantello: non nella prima versione.** Da una vista frontale la schiena non si vede, e un mantello vuole la stoffa simulata.
Lo aggiungo dopo una prova a parte (in "Seguiti").

## 4. Il flusso (modi 2 e 3)

1. **Concept sul corpo**: `sd-cli` in modifica con `-r body.png`, ricetta del Turbo a 1024 px.
   - Il prompt è composto dai pezzi scelti: "aggiungi elmo, busto e stivali …, non cambiare altro".
   - Si generano più semi e l'utente sceglie l'immagine.
   - Corpo, posa e contorno restano quelli del corpo base, quindi lo scheletro proiettato combacia.
   - Nel modo 3, per un set nuovo sullo stesso corpo si rifà solo questo passo e quelli dopo.
2. **Maschera degli slot**: lo scheletro del corpo proiettato con la camera; ogni pixel va al segmento d'osso più vicino.
   Restano i soli slot scelti.
3. **Un ritaglio per slot**, appiattito su fondo grigio, con la scala e la posizione segnate. Per i pezzi doppi si
   tiene il lato destro del personaggio.
4. **Un'immagine pulita per pezzo**: modifica con `-r ritaglio.png`, "completa questo pezzo: …, niente corpo dentro".
   Più semi; l'utente può scartarne uno e rigenerarlo.
5. **Pixal3D per pezzo**, con il budget di facce dello slot, poi la finitura e il raddrizzamento delle normali.
6. **Posizionamento**: il riquadro del pezzo nella sua immagine, riportato al concept invertendo il ritaglio.
7. **Adattamento** in Blender, la tecnica della prova:
   - allineamento degli arti per i guanti;
   - gabbia con Shrinkwrap e Surface Deform;
   - pesi copiati dal corpo, al massimo 4 per vertice;
   - elmo, arma e scudo rigidi su un osso;
   - i pezzi doppi si adattano **per lato**, non per specchio: lo spallaccio specchiato era il peggiore (6%);
   - pulizia delle punte che la gabbia lascia vicino alle spalle;
   - maschera del corpo coperto.
8. **Riserva B0**, per gli slot che l'utente sposta su B0: Pixal3D sul concept intero, divisione per slot (maschera per
   le facce visibili, scheletro per quelle nascoste), chiusura dei bordi di taglio, poi lo stesso adattamento.
9. **Controlli** (sezione 5) in tre pose: riposo, camminata, fendente.
10. **Uscita**: una cartella per set e per corpo, `output/sets/<set>/<corpo>/`:
    - un GLB per pezzo;
    - il corpo con la maschera;
    - `<set>.parts.json`, con modo, slot scelti, strada per slot, corpo (nome e SHA-256), misure e licenze (provenienza).
11. **Riadattamento agli altri corpi**: lo stesso pezzo si adatta a ogni corpo; immagini e Pixal3D non si rifanno.
    Si fa una volta per corpo, perché pesi e scheletro sono di quel corpo.

### Codice

- `image_to_3dlab/slot_parts/`, tutto in Python puro e testato:
  - catalogo e scelta degli slot, ruoli delle ossa, camera, maschera, ritagli con la loro trasformazione;
  - posizionamento dal riquadro;
  - la parte pura dell'adattamento (assi degli arti, criteri della maschera, misure);
  - la divisione per la riserva B0.
- `scripts/blender_fit_parts.py`: solo le chiamate a `bpy` (bordi, gabbia, pesi, maschera, pose).
  Riceve un file JSON di lavoro e non contiene nomi, colori o misure di un set preciso.
- `scripts/slot_parts.py`: un solo comando con cinque sottocomandi: `body`, `cuts`, `pieces`, `fit`, `run`.
  Va elencato in `scripts/README.md`.

## 5. Controlli (funzione 7)

Due controlli nuovi, con `needs=("body",)`. `Asset` riceve un campo `body`: il corpo base registrato.

| Controllo | Che cosa misura | Soglie iniziali (dalle prove) |
|---|---|---|
| `fit.clipping` | Peggior quota, fra le tre pose, dei vertici della superficie esterna del pezzo che affondano nella pelle visibile | warn 3%, fail 10% |
| `fit.holes` | Quota del corpo nascosto dalla maschera che, in posa, non ha un pezzo sopra | warn 2%, fail 5% |

- Sono misure geometriche, quindi rispettano la regola "niente controlli sui pixel".
- Sul pezzo da solo valgono i controlli che ci sono già: `mesh.holes`, `mesh.loose_parts`, `mesh.flipped_faces`.
  Nella prova separano bene i pezzi interi (C) dai gusci aperti (B0).
- La **copertura** (quanta parte del corpo è nascosta) dipende dai pezzi scelti: va nei dettagli del sidecar, non è un controllo.
- `rig.deformation` gira su ogni pezzo come sugli altri modelli.
- Ogni pezzo entra nella coda di revisione.
- Le soglie restano provvisorie finché l'utente non ha giudicato i primi set.

## 6. Nello studio

Una pagina nuova, **Personaggi in pezzi**: un modulo mio, `viewer/studio/slot-parts.js`, più `viewer/slot_parts_api.py`.
Nei file condivisi metto solo le righe d'aggancio.

- Per prima cosa si sceglie il **modo**:
  - il modo 1 rimanda al flusso di oggi (Create, poi Steps);
  - il modo 2 parte da un concept del vestito (generato sul corpo, o caricato);
  - il modo 3 parte da un corpo base registrato.
- Poi gli slot con le caselle e le scelte della tabella sopra. Per ogni slot ci sono il prompt del pezzo e la strada
  (C predefinita, B0 di riserva).
- Si sceglie uno dei concept generati, poi le immagini dei pezzi; si avvia, con l'avanzamento come per gli altri lavori.
- **Anteprima**: il visore esistente mostra i pezzi sul corpo; le chip suonano le clip.
- Nel modo 3, una **libreria dei set** per corpo: si cambia un pezzo alla volta.
- Un bottone **"Adatta anche a…"** lancia il riadattamento sugli altri corpi registrati.

## 7. Consegna: PR piccole

| PR | Contenuto |
|---|---|
| 1 | Progetto (PR #20, unita) e questa revisione, con la seconda prova della strada C e FLUX.2 klein nel catalogo |
| 2 | Catalogo degli slot, ruoli, camera, maschera, ritagli; sottocomandi `body` e `cuts` |
| 3 | `-r` in `image_api.py` (riferimenti e file della visione); immagini per pezzo, Pixal3D per pezzo, normali; sottocomando `pieces` |
| 4 | Posizionamento dal riquadro e adattamento (gabbia, pesi, rigidi, pezzi doppi per lato, maschera); sottocomando `fit` |
| 5 | `fit.clipping` e `fit.holes` nella funzione 7, con le tre pose; `Asset.body`; sottocomando `run` e sidecar |
| 6 | Riserva B0: divisione per slot e chiusura dei bordi |
| 7 | Pagina **Personaggi in pezzi** nello studio: i tre modi, la libreria dei set |
| 8 | Uscita per Unreal e riadattamento a un secondo corpo; prova di accettazione |

Dettagli:
- **PR 3 e la funzione 2**: la funzione 2 ha messo su `main` `image_to_3dlab/concept/` (`edit.with_references` per
  `-r` e il file della visione, `prompts.dress` e `prompts.add_piece`, `maps.flatten`, il sidecar `.concept.json`).
  Lo uso così com'è e aggiungo solo il prompt "completa questo pezzo".
- **PR 8, Unreal**:
  - la maschera del corpo va nel colore dei vertici (attributo `covered`), che il materiale usa per nascondere;
  - i pezzi condividono lo scheletro del corpo (Leader Pose);
  - l'arma va su un socket della mano.
  Il resto c'è già nel profilo della funzione 5.

Ogni PR:
- test prima del codice;
- riga in `CHANGELOG.md`;
- script nuovi in `scripts/README.md`;
- Codex al massimo un giro facoltativo, solo P1 veri.

Non unisco io. Dopo ogni PR chiudo il turno con il comando e il risultato.

## 8. GPU e tempi

Per set di 6 slot, sul primo corpo, misurati nella seconda prova:
- concept 25 s a seme;
- 6 immagini per pezzo da 21 s;
- 6 Pixal3D da circa 150–250 s con la GPU libera, più 6 finiture da 22 s;
- adattamento e controlli su CPU, ~3 minuti.

In tutto **20–30 minuti di GPU per set**, contro circa 4 della riserva B0. Il riadattamento a un altro corpo è solo
CPU, ~2–3 minuti stimati.

Il lucchetto della GPU si prende solo mentre `sd-cli` o Pixal3D calcolano, e si libera dopo ogni lavoro.

## 9. Rischi

- **Tempo**: 6 Pixal3D a set. Per Auroconts sono 427 forme di armatura: va messo in lotti di notte (funzione 1).
- **Un solo corpo misurato**: con proporzioni diverse (razze), il riadattamento può aumentare le compenetrazioni.
  La PR 8 lo misura su un secondo corpo.
- **Braccia alzate**: non provate. Gli spallacci sono i pezzi più a rischio.
- **Immagini per pezzo**: tre semi su tre sono venuti con Turbo su questo set; su set più strani (mantelli, pezzi
  asimmetrici) può servire scegliere a mano. Per questo lo studio fa vedere le immagini prima di Pixal3D.
- **Concept che sposta il corpo**: se un seme cambia la posa, lo scheletro proiettato non combacia.
  L'IoU della sagoma del corpo dentro il concept lo scarta prima di continuare (geometria sulle maschere, non pixel).

## 10. Esito della seconda prova della strada C

Fatta il 16/10/2026: dettagli nell'[appendice del rapporto](2026-10-15-slot-parts-spike.md#appendice-seconda-prova-della-strada-c-16102026-sera).
- Qwen Turbo completa i ritagli: **6 pezzi su 6** su tre semi. FLUX.2 klein: 5 su 6, con i guanti girati.
- I pezzi C sono interi e chiusi: facce su un bordo aperto ≤ 0,2% contro 4–18% di B0; interno visto ≤ 0,2% contro 20–37%.
- Sul corpo misurano quanto B0: compenetrazioni ≤ 3,1% (spallaccio specchiato 6%), buchi 1,2–1,8%, copertura 71,5%.
- Il posizionamento dal riquadro funziona; l'allineamento al pezzo B0 no.
- FLUX.2 klein 4B resta nel catalogo, scaricabile su richiesta. I pesi di prova sono stati cancellati.

## 11. Come provo l'accettazione

1. Due corpi base registrati, per esempio uomo e donna della stessa razza.
2. Modo 3: un set scelto nello studio, con elmo, busto, spallacci separati, guanti, stivali, niente gambe, più una spada.
   Poi un secondo set sullo stesso corpo, con un pezzo cambiato.
3. Modo 2: un concept vestito caricato da fuori, diviso nei pezzi scelti.
4. `fit.clipping` e `fit.holes` almeno warn su tutti i pezzi, in tutte e tre le pose, sul primo corpo; misurati anche sul secondo.
5. In Unreal 5.8: i pezzi seguono il corpo nella camminata, il corpo sotto non buca, la spada sta nella mano.
   Ci sono un'immagine a metà passo e una a metà fendente.
6. Tutti i pezzi nella coda di revisione per il giudizio dell'utente.
7. `PYTHONPATH=. pytest -q` e `ruff check` verdi sui file toccati; i test Blender con `--factory-startup`.

## Seguiti

- Mantello: prova a parte, con la vista di schiena (funzione 2, caso 4) e la stoffa in Unreal.
- Pesi di HoloPart: cancellati dal coordinatore il 16/10 insieme alla cartella della prova in WSL.
