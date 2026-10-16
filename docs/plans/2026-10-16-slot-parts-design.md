# Funzione 4: pezzi per slot su un corpo base (progetto)

Data: 16 ottobre 2026. Stato: **progetto, in attesa di via libera**.
Piano: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 4.
Prova che lo precede: [`2026-10-15-slot-parts-spike.md`](2026-10-15-slot-parts-spike.md).

In una riga: si sceglie un corpo base e i pezzi che si vogliono (elmo sì, mantello no, spallacci separati…).
AssetFurnace disegna l'armatura sul corpo, ne fa un modello 3D unico con Pixal3D e lo taglia nei pezzi scelti.
Poi adatta ogni pezzo al corpo in Blender, lo controlla in tre pose e lo esporta per Unreal.
Armi e scudi possono venire da un'immagine a parte (la strada C).

## 1. Decisioni prese (16/10/2026)

Approvate dall'utente, riferite dal coordinatore:

- **Strada principale B0**: Pixal3D sul personaggio intero, poi divisione per slot. HoloPart non serve.
- **L'utente sceglie quali pezzi e quanti**, slot per slot.
- **La strada C resta come alternativa** per i pezzi a sé (armi, scudi): un'immagine per pezzo, poi Pixal3D sul pezzo.
- **Il concept sul corpo si fa con `-r corpo.png`**, come ha trovato la funzione 2, al posto dell'inpainting in due passate.
- **Una nuova prova della strada C** con un modello di modifica Apache-2.0 è approvata a parte (sezione 10).
  I pesi non si scaricano finché il disco C non ha spazio.

## 2. Cosa c'è già

| Cosa | Dove | Stato |
|---|---|---|
| Profilo dei controlli per un pezzo | `image_to_3dlab/checks/profiles/equipment_part.json` | Su `main`: facce ≤ 8000, 1 materiale, buchi più tolleranti |
| Uscita Unreal di un pezzo | `scripts/export_unreal.py --kind equipment --body RIGGED.glb` | Su `main`: misura, verso e scheletro presi dal corpo |
| Deformazione in posa | controllo `rig.deformation` (funzione 6) | Su `main` |
| Clip per umanoidi | `image_to_3dlab/motions/` + `scripts/kimodo_retarget.py` (`walk`, `sword_swing`…) | Su `main` |
| Immagini | `sd-cli` con Qwen-Image 2.1 (`viewer/image_api.py`) | Su `main`, **ma senza `-r`**: la funzione 2 ha provato i riferimenti a mano, il comando non li passa ancora |
| 3D, rig | Pixal3D, finitura, SkinTokens, `clean_skin_weights.py` | Su `main` |
| Script della prova | `journal/spike-slot-parts/` (locali) | Da ripulire: `roles.py`, `camera.py`, `slot_masks.py`, `segment_project.py`, `fit_set.py` |

Niente pesi nuovi per la strada B0: tutto ciò che serve è già installato.

## 3. Gli ingressi

### Il corpo base

Un corpo base è un personaggio già fatto con i passi che esistono:
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

### La lista dei pezzi

Il catalogo degli slot è fisso. La scelta per slot è dell'utente.

| Slot | Strada predefinita | Scelte |
|---|---|---|
| Elmo | B0 | sì/no; rigido sulla testa dopo l'adattamento |
| Busto | B0 | sì/no |
| Spallacci | B0 | sì/no; **un pezzo o due** (sinistro e destro) |
| Guanti | B0 | sì/no; sempre due pezzi |
| Gambe | B0 | sì/no |
| Stivali | B0 | sì/no; sempre due pezzi |
| Arma | C | sì/no; B0 se è già nel concept; mano destra o sinistra |
| Scudo | C | sì/no; B0 se è già nel concept; avambraccio sinistro |

Uno slot non scelto resta corpo: i suoi pixel non diventano un pezzo e la maschera del corpo lì non nasconde niente.

**Mantello: non nella prima versione.** Da una vista frontale la schiena non si vede, e un mantello vuole la stoffa simulata.
Lo aggiungo dopo una prova a parte (in "Seguiti").

## 4. Il flusso

1. **Concept sul corpo** (B0): `sd-cli` in modifica con `-r body.png`.
   - Il prompt è composto dai pezzi scelti: "aggiungi elmo, busto e stivali …, non cambiare altro".
   - Si generano più semi e l'utente sceglie l'immagine.
   - Corpo, posa e contorno restano quelli del corpo base, quindi lo scheletro proiettato combacia.
2. **Pixal3D** sul concept scelto: un solo modello con la texture.
3. **Maschera degli slot**: lo scheletro del corpo proiettato con la camera; ogni pixel va al segmento d'osso più vicino.
   Restano i soli slot scelti.
4. **Divisione** in un GLB per pezzo:
   - le facce visibili leggono la maschera;
   - le facce nascoste prendono l'osso più vicino dello scheletro 3D;
   - i pezzi doppi si dividono per lato.
5. **Chiusura dei bordi di taglio** in Blender: si riempie il buco e si dà un piccolo spessore.
   Nella prova questo passo mancava.
6. **Adattamento** in Blender, la tecnica della prova:
   - posizionamento;
   - allineamento degli arti per i guanti;
   - gabbia con Shrinkwrap e Surface Deform;
   - pesi copiati dal corpo, al massimo 4 per vertice;
   - elmo, arma e scudo rigidi su un osso;
   - maschera del corpo coperto.
7. **Strada C per i pezzi a sé**:
   - un'immagine del solo pezzo, con il prompt dell'utente, oppure ritagliata dal concept;
   - Pixal3D sul pezzo;
   - attacco rigido all'osso della mano o dell'avambraccio, con un verso predefinito;
   - il verso fine si aggiusta in Blender, con il giro già esistente (`blender_edit_api`).
8. **Controlli** (sezione 5) in tre pose: riposo, camminata, fendente.
9. **Uscita**: una cartella per set e per corpo, `output/sets/<set>/<corpo>/`:
   - un GLB per pezzo;
   - il corpo con la maschera;
   - `<set>.parts.json`, con slot scelti, strade, corpo (nome e SHA-256), misure e licenze (provenienza).
10. **Riadattamento agli altri corpi**: lo stesso pezzo, già diviso, si adatta a ogni corpo; il concept e Pixal3D non si rifanno.
    Si fa una volta per corpo, perché pesi e scheletro sono di quel corpo.

### Codice

- `image_to_3dlab/slot_parts/`: catalogo e scelta degli slot, ruoli delle ossa, camera, maschera, divisione.
  Contiene anche la parte pura dell'adattamento: posizionamento, assi degli arti, criteri della maschera e misure.
  Tutto in Python puro e testato.
- `scripts/blender_fit_parts.py`: solo le chiamate a `bpy` (bordi, gabbia, pesi, maschera, pose).
  Riceve un file JSON di lavoro e non contiene nomi, colori o misure di un set preciso.
- `scripts/slot_parts.py`: un solo comando con quattro sottocomandi: `body`, `split`, `fit`, `run`.
  Va elencato in `scripts/README.md`.

## 5. Controlli (funzione 7)

Due controlli nuovi, con `needs=("body",)`. `Asset` riceve un campo `body`: il corpo base registrato.

| Controllo | Che cosa misura | Soglie iniziali (dalla prova) |
|---|---|---|
| `fit.clipping` | Peggior quota, fra le tre pose, dei vertici della superficie esterna del pezzo che affondano nella pelle visibile | warn 3%, fail 10% |
| `fit.holes` | Quota del corpo nascosto dalla maschera che, in posa, non ha un pezzo sopra | warn 2%, fail 5% |

- Sono misure geometriche, quindi rispettano la regola "niente controlli sui pixel".
- La **copertura** (quanta parte del corpo è nascosta) dipende dai pezzi scelti: va nei dettagli del sidecar, non è un controllo.
- `rig.deformation` gira su ogni pezzo come sugli altri modelli.
- Ogni pezzo entra nella coda di revisione.
- Le soglie restano provvisorie finché l'utente non ha giudicato i primi set.

## 6. Nello studio

Una pagina nuova, **Pezzi per slot**: un modulo mio, `viewer/studio/slot-parts.js`, più `viewer/slot_parts_api.py`.
Nei file condivisi metto solo le righe d'aggancio.

- Si sceglie il corpo base, poi gli slot con le caselle e le scelte della tabella sopra.
  Per ogni slot ci sono il prompt del pezzo e la strada (B0 o C).
- Si scelgono uno dei concept generati e i semi.
- Si avvia: avanzamento come gli altri lavori.
- **Anteprima**: il visore esistente mostra i pezzi sul corpo; le chip suonano le clip.
- Un bottone **"Adatta anche a…"** lancia il riadattamento sugli altri corpi registrati.

## 7. Consegna: PR piccole

| PR | Contenuto |
|---|---|
| 1 | Questo progetto (solo documenti) |
| 2 | Catalogo degli slot, ruoli, camera, maschera, divisione e chiusura dei bordi; sottocomandi `body` e `split` |
| 3 | Adattamento (gabbia, pesi, rigidi, maschera del corpo); sottocomando `fit` |
| 4 | `fit.clipping` e `fit.holes` nella funzione 7, con le tre pose; `Asset.body` |
| 5 | Concept sul corpo con `-r` e strada C per i pezzi a sé; sottocomando `run` e sidecar |
| 6 | Pagina **Pezzi per slot** nello studio |
| 7 | Uscita per Unreal e riadattamento a un secondo corpo; prova di accettazione |

Dettagli:
- **PR 5 e la funzione 2**: `-r` serve anche alla funzione 2. Se il suo `generate(prompt, references, seed)` è già su `main`, lo uso.
  Altrimenti aggiungo a `image_api.py` solo il passaggio dei riferimenti e del file della visione, e lo dico alla sua sessione.
- **PR 7, Unreal**:
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

Per set, sul primo corpo, misurati nella prova:
- concept ~1 min a seme;
- Pixal3D ~2,5 min;
- divisione e adattamento ~3 min su CPU.

Il riadattamento a un altro corpo è solo CPU, ~2–3 minuti stimati.

Il lucchetto della GPU si prende solo mentre `sd-cli` o Pixal3D calcolano, e si libera dopo ogni lavoro.

## 9. Rischi

- **Licenza di Qwen-Image 2.1**: secondo la funzione 2 permette solo l'uso non commerciale, e il concept parte da lì.
  - Va deciso per tutto il progetto.
  - Se la prova della sezione 10 va bene, FLUX.2 klein 4B (Apache-2.0) può fare anche il concept sul corpo, senza cambiare il resto.
- **Un solo corpo misurato**: con proporzioni diverse (razze), il riadattamento può aumentare le compenetrazioni.
  La PR 7 lo misura su un secondo corpo.
- **Braccia alzate**: il guanto destro era già al 9–11% nella camminata. Nelle pose estreme potrebbe servire una gabbia più fine sul polso.
- **Bordi chiusi**: riempire il taglio può lasciare una piega visibile dove il pezzo tocca il corpo. Di solito è coperta, ma va guardata.
- **Concept che sposta il corpo**: se un seme cambia la posa, lo scheletro proiettato non combacia.
  L'IoU della sagoma del corpo dentro il concept lo scarta prima di Pixal3D (geometria sulle maschere, non pixel).

## 10. Prova della strada C con un modello di modifica Apache-2.0 (in attesa del disco)

- Download approvato per **FLUX.2 klein 4B** (~13 GB). Se serve per il confronto, anche **Qwen-Image-Edit-2511** (~19,5 GB).
  Entrambi andranno registrati come `WeightSet`.
- **Non scarico finché il coordinatore non dice che c'è spazio**: oggi su C ci sono 2,1 GB liberi.
- Prima dello scarico verifico, file per file:
  - licenza di codice e pesi, compresi i pesi che il codice scarica da solo;
  - dimensioni;
  - che la versione di `sd-cli` in `vendor/` li legga.
- La prova ripete le strade C1 e C3 sullo stesso set della prima prova, con le stesse misure.
  Domanda: con un modello di modifica migliore, i pezzi a sé (arma, scudo, e magari l'elmo) escono completi al primo tentativo?
- L'esito va in un'appendice del rapporto della prova.

## 11. Come provo l'accettazione

1. Due corpi base registrati, per esempio uomo e donna della stessa razza.
2. Un set scelto nello studio: elmo, busto, spallacci separati, guanti, stivali, niente gambe; più una spada dalla strada C.
3. `fit.clipping` e `fit.holes` almeno warn su tutti i pezzi, in tutte e tre le pose, sul primo corpo; misurati anche sul secondo.
4. In Unreal 5.8: i pezzi seguono il corpo nella camminata, il corpo sotto non buca, la spada sta nella mano.
   Ci sono un'immagine a metà passo e una a metà fendente.
5. Tutti i pezzi nella coda di revisione per il giudizio dell'utente.
6. `PYTHONPATH=. pytest -q` e `ruff check` verdi sui file toccati; i test Blender con `--factory-startup`.

## Seguiti

- Mantello: prova a parte, con la vista di schiena (funzione 2, caso 4) e la stoffa in Unreal.
- Pesi di HoloPart (3,8 GB) ancora in WSL: si possono cancellare.
