# Correzioni dalla lettura del codice del 18/10/2026

Lettura completa di `6042e60..7f3a57f` (produzione degli asset, studio e installazione). Ogni punto si verifica
sul codice prima di correggerlo; un punto che non regge ha una riga con il motivo. Una riga per punto, aggiornata
in ogni PR.

Stato: **fatto** (PR), **in corso**, **da fare**, **respinto** (motivo).

## P1

| # | Punto | Stato |
| --- | --- | --- |
| P1-1 | Armi e scudi di un set senza controlli, licenza e coda di revisione (`_finish_set` controllava solo i pezzi adattati) | fatto: profilo `weapon_or_shield`, provenienza legata al GLB finito, voce `<set>-<corpo>-<slot>` nella coda |
| P1-2 | Riscrittura del prompt attiva di default ma llama.cpp lo installa solo la scheda del giudice | fatto: il Setup di Qwen-Image installa llama.cpp (30 MB, MIT) e lo conta per "ready"; Create, il lavoro e la provenienza dicono cosa manca e dove installarlo |

## Resti delle decisioni cambiate

| # | Punto | Stato |
| --- | --- | --- |
| R-1 | Arma tenuta in mano (fit, controlli, `asset.py`, `equipment_part.json`, `orient.py`, progetto della funzione 4) | fatto: il fit rifiuta un'arma o uno scudo; un fit di prima che li elenca si rifà (`fitted_files`); tolti `worn`, il codice morto e i commenti; il progetto della funzione 4 rimanda alle armi autonome, che hanno la loro riga nel README |
| R-2 | Zone di materiale e maschera di tinta nella catena (materials, export, Setup, pannello Unreal) | fatto (strada 1 del rapporto: le zone restano una prova dichiarata, fuori dalla catena): docstring di `materials`, `material_zones` e `game_materials` corrette (SAM 3.1 predefinito), `apply_tint` spostata nel modulo delle zone senza la frase falsa su Unreal; export, passo `export_unreal`, pannello Unreal, Setup e Hugging Face dicono che la maschera la scrive solo la prova |

## Sicurezza (P2)

| # | Punto | Stato |
| --- | --- | --- |
| S-1 | `style_api` senza controllo dell'Host | fatto: risponde dopo `_allowed`, come ogni altro GET di `/api/` |
| S-2 | `bible` accetta qualsiasi cartella | fatto: dalle richieste dello studio solo un nome registrato o "-" (`allow_path=False`); la riga di comando tiene la cartella |
| S-3 | Bearer solo sugli endpoint vecchi | fatto: `check_request` lo accetta, e i cinque controlli copiati dell'header passano da lì con il messaggio della pagina |
| S-4 | Token nel cookie senza controllo | fatto: un `ASSETFURNACE_TOKEN` fuori da `[A-Za-z0-9_-]{24,}` si rifiuta all'avvio |
| S-5 | `path` dei lotti senza limiti | fatto: su un server aperto alla rete il percorso della spec e quelli che contiene restano nel repository (il controllo di MCP senza radici) |

## Altri P2

| # | Punto | Stato |
| --- | --- | --- |
| A-1 | Campo Steps di Create che non fa niente | da fare |
| A-2 | Stato di Pieces solo in memoria | da fare |
| A-3 | `install.sh` sceglie la release dai tag dell'upstream | da fare |
| A-4 | `fit.holes` conta GLB vecchi | fatto: il set viene da `fit.json`; un pezzo che non si legge è nominato nei `details` |
| A-5 | Alfa < 0,5 dà MASK su qualsiasi asset | da fare |
| A-6 | B0 ignora le distanze camera di MoGe-2 | da fare |
| A-7 | `concept_box` presume la dimensione della tela | da fare |
| A-8 | Provenienza "Metal" su Windows | da fare |
| A-9 | Materiale non riportabile ad Auto | da fare |
| A-10 | Versione Blender senza scheletro esportata come creatura | da fare |
| A-11 | MoGe-2 assente dalla provenienza | da fare |
| A-12 | Cambio classe dell'arma non sposta la presa | da fare |
| A-13 | FLUX.2 klein installabile ma non usato | da fare |
| A-14 | Errori del giudice gestiti in quattro modi | da fare |

## Qualità (nessun cambio di comportamento)

| # | Punto | Stato |
| --- | --- | --- |
| Q-1 | Un modulo comune dello studio per richieste e percorsi (header, body, "dentro output/") | da fare |
| Q-2 | Una sola conversione glTF→Blender e un solo `pack_orm` | da fare |
| Q-3 | Dividere `blender_fit_parts.main` e `run_set`/`_finish_set` | da fare |
| Q-4 | Commenti falsi o che raccontano la storia | da fare |
| Q-5 | Test di parità fra le costanti del catalogo e i moduli d'origine | da fare |
| Q-6 | Parole intere nelle regex delle classi d'arma (bow/elbow, axe/waxed) | da fare |
| Q-7 | `batch.py`: hash dei file in memoria per percorso+mtime, niente calcolo doppio fra `fingerprint` e `step_fingerprints` | da fare |
