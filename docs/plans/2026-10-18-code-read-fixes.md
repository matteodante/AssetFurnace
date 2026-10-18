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
| A-1 | Campo Steps di Create che non fa niente | fatto: tolto il campo e `steps` dalle impostazioni dell'immagine; la Turbo fa sempre 8 passi |
| A-2 | Stato di Pieces solo in memoria | fatto: la pagina legge `output/sets/.run.json` (`runs.status`/`runs.cancel`), il codice d'uscita va in `<log>.exit` |
| A-3 | `install.sh` sceglie la release dai tag dell'upstream | fatto: `git ls-remote` del repository installato, come `install.ps1` |
| A-4 | `fit.holes` conta GLB vecchi | fatto: il set viene da `fit.json`; un pezzo che non si legge è nominato nei `details` |
| A-5 | Alfa < 0,5 dà MASK su qualsiasi asset | fatto: solo con `opacity_mask` nel job, che mette `unreal_set` per il corpo (`--opacity-mask` da riga di comando); `cutoff` al posto di 0,5 |
| A-6 | B0 ignora le distanze camera di MoGe-2 | fatto: la profondità usa la stessa scala di x (distanze delle camere e dimensione del concept) |
| A-7 | `concept_box` presume la dimensione della tela | fatto: il riquadro torna alla tela prima del concept |
| A-8 | Provenienza "Metal" su Windows | fatto: `text_image.runtime()` dal build della macchina (Metal, CUDA 12, Vulkan, ROCm) |
| A-9 | Materiale non riportabile ad Auto | fatto: "Auto (materiale della regola)" nel menu dei set fatti; sceglierlo toglie la correzione |
| A-10 | Versione Blender senza scheletro esportata come creatura | fatto: il tipo viene da `facts.skins` della versione |
| A-11 | MoGe-2 assente dalla provenienza | fatto: componente `Ruicheng/moge-2-vitl@<rev>`, MIT, quando ha scelto il FOV |
| A-12 | Cambio classe dell'arma non sposta la presa | fatto: la nuova classe ricalcola la presa dal riquadro del giudice (tenuto in `judge_box`) o dal retro dello scudo; un punto messo dalla persona resta |
| A-13 | FLUX.2 klein installabile ma non usato | fatto: tolto da Setup con il suo bootstrap |
| A-14 | Errori del giudice gestiti in quattro modi | fatto: `candidates.with_judge` per pezzi, arma, presa e orientamento. Respinta la parte "un server per set": il server tiene il lucchetto della GPU, aperto per tutto il set bloccherebbe sd-cli e Pixal3D |

## Altri punti dei rapporti

| # | Punto | Stato |
| --- | --- | --- |
| X-1 | Il controllo aggiornamenti tiene la risposta di un altro repository | fatto: la cache dice da dove viene; un 404 la azzera |
| X-2 | `install.ps1` interroga l'URL HTTPS invece del remote | respinto: anche il `fetch` subito prima usa lo stesso `$Repo`, quindi `ls-remote` non aggiunge un modo di fallire |
| X-3 | L'asse del guanto punta a un dito (P2-8) | fatto: `limb_axis`, dall'avambraccio al polso |
| X-4 | Le clip in loop della libreria perdono il fotogramma di chiusura (P2-9) | fatto: `closed_loop` rimette il primo fotogramma in coda; da guardare in Unreal |

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
