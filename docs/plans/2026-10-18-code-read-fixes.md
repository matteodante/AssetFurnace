# Correzioni dalla lettura del codice del 18/10/2026

Lettura completa di `6042e60..7f3a57f` (produzione degli asset, studio e installazione). Ogni punto si verifica
sul codice prima di correggerlo; un punto che non regge ha una riga con il motivo. Una riga per punto, aggiornata
in ogni PR. Tutti i punti sono chiusi: fatti, o respinti con il motivo.

Stato: **fatto** (PR), **in corso**, **da fare**, **respinto** (motivo).

## P1

| # | Punto | Stato |
| --- | --- | --- |
| P1-1 | Armi e scudi di un set senza controlli, licenza e coda di revisione (`_finish_set` controllava solo i pezzi adattati) | fatto (#120): profilo `weapon_or_shield`, provenienza legata al GLB finito, voce `<set>-<corpo>-<slot>` nella coda |
| P1-2 | Riscrittura del prompt attiva di default ma llama.cpp lo installa solo la scheda del giudice | fatto (#121): il Setup di Qwen-Image installa llama.cpp (30 MB, MIT) e lo conta per "ready"; Create, il lavoro e la provenienza dicono cosa manca e dove installarlo |

## Resti delle decisioni cambiate

| # | Punto | Stato |
| --- | --- | --- |
| R-1 | Arma tenuta in mano (fit, controlli, `asset.py`, `equipment_part.json`, `orient.py`, progetto della funzione 4) | fatto (#122): il fit rifiuta un'arma o uno scudo; un fit di prima che li elenca si rifà (`fitted_files`); tolti `worn`, il codice morto e i commenti; il progetto della funzione 4 rimanda alle armi autonome, che hanno la loro riga nel README |
| R-2 | Zone di materiale e maschera di tinta nella catena (materials, export, Setup, pannello Unreal) | fatto (#123) (strada 1 del rapporto: le zone restano una prova dichiarata, fuori dalla catena): docstring di `materials`, `material_zones` e `game_materials` corrette (SAM 3.1 predefinito), `apply_tint` spostata nel modulo delle zone senza la frase falsa su Unreal; export, passo `export_unreal`, pannello Unreal, Setup e Hugging Face dicono che la maschera la scrive solo la prova |

## Sicurezza (P2)

| # | Punto | Stato |
| --- | --- | --- |
| S-1 | `style_api` senza controllo dell'Host | fatto (#124): risponde dopo `_allowed`, come ogni altro GET di `/api/` |
| S-2 | `bible` accetta qualsiasi cartella | fatto (#124): dalle richieste dello studio solo un nome registrato o "-" (`allow_path=False`); la riga di comando tiene la cartella |
| S-3 | Bearer solo sugli endpoint vecchi | fatto (#124): `check_request` lo accetta, e i cinque controlli copiati dell'header passano da lì con il messaggio della pagina |
| S-4 | Token nel cookie senza controllo | fatto (#124): un `ASSETFURNACE_TOKEN` fuori da `[A-Za-z0-9_-]{24,}` si rifiuta all'avvio |
| S-5 | `path` dei lotti senza limiti | fatto (#124): su un server aperto alla rete il percorso della spec e quelli che contiene restano nel repository (il controllo di MCP senza radici) |

## Altri P2

| # | Punto | Stato |
| --- | --- | --- |
| A-1 | Campo Steps di Create che non fa niente | fatto (#125): tolto il campo e `steps` dalle impostazioni dell'immagine; la Turbo fa sempre 8 passi |
| A-2 | Stato di Pieces solo in memoria | fatto (#125): la pagina legge `output/sets/.run.json` (`runs.status`/`runs.cancel`), il codice d'uscita va in `<log>.exit` |
| A-3 | `install.sh` sceglie la release dai tag dell'upstream | fatto (#125): `git ls-remote` del repository installato, come `install.ps1` |
| A-4 | `fit.holes` conta GLB vecchi | fatto (#122): il set viene da `fit.json`; un pezzo che non si legge è nominato nei `details` |
| A-5 | Alfa < 0,5 dà MASK su qualsiasi asset | fatto (#126): solo con `opacity_mask` nel job, che mette `unreal_set` per il corpo (`--opacity-mask` da riga di comando); `cutoff` al posto di 0,5 |
| A-6 | B0 ignora le distanze camera di MoGe-2 | fatto (#126): la profondità usa la stessa scala di x (distanze delle camere e dimensione del concept) |
| A-7 | `concept_box` presume la dimensione della tela | fatto (#126): il riquadro torna alla tela prima del concept |
| A-8 | Provenienza "Metal" su Windows | fatto (#125): `text_image.runtime()` dal build della macchina (Metal, CUDA 12, Vulkan, ROCm) |
| A-9 | Materiale non riportabile ad Auto | fatto (#125): "Auto (materiale della regola)" nel menu dei set fatti; sceglierlo toglie la correzione |
| A-10 | Versione Blender senza scheletro esportata come creatura | fatto (#125): il tipo viene da `facts.skins` della versione |
| A-11 | MoGe-2 assente dalla provenienza | fatto (#126): componente `Ruicheng/moge-2-vitl@<rev>`, MIT, quando ha scelto il FOV |
| A-12 | Cambio classe dell'arma non sposta la presa | fatto (#126): la nuova classe ricalcola la presa dal riquadro del giudice (tenuto in `judge_box`) o dal retro dello scudo; un punto messo dalla persona resta |
| A-13 | FLUX.2 klein installabile ma non usato | fatto (#125): tolto da Setup con il suo bootstrap |
| A-14 | Errori del giudice gestiti in quattro modi | fatto (#126): `candidates.with_judge` per pezzi, arma, presa e orientamento. Respinta la parte "un server per set": il server tiene il lucchetto della GPU, aperto per tutto il set bloccherebbe sd-cli e Pixal3D |

## Altri punti dei rapporti

| # | Punto | Stato |
| --- | --- | --- |
| X-1 | Il controllo aggiornamenti tiene la risposta di un altro repository | fatto (#125): la cache dice da dove viene; un 404 la azzera |
| X-2 | `install.ps1` interroga l'URL HTTPS invece del remote | respinto (#125): anche il `fetch` subito prima usa lo stesso `$Repo`, quindi `ls-remote` non aggiunge un modo di fallire |
| X-3 | L'asse del guanto punta a un dito (P2-8) | fatto (#126): `limb_axis`, dall'avambraccio al polso |
| X-4 | Le clip in loop della libreria perdono il fotogramma di chiusura (P2-9) | fatto (#126): `closed_loop` rimette il primo fotogramma in coda; da guardare in Unreal |

## Il resto dei rapporti (fuori dall'elenco della richiesta)

| # | Punto | Stato |
| --- | --- | --- |
| Y-1 | `orient.py` specchia l'arma girata (`ImageOps.flip`) | fatto (#131): mezzo giro (`rotate(180)`) |
| Y-2 | La riscrittura del prompt non si annulla mentre gira | fatto (#131): il processo di llama-completion va al lavoro, che lo ferma al Cancel |
| Y-3 | MoGe-2 installa sempre PyTorch cu128 | fatto (#131): con un driver sotto CUDA 12.8 l'installazione lo dice (girerebbe sulla CPU); il pin cu128 resta, è la build provata |
| Y-4 | La pagina Batches non conosce `skipped` | fatto (#131): "Skipped" e "N skipped" nel riassunto |
| Y-5 | La review valida le soglie sempre contro "prop" | fatto (#131): contro il profilo richiesto (oggi lo stesso risultato: il controllo non dipende dal profilo) |
| Y-6 | Codice morto tenuto in vita dai test (`unreal_profile.stride`, `fit.coverage_report`, `slot_parts.orient_held`) | fatto (#131): tolti, i test usano `orient.orient_held` |
| Y-7 | Il nome del gioco nel testo dello studio | fatto (#131) |
| Y-8 | Nomi del vecchio progetto negli installer e nello User-Agent | fatto (#131) nei testi; la cartella predefinita `~/image-to-3dlab` resta, è quella delle installazioni esistenti |
| Y-9 | Escape HTML in 8 copie, rotte di `do_GET`/`do_POST` a tabella, `assets_api.list_assets` lungo, `HINTS` come tuple, compatibilità "legacy" in `batch.py`, Icosphere riconosciuta in tre modi, `piece_materials.outfit_words`, `masks.slot_for` per pixel, parametro `dressing` che copre la funzione | rimandati: rifattorizzazioni ampie che la richiesta non elenca; nessun difetto di comportamento |

## Qualità (nessun cambio di comportamento)

| # | Punto | Stato |
| --- | --- | --- |
| Q-1 | Un modulo comune dello studio per richieste e percorsi (header, body, "dentro output/") | fatto (#127): `viewer/request_utils.py` (controllo della richiesta, lettura del body, `inside_output` con una regola sola); `blender_versions.safe_path` resta più stretto e lo dice |
| Q-2 | Una sola conversione glTF→Blender e un solo `pack_orm` | fatto (#129): `image_to_3dlab/frames.py` per il fit, B0, il profilo Unreal e la presa nell'export; `unreal_profile.pack_orm` passa da `materials.pack_orm`; la distanza della maschera (6 cm) è `fit.COVER_GAP` per il fit e per `fit.holes` |
| Q-3 | Dividere `blender_fit_parts.main` e `run_set`/`_finish_set` | fatto (#130): `Fitter` con un metodo per passo (corpo, pezzi, maschera, spalle, corpo rifilato, file); `SetRun` e `_draw_concept`, `_offer_pictures`, `_resume_pictures`, `_finish_set(set_run, made)`. Il fit vero su `red-gold-knight` dà gli stessi byte prima e dopo (fit normale, spalle unite, riadattamento) |
| Q-4 | Commenti falsi o che raccontano la storia | fatto (#128): `slot_parts.py` (scelta dal punteggio, comandi `grip` e `materials`), socket `grip` (l'import di UE 5.8 non lo crea: il controllo dice chi l'ha fatto), `picture_point`, `_heading`, `slot_parts/__init__`, `clip_library`, `body.dressing_words`, `request-header.js`, Guida 6-7-8, `slot-parts.js`, `text_image` e README senza cronologia, righe unite in `fit.py` e `runs.py`; il log del giudice ora si chiude; MoGe-2 dice perché non stima; via un `if` ridondante in `bootstrap_style` |
| Q-5 | Test di parità fra le costanti del catalogo e i moduli d'origine | fatto (#128): c'erano già per MODES, SAMPLERS e le scelte del concept; aggiunti FAMILIES, PICTURE_SIZE e PICTURE_MAX (un test di guardia: sul codice di oggi passa) |
| Q-6 | Parole intere nelle regex delle classi d'arma (bow/elbow, axe/waxed) | fatto (#128): la classe è la fine di una parola (longbow, battle-axe, longspear, sledgehammer, greatclub, quarterstaff, greatsword, plurali), con un elenco di parole che finiscono come un'arma e non lo sono (elbow, rainbow, balance, spike, prod, grimace); waxed non conta più |
| Q-7 | `batch.py`: hash dei file in memoria per percorso+mtime, niente calcolo doppio fra `fingerprint` e `step_fingerprints` | fatto (#128): `_source()` comune; `remembered_digests()` tiene gli hash per percorso, mtime e dimensione solo dentro `with_picks` e `fingerprints()`, così un controllo più tardi rilegge un file cambiato |
