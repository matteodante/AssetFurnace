# Cosa portare nel repo pubblico: stato al 25/10/2026

Il repo pubblico è `Bingeljell/image-to-3dlab`; il nostro fork è `matteodante/AssetFurnace`.
Al 25/10 eravamo 696 commit avanti e 304 indietro, con 46 file in conflitto: una PR con
"tutto" non si può fare. Ogni PR verso di loro parte quindi dal loro `main` (0.5.2,
`001c75d`) e prende solo pezzi scelti.

## Cosa abbiamo fatto

1. **Esplorato tutte le differenze**, area per area: materiali e stile, creature e pezzi,
   Unreal e controlli, batch/MCP/fix. Risultato: le prove fallite e il codice morto vanno
   tolti, batch/MCP/pezzi d'armatura/Unreal restano nel fork, e verso di loro vanno i fix e
   una funzione piccola ("In pieces").
2. **Licenza Qwen:** decisione dell'utente di lasciar perdere la differenza con loro. Da noi il
   testo è tornato identico al loro, e verso di loro non portiamo niente sulle licenze.
3. **Tre branch pronte**, tutte pubblicate su `origin` e a nome di Matteo Dante, senza
   trailer (le righe di firma in coda al commit):

| Branch | Base | Dove va | Stato |
| --- | --- | --- | --- |
| `claude/cleanup-dead-code` | nostro `main` | nostro fork, [PR #1](https://github.com/matteodante/AssetFurnace/pull/1) | bozza aperta |
| `claude/upstream-fixes` | loro `main` | Bingeljell | pronta; la PR va aperta a mano ([link](https://github.com/Bingeljell/image-to-3dlab/compare/main...matteodante:AssetFurnace:claude/upstream-fixes?expand=1)), testo in [upstream-fixes-PR.md](2026-10-25-upstream-handoff/upstream-fixes-PR.md) |
| `claude/upstream-pieces` | loro `main` | Bingeljell | aspetta una prova su GPU vera |

La sessione cloud non poteva aprire PR su Bingeljell (repo non autorizzato), per questo il
link va aperto a mano.

## PR 1: pulizia del nostro fork (#1)

Circa 9k righe e 57 file in meno; niente di ciò che funziona oggi cambia.

- **Prove che non hanno superato la soglia, o spente:** PBR stimato, luce cotta, zone di
  tinta (e `game_materials.py --zones`), delight, set di etichette materiali col suo pannello,
  render multi-vista, gara dei giudici di stile (SigLIP 2 / CSD). Via anche le loro schede in
  Setup (Marigold, Qwen3.5, SAM); `bootstrap_style.py` installa solo llama.cpp; `transformers`
  non è più un requisito.
- **Editor guidato** (rimandato): `edit_document`, `edit_math`, `edit_probe`,
  `edit_topology`, `inspect_edit_asset.py`, `blender_edit_bridge.py`. L'apertura in Blender
  resta.
- **`migrate_set_outfits.py`**, usa e getta: un set vecchio senza parole d'outfit ora usa il
  default dello slot e lo dice.
- **Licenza Qwen** come la loro (commercial-conditional, "le immagini sono tue");
  `validate_run_policy` perde `commercial_intent`. Restano le note DINOv3, SF3D e BiRefNet.
- **Tracce private tolte:** percorsi locali, numerazione e link del piano di produzione
  privato, "MMORPG", nomi di nostri asset nei commenti. Il catalogo dei passi perde il numero
  `feature`, quindi `describe_steps` di MCP non ha più quella chiave.
- **Tenuti apposta:** controllo d'importazione in Unreal, controlli di fit, avvii Windows,
  `WINDOWS.md`, `docs/plans/*`.
- **Test:** 4409 passati, 5 falliti, gli stessi 5 di `main`, dovuti all'ambiente cloud.

## PR 2: fix verso il pubblico (`claude/upstream-fixes`)

15 commit, un fix per commit, ognuno col suo test. Test: 2 falliti prima e dopo, già rotti da
loro; 3753 passati. Ruff pulito.

- **Sicurezza:** una scrittura nello studio richiede `X-AssetFurnace-Request` (studio) o
  `X-AssetFurnace-Client` (il loro client MCP), e un `Origin` di un altro sito è rifiutato:
  una pagina web non può più avviare lavori o cancellare pesi. Uno studio su un indirizzo di
  rete (RunPod, LAN) chiede un codice d'accesso. Un `.blend` caricato non esegue i suoi script.
- **Windows e testo:** UTF-8 in tutte le API e nei processi figli; i processi figli si fermano
  quando lo studio viene chiuso (job object di Windows); niente segnali a pid 0 o -1;
  `install.ps1` installa da `requirements.lock` con PyTorch CUDA alla versione bloccata.
- **Dati e download:** scritture atomiche del file della Libreria; archivi sd-cli verificati
  con SHA-256; pesi Qwen da commit Hugging Face fissati.
- **Finish:** UV senza isole sovrapposte (da circa il 10 % a meno dello 0,1 %); Blender avviato con
  `--factory-startup`; percorsi con `/`; `bpy` 5.0.1 per l'autorig; pytest tiene i file
  temporanei solo dei test falliti.
- **Lasciati fuori:** fix dei fianchi in kimodo (loro hanno riscritto `character_frame`),
  metà Pixal3D di #96 (hanno la loro build NVIDIA), percorso autorig su Windows (già fatto da
  loro), avvii in italiano, gestore del SIGTERM (proteggeva solo il nostro lock GPU).
- **Da sapere per i revisori:** chi scrive nello studio senza uno dei due header riceve 403; il
  test UV (serve Blender) e quello del job object (serve Windows) qui erano saltati.

## Branch 3: "In pieces" (`claude/upstream-pieces`)

Un personaggio, un mostro o un oggetto disegnato "esploso", cioè con i pezzi staccati, costruito
con un solo giro di Pixal3D e diviso dal loro split dei prop sheet in modalità `--pieces`: una sola
inclinazione per tutti, nessuna rotazione, pezzi piccoli tenuti (0,3 %), disposizione del
disegno conservata, un punto d'attacco `SOCKET_<vicino>` per ogni coppia di pezzi vicini in
`props.json` e `props.blend`. I prop sheet non cambiano. Circa 170 righe di codice e 110 di test.

**Prima di aprire la PR**, sulla GPU: 3 soggetti (umanoide, mostro a quattro zampe, oggetto) × 3
seed. Controllare: numero dei pezzi giusto (le mani ci sono?), pezzi non fusi, dritti,
disposizione e scala tenute in `props.blend`, punti d'attacco vicino alle giunture, un
nuovo split dal pannello Steps che resta in modalità pezzi.

## Ancora da verificare

- **Qwen Turbo è davvero meglio del loro modello base a 8 passi?** Non è dimostrato: noi
  seguiamo la ricetta ufficiale del Turbo, ma "le loro immagini sono sfocate" non l'abbiamo
  mai misurato. Test alla cieca: [qwen_ab.py](2026-10-25-upstream-handoff/qwen_ab.py). Il
  modello base (`leejet/Qwen-Image-2.1-GGUF`, `qwen_image_2.1-Q8_0.gguf`, 7,7 GB) va scaricato a
  mano; lo script non scarica niente.
- **Controlli mesh:** noi troviamo più difetti (UV sovrapposte, budget, scala), loro riparano
  meglio le facce girate. Nostri difetti: un buco piccolo passa, le card a un lato e a due
  lati sono bocciate per errore, un granello di 12 triangoli passa. Dati:
  [mesh_ab_results.txt](2026-10-25-upstream-handoff/mesh_ab_results.txt).
- **Dopo, se servono:** controlli mesh (sistemati prima), poi creature + libreria di
  animazioni, poi Turbo + style bible. Ognuna come PR separata, ricostruita sul loro `main`.
