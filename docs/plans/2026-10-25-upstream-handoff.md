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
| `claude/cleanup-dead-code` | nostro `main` | nostro fork, [PR #1](https://github.com/matteodante/AssetFurnace/pull/1) | unita il 25/10 |
| `claude/upstream-fixes` | loro `main` | Bingeljell | 8 fix, rivista e corretta, pronta; la PR va aperta a mano ([link](https://github.com/Bingeljell/image-to-3dlab/compare/main...matteodante:AssetFurnace:claude/upstream-fixes?expand=1)), testo in [upstream-fixes-PR.md](2026-10-25-upstream-handoff/upstream-fixes-PR.md) |
| `claude/upstream-pieces` | loro `main` | Bingeljell | rivista e corretta; aspetta una prova su GPU vera |

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

8 commit, un fix per commit, ognuno col suo test. Il 25/10 una revisione indipendente ha
trovato 5 problemi, tutti corretti dentro i rispettivi commit. Test: 2 falliti prima e
dopo (già rotti da loro), 3750 passati. Ruff pulito. Testo pronto in
[upstream-fixes-PR.md](2026-10-25-upstream-handoff/upstream-fixes-PR.md).

- **Sicurezza:** ogni scrittura nello studio richiede l'origine dello studio e un header
  (`X-AssetFurnace-Request` o `X-AssetFurnace-Client`). Lo mandano le pagine, il plan runner,
  il loro MCP e `pod_smoke_test.py`. Su 127.0.0.1 risponde solo a nomi locali; dietro
  tailscale, cloudflared, ngrok o VS Code si aggiunge il nome in `I3D_ALLOWED_HOSTS`. Uno
  studio su un altro indirizzo si fida della rete. Non provato su RunPod.
- **Rig:** un `.blend` caricato non esegue script d'avvio, driver né lo script finale di
  Rigify.
- **Windows e processi:** UTF-8 dappertutto; processi figli chiusi con lo studio; niente
  segnali a pid 0 o -1; `install.ps1` dal lock, e se la build di PyTorch bloccata manca
  avvisa e installa come prima.
- **Finish:** UV senza isole sovrapposte (circa 10 % → meno di 0,1 %), margine 0.0002 → 0.002.
- **Non verificato qui:** Blender, Windows e PowerShell non c'erano.
- **Tolti nella riduzione:** codice d'accesso in rete, scritture atomiche della Libreria,
  SHA-256 di sd-cli e revisioni Qwen, Blender pulito nel Finish, percorsi con `/`, `bpy`
  5.0.1, file temporanei di pytest. La branch `claude/upstream-fixes-full` che li conservava
  va cancellata a mano su GitHub (la sessione cloud non poteva cancellare branch).

## Branch 3: "In pieces" (`claude/upstream-pieces`)

Un personaggio, un mostro o un oggetto disegnato esploso, costruito con un solo giro di
Pixal3D e diviso dal loro split dei prop sheet in modalità `--pieces`: una sola inclinazione,
nessuna rotazione, pezzi uniti solo se le mesh si toccano (entro l'1 % del foglio), pezzi
piccoli tenuti (0,3 %), disposizione conservata, punti d'attacco `SOCKET_<pezzo>_<vicino>`.
`--turn` è rifiutato in questa modalità. Fix collaterale: niente LOD più pesante del pezzo.
Circa 300 righe di codice e 320 di test; revisione indipendente fatta, 7 problemi corretti.
Testo pronto in [upstream-pieces-PR.md](2026-10-25-upstream-handoff/upstream-pieces-PR.md).

**Prima di aprire la PR**, sulla GPU: 3 soggetti (umanoide, mostro a quattro zampe, oggetto) ×
3 seed. Controllare: numero dei pezzi giusto (le mani ci sono?), pezzi non fusi, dritti,
disposizione e scala in `props.blend`, punti d'attacco vicino alle giunture, un nuovo split
dal pannello Steps che resta in modalità pezzi. Poi scrivere i risultati nel testo della PR.

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
