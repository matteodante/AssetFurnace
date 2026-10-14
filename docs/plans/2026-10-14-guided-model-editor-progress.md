# Avvio dell'implementazione: S0/RIG0

**15 ottobre, cambio di priorità:** sviluppo dell'editor guidato rimandato su richiesta dell'utente. Il lavoro corrente è il [flusso manuale Blender](2026-10-15-blender-roundtrip.md). I risultati sotto restano esperimenti precedenti, non la descrizione della funzione disponibile in Studio.

Data: 14 ottobre 2026. Collegato al [piano revisionato](2026-10-14-guided-model-editor.md).

Ultimo aggiornamento: 15 ottobre 2026. I risultati del primo incremento restano sotto come storico; la sezione finale registra il documento canonico e le prove reali con Blender aggiunti successivamente.

**Implementazione avviata; S0/RIG0 ancora in corso.** Questo primo incremento costruisce i controlli di ingresso e le funzioni numeriche riusabili. Non abilita ancora Modifica nella web app e non dichiara superati i gate del piano.

## Codice aggiunto

- [edit_glb.py](../../image_to_3dlab/edit_glb.py): lettura GLB con limiti di memoria e controlli di header, chunk, bufferView, accessors, stride, dati normalizzati e riferimenti. Mantiene distinte primitive, istanze, materiali, UV e vertici coincidenti. Legge matrici globali includendo antenati non deformanti; rifiuta cicli e gerarchie ambigue. Compressioni, sparse, morph e profili non gestiti hanno un motivo esplicito. Non è un validatore glTF completo.
- [edit_math.py](../../image_to_3dlab/edit_math.py): inversione della matrice di skinning combinata per i gesti in posa; blocco di pose instabili e spostamenti eccessivi; certificazione del bind comune; compensazione delle inverse bind dopo un cambio di articolazione; calibrazione del delta di rotazione locale. Nessuna normalizzazione automatica dei pesi o modifica delle clip.
- [edit_probe.py](../../image_to_3dlab/edit_probe.py) e [inspect_edit_asset.py](../../scripts/inspect_edit_asset.py): controllo ripetibile su un file reale, rapporto JSON con hash sorgente/output, ricostruzione del contenitore, confronto dei dati ricaricati e prova numerica bind/pesi. Scrivono esclusivamente in una cartella nuova e preservano la provenienza disponibile. Non richiedono GPU o download di modelli.

Il percorso di lettura è separato dalle utilità GLB storiche: Generate/Finish/Rig/Animate continuano a usare il codice precedente. Il futuro documento canonico può riutilizzare questi controlli; non viene dichiarato già implementato dal solo inventario.

## Risultati sui file reali

Ambiente: Windows, Python 3.11.14 del virtualenv, NumPy 1.26.4. Blender 5.2 rilevato; in questo incremento non è ancora stato eseguito un round-trip Blender.

| Caso | Geometria | Risultato effettivamente misurato |
| --- | --- | --- |
| Guerriero riggato 9k | 10.949 vertici render, 8.984 triangoli, 34 joint | JSON/BIN e array decodificati conservati; posa iniziale coerente col bind; errore massimo nella prova spostamento/pesi `1,01e-7`, sotto la tolleranza `1e-5` |
| Guerriero 18k, clip walk | 20.336 vertici render, 17.984 triangoli, 34 joint, una clip | Dati conservati anche nel file animato; prova spostamento/pesi con errore `1,25e-7`, sotto la tolleranza `1e-5` |
| Robot, secondo asset generato | 595.681 vertici render, 900.128 triangoli, statico | Contenitore, geometria e immagini WebP conservati. Questo non certifica prestazioni interattive su una mesh così grande |

Hash SHA-256 dei sorgenti:

- Guerriero 9k: `107ae14058c0b4ad7369c8376c55e261a004133bda126d3d5012037fed8bc6db`.
- Guerriero 18k walk: `b4322ebd412b250c949d286f7058e10dbfb5227c90c7e112979a6475617eef10`.
- Robot: `2ba95603fd2a01ad2ddbe9aed1f3a073566eca2f90f6cb240a991cc18afd17c8`.

Gli artefatti locali ignorati da Git si trovano in `output/editor-feasibility/2026-10-14-start/{warrior-9k,warrior-18k-walk,robot}/`: `report.json`, `roundtrip.glb` e sidecar di provenienza quando presente. Il report conserva il percorso sorgente completo; questi nomi non sono codificati negli strumenti.

**Limiti della prova:** la ricostruzione riguarda JSON/BIN e array decodificati, non ancora GLB → documento canonico → Blender → documento → GLB. Il controllo del bind usa la posa iniziale del file: quando non è coerente segnala `needs_bind_adapter`, senza dichiarare impossibile un recupero del bind autorevole. La prova di spostamento/pesi opera su matrici in memoria e fino a 32 vertici per primitiva; non è una modifica del gomito esportata o una verifica dell'animazione. Ogni rapporto mantiene `production_editing_ready: false`.

## Verifica automatica e uso

50 nuovi casi di test coprono conservazione di buffer/attributi, UV interleaved, pesi normalizzati, istanze, file malformati, input non supportati, bind non identità, posa singolare, limiti di spostamento, calibrazione della piega, output esclusivi, provenienza e CLI reale. Suite mirata con regressioni GLB/Kimodo e registro script: **233 test passati**. Su Windows viene usata l'opzione Python `-X utf8`: il test storico del registro legge alcuni docstring UTF-8 con l'encoding predefinito e fallisce altrimenti su due descrizioni già presenti.

```powershell
.\.venv\Scripts\python.exe scripts/inspect_edit_asset.py percorso/modello.glb --output-dir output/editor-feasibility/nuova-prova
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests/test_edit_math.py tests/test_edit_glb.py tests/test_edit_probe.py tests/test_scripts_registry.py tests/test_glb_turn.py tests/test_kimodo_retarget.py
```

Exit code della CLI: `0` inventario/ricostruzione del contenitore riusciti; `1` input non supportato o prova fallita; `2` errore di uso/I/O. Lo stato del bind e i gate ancora aperti si leggono nel JSON; exit code `0` non significa «editor pronto».

## Prossime prove necessarie

1. Implementare il documento canonico con buffer/identità e l'adapter Blender; dimostrare il round-trip di geometria, UV, materiali e rig. Verificare l'identità delle cuciture senza fondere strati coincidenti.
2. Collegare i calcoli al renderer Three r170: stesso risultato per modifica in posa, picking e calibrazione con propagazione ai figli. Il modulo numerico da solo non certifica il retarget legacy o la preview.
3. Fissare regioni/difetti nel corpus reale e provare fessura e distacco, inclusi tessuto aperto e trasferimento degli attributi. Il secondo asset inventariato non sostituisce il secondo caso reale di taglio richiesto dal piano.
4. Chiudere RIG0 con spostamento di un'articolazione esportata/ricaricata, recupero del bind ove necessario, mapping e retarget delle clip. Poi procedere alle sessioni S1 e all'interfaccia secondo le dipendenze già approvate.

Le regole di lettura sono basate sulla [specifica glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html); il contratto di deformazione è descritto nel [tutorial Khronos sullo skinning](https://github.com/KhronosGroup/glTF-Tutorials/blob/main/gltfTutorial/gltfTutorial_020_Skins.md). I risultati della tabella sono misure locali, non garanzie tratte dalla documentazione.

## Incremento del 15 ottobre: confine effettivo con Blender

[edit_document.py](../../image_to_3dlab/edit_document.py) conserva il GLB sorgente e scrive una copia di lavoro composta da `document.json` e buffer NumPy `.npz`, con hash e identità esplicite. Il caricamento verifica che sorgente e buffer corrispondano al manifest. L'assemblatore restituisce il file originale byte per byte se nulla è cambiato; per le modifiche aggiunge gli attributi sostitutivi, isolando le istanze che condividevano la stessa mesh. I dati non modificati, comprese immagini, skin e tracce delle animazioni, restano nel contenitore originale. Il chiamante deve fornire normali e tangenti coerenti con la forma modificata: non è ancora implementato un pass di rifinitura.

[blender_edit_bridge.py](../../scripts/blender_edit_bridge.py) legge questa copia di lavoro in un processo Blender separato, ricostruisce mesh, UV, normali personalizzate e gruppi dei pesi mediante `bpy`, poi rilegge i dati effettivi da Blender. Il backend può avviarlo con:

```text
blender --background --factory-startup --python-exit-code 1 --python scripts/blender_edit_bridge.py -- --input cartella-input --output nuova-cartella-output
```

Il test invoca già realmente questo comando. La web app non lo invoca ancora. Non c'è un collegamento diretto browser–Blender e non viene aperta la sua interfaccia: Blender gira sulla macchina che esegue il processo, riceve file e produce file più un rapporto JSON. L'integrazione futura deve aggiungere richieste HTTP, sessioni, operazioni, avanzamento e gestione degli errori; non basta esporre questo script come endpoint.

**Ambito attuale:** round-trip senza modifica. Non sono ancora presenti pennelli, tagli o distacchi, ricostruzione dell'armatura, correzione del rig e ricalcolo delle animazioni. Il rig e le clip sono conservati dall'assemblatore, non importati e riesportati da Blender. Il test riuscito non certifica il risultato di un'operazione di editing.

### Prove reali

Blender locale: **5.2.2 LTS**. Entrambe le varianti del guerriero della tabella iniziale hanno superato il passaggio reale attraverso Blender: posizioni, triangoli, UV e pesi riletti correttamente; GLB finale identico al sorgente. Le normali vengono misurate dopo la ricostruzione, con tolleranza `5e-4`, e conservate nei byte originali per il no-op. Sulla versione 18k l'errore massimo misurato è `0,00011204`.

Artefatti locali: `output/editor-feasibility/2026-10-15-bridge/{warrior-9k,warrior-18k-walk}/`, con snapshot `input`, risultato `roundtrip`, `bridge-report.json` nella cartella di risultato e `seam-report.json` nella cartella del caso.

[edit_topology.py](../../image_to_3dlab/edit_topology.py) studia i duplicati delle cuciture UV usando coppie uniche di bordi orientati in senso opposto, orientamento delle facce e compatibilità dei pesi. Non unisce vertici per sola vicinanza. Le sovrapposizioni ambigue rimangono separate e vengono segnalate. È una mappa di candidati sperimentale, non ancora integrata nel documento o certificata per i pennelli.

| Caso | Vertici render | Vertici dopo i raggruppamenti candidati | Vertici segnalati come ambigui |
| --- | --- | --- | --- |
| Guerriero 9k | 10.949 | 7.788 | 4.843 |
| Guerriero 18k walk | 20.336 | 13.320 | 6.499 |

L'ambiguità è ancora estesa: questi risultati non autorizzano ad attivare il pennello su tutto il modello. Anche una coppia geometricamente compatibile può appartenere a superfici distinte; il riconoscimento richiede ulteriori prove e protezioni. Nessun file del guerriero è stato modificato da questo studio.

### Verifica e lavoro restante

Suite mirata aggiornata con Blender reale: **250 test passati**. Comprende le regressioni precedenti, snapshot e assemblatore, casi di cuciture e prova del processo Blender. Comando:

```powershell
$env:I2L_TEST_EDIT_BLENDER = '1'
.\.venv\Scripts\python.exe -X utf8 -m pytest -q tests/test_edit_math.py tests/test_edit_glb.py tests/test_edit_probe.py tests/test_edit_document.py tests/test_edit_topology.py tests/test_edit_blender.py tests/test_scripts_registry.py tests/test_glb_turn.py tests/test_kimodo_retarget.py
```

Ruff sui moduli, script e test aggiunti: superato con target Python 3.10, per conservare la compatibilità 3.10/3.11 prevista dalla repository. Anche `git diff --check` è passato. Non è stata eseguita l'intera suite della repository.

Restano aperti i gate S0/RIG0: identità delle cuciture utilizzabile per l'editing, prove reali di fessura/distacco, rig esportato e ricaricato, parità con la preview e verifica UE. Non sono ancora implementate sessioni web, strumenti interattivi o revisioni accettabili dalla Library. Il piano completo rimane invariato; un flusso di modifica manuale in Blender con ritorno nella Library sarebbe un'alternativa più contenuta, non una capacità già realizzata da questo bridge.
