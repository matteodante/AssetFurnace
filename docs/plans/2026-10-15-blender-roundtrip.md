# Modifica manuale in Blender e nuove versioni

Data: 15 ottobre 2026. Richiesta corrente: una funzione semplice per aprire un modello
in Blender, modificarlo, esportarlo e importarlo come nuova versione nella web app.

## Confine del rilascio

Blender fornisce gli strumenti per forma, topologia, scheletro e pesi. Studio gestisce
apertura, copia sorgente, importazione esplicita, versioni, provenienza e preview.
L'editor guidato dentro il browser e la correzione automatica delle animazioni sono
rimandati. I prototipi precedenti restano isolati dal percorso manuale.

## Architettura implementata

| Parte | Responsabilità |
| --- | --- |
| `viewer/studio/blender-edit.js` | Versioni, sessione catturata prima della modifica, apertura/download, upload, errori e conferma accessibile |
| `viewer/blender_edit_api.py` | Endpoint HTTP, controlli origine/host e dimensione, avvio desktop locale senza shell, stato del processo |
| `viewer/blender_versions.py` | Appartenenza al modello, snapshot e sidecar, hash, parentela, scrittura atomica e retry idempotenti |
| `image_to_3dlab/blender_exchange.py` | Controlli strutturali GLB e inventario del file importato, senza riscriverlo |
| `scripts/blender_open_asset.py` | Import del GLB in un workspace separato o riapertura del `.blend` salvato |
| Library e viewer esistenti | Versioni associate all'asset, preview del GLB scelto e riproduzione delle sue clip incorporate |

Non viene passato un oggetto JavaScript a Blender. Il backend salva un GLB immutabile
e avvia Blender con percorsi controllati come argomenti. Il ritorno è un upload esplicito
del GLB esportato, non un watcher né un add-on. Il server non riscrive il file importato.

## Casi da rispettare

- Sorgente identificata da asset, file, hash e versione; un cambio di selezione durante
  l'upload non cambia la destinazione né forza l'utente a tornare al modello precedente.
- Originali e licenze conservati. Sidecar malformati bloccano la cattura invece di essere
  ignorati. Le licenze di eventuale contenuto aggiunto manualmente restano da verificare.
- Nessun riuso implicito di vecchie clip, LOD, collisioni o pacchetti UE sul nuovo rig.
- Input e metadati limitati, percorsi relativi senza traversal/link, niente fetch di
  risorse esterne. Snapshot, workspace e log non sono serviti come file statici.
- Richieste di modifica dalla stessa origine; avvio desktop solo locale. Server remoto
  o Blender assente: download della copia catturata e successivo upload.
- Pubblicazione atomica, retry sicuro, sessioni recuperabili dopo riavvio, messaggi di
  errore sull'avvio/import. Un export diverso richiede una nuova sessione.
- Il GLB resta autorevole. Avvisi su perdita o cambiamento di rig/clip/texture non sono
  una certificazione di compatibilità e non promettono una ricostruzione automatica.

## Verifica

Test dedicati: `test_blender_versions.py`, `test_blender_edit_api.py`,
`test_blender_open_asset.py`, più regressioni Library, Studio, server e registro script.
Il test Blender è opt-in con `I2L_TEST_EDIT_BLENDER=1`: apre un modello, modifica un
vertice, esporta, importa la revisione e controlla che l'originale sia identico.

Esecuzione estesa su Windows/Python 3.11: **498 passati, 3 fallimenti già riprodotti
sulla base `main`** in `test_viewer_generate_api.py`. Due assumono `os.killpg`, assente
su Windows; il terzo simula un host NVIDIA senza sostituire il controllo Windows.
Nessun nuovo fallimento nel percorso Blender. Ruff passa con target Python 3.10.

Prova manuale su una copia isolata del guerriero: apertura con finestra Blender 5.2.2,
export modificato, nuova versione nella Library, 34 joint e due clip incorporate
(la seconda è una copia creata per il test), selezione clip/scrubber e ritorno
all'originale. Verificati file non valido, retry, download senza Blender, conferma con
focus sul selettore Version e cambio di selezione durante l'importazione.

Revisione UI indipendente su desktop, viewport 1226px e mobile: rispettata l'interfaccia
esistente; corretta la conferma e il focus dopo l'import. Il piano precedente rimane
uno studio futuro, non un prerequisito di questo flusso. [Uso, limiti e recupero](../blender-editing.md).
