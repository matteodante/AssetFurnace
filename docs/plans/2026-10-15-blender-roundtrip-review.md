# Revisione prima del merge

15 ottobre 2026. Revisione della PR #1, inclusi i prototipi precedenti e i percorsi
condivisi di Library, viewer, gestione processi, aggiornamenti e metadati di output.
Questa è la revisione tecnica del codice; la revisione indipendente precedente
dell'interfaccia resta distinta.

## Problemi corretti

| Problema | Correzione e verifica |
| --- | --- |
| GLB con scena attiva vuota o joint fuori dallo scheletro accettato | Controllo della scena effettivamente mostrata, nodi ripetuti, indici e pesi. Rifiuto esplicito di più di quattro influenze, che il viewer non riproduce. Test negativi sui file reali GLB |
| Un solo sidecar di provenienza finiva nella versione | Conservati tutti i sidecar e relativi hash. Il record principale è referenziato per evitare crescita esponenziale tra versioni. Test con vincoli di licenza aggiuntivi dell'animazione |
| Metadati incompleti potevano interrompere l'elenco delle sessioni o delle versioni | Record controllati e quelli corrotti saltati; gli altri restano accessibili. Test con sessione e versione danneggiate |
| Retry idempotente poteva dichiarare riuscita una versione modificata sul disco | Controllo del contenuto pubblicato contro l'hash prima di rispondere con successo |
| Alias di un nome pubblico poteva puntare a uno snapshot privato | Controllo del percorso risolto anche nei file pubblici. Test HTTP con symlink |
| Viewer precedente tratteneva risorse dopo il cambio modello | Rilascio deduplicato di geometrie, materiali, texture, immagini e skeleton, incluso il materiale nascosto dalle viste di ispezione. Rilascio del target di illuminazione |
| Risposte tardive di caricamenti chiusi potevano scrivere nella vista corrente | Stato di chiusura idempotente; callback ignorati e risorse tardive rilasciate. Prova nel browser |
| Messaggi dinamici nella preview potevano essere interpretati come HTML | Escape di errori, didascalie e nomi delle clip; test della funzione effettivamente usata |
| PID nulli o negativi accettati nel codice precedente di cleanup | Rifiuto nei record e nei controlli prima di inviare segnali o avviare taskkill. Test assicurano che nessuna chiamata raggiunga il sistema |
| Percorsi con spazi/caratteri shell nelle istruzioni di aggiornamento | Quoting corretto per shell POSIX e PowerShell, verificato con percorsi avversi |
| Percorsi dei prop e librerie ROCm non portabili | Separatore POSIX nei metadati JSON, separatore della piattaforma nella variabile di ambiente |
| Test precedenti assumevano macOS/Linux sul runner Windows | Mock della piattaforma dichiarata, percorsi nativi nei confronti, fixture di filesystem reali; test POSIX eseguiti solo dove shell/segnali/permessi sono disponibili |

## Verifica finale

- Windows, Python 3.11.14, Blender 5.2.2 LTS; nessun download di pesi.
- `$env:I2L_TEST_EDIT_BLENDER='1'`, poi `python -X utf8 -m pytest tests -q --tb=short`:
  **2.566 passati, 42 saltati, zero fallimenti**. I test saltati includono percorsi
  specifici POSIX e integrazioni che richiedono condizioni esterne; non sono dichiarati
  verificati su Windows. Restano cinque warning di librerie/test precedenti.
- Ruff completo sui moduli dell'editor e sui moduli processi/aggiornamento revisionati;
  controlli Python `E9,F63,F7,F82` su package, viewer, scripts e tests: passati.
- `git diff --check`: passato.
- Browser reale: caricamento del guerriero, rilascio della geometria una sola volta,
  doppia chiusura sicura, nessuna callback tardiva dopo la chiusura. Il round-trip
  Blender e il cambio versione/clip sono coperti dalle prove precedenti e dai test.

Non risultano problemi aperti trovati in questa revisione. I limiti di formato e di
consegna Unreal documentati in [blender-editing.md](../blender-editing.md) restano parte
del contratto della funzione; il percorso non certifica automaticamente un asset UE.
