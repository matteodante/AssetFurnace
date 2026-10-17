# Windows nativo

Questa copia privata indipendente parte da AssetFurnace v0.4.3. Il progetto originale
rimane il remote `upstream`; il repository privato e' `origin`. Licenza, crediti e
storia del progetto originale sono conservati.

## Installazione

Si clona questa copia e si esegue il suo `install.ps1`. Lo script pubblicato sul sito del
progetto originale installa il codice upstream, non questa copia.

```powershell
git clone https://github.com/matteodante/AssetFurnace.git "$HOME\Desktop\Projects\AssetFurnace"
cd "$HOME\Desktop\Projects\AssetFurnace"
.\install.ps1
```

Rieseguito dalla cartella, `.\install.ps1` aggiorna quella stessa copia: prende l'ultima
release `vX.Y.Z` di questo repository (oggi nessuna, quindi `main`) e installa i pacchetti
Python da `requirements.lock`, come `install.sh`. Con un driver NVIDIA per CUDA 12.8 o 12.9
installa `requirements-cu128.lock`: l'indice cu128 di PyTorch arriva a torch 2.11.0, mentre il
torch di `requirements.lock` è solo su cu130. `$env:I3D_REF` sceglie un altro branch o tag.

Poi usare Setup nell'interfaccia, oppure gli script originali del progetto:

```powershell
.venv\Scripts\python.exe scripts\bootstrap_pixal3d.py --yes
.venv\Scripts\python.exe scripts\bootstrap_qwen_image.py --yes
.venv\Scripts\python.exe scripts\bootstrap_autorig.py --yes
.venv\Scripts\python.exe scripts\bootstrap_gltfpack.py --yes
.venv\Scripts\python.exe scripts\bootstrap_gltf_validator.py --yes
```

Il validatore glTF di Khronos (circa 2 MB, anche da Setup) serve al controllo `gltf.valid`:
senza, i report degli asset lo segnalano come non eseguito.

Questi comandi scaricano i modelli: Pixal3D circa 8,4 GB, Qwen circa 14,6 GB e
SkinTokens circa 1,6 GB, oltre a programmi e dipendenze. Blender 4.2 o successivo
deve essere installato su Windows.

## Avvio

Doppio clic su `Avvia-AssetFurnace.cmd`, oppure il comando originale:

```powershell
.venv\Scripts\python.exe viewer\serve.py
```

Studio: http://localhost:8777/viewer/studio.html

Setup: http://localhost:8777/viewer/setup.html

Il collegamento avvia il servizio in background se necessario. I suoi log sono in
`journal/server.out.log` e `journal/server.err.log`.

Per fermarlo: doppio clic su `Ferma-AssetFurnace.cmd`. Il server tiene i processi che
lancia (sd-cli, llama-server, i worker dei materiali) in un job object di Windows, quindi
si chiudono con lui anche se viene terminato da Gestione attività. Restano aperti solo il
Blender in cui si sta modificando un modello e i lavori in background con il loro file pid
(batch, generazioni), che il server riprende o chiude al riavvio.

## Differenze rispetto alla release originale

- Il setup SkinTokens usa `.venv/Scripts/python.exe` su Windows, invece del percorso
  Linux. Gli altri passaggi di installazione rimangono quelli originali.
- I file `Avvia-AssetFurnace` e `Ferma-AssetFurnace` avviano e fermano il server da
  Esplora file.

Modelli, ambienti Python, output e credenziali non vengono pubblicati nel repository.
TRELLIS.2 e Hunyuan3D-2.1 non sono configurati in questa installazione Windows.
