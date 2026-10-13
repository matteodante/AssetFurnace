# Windows nativo

Questa copia privata indipendente parte da AssetFurnace v0.4.3. Il progetto originale
rimane il remote `upstream`; il repository privato e' `origin`. Licenza, crediti e
storia del progetto originale sono conservati.

## Installazione

Si usa lo script ufficiale del sito. Per installare questa copia privata e conservare
la correzione del setup Windows del rig, impostare repository e branch prima di eseguirlo:

```powershell
$env:I3D_DIR = "$HOME\Desktop\Projects\AssetFurnace"
$env:I3D_REPO = 'https://github.com/matteodante/AssetFurnace.git'
$env:I3D_REF = 'main'
irm https://assetfurnace.com/install.ps1 | iex
```

Poi usare Setup nell'interfaccia, oppure gli script originali del progetto:

```powershell
.venv\Scripts\python.exe scripts\bootstrap_pixal3d.py --yes
.venv\Scripts\python.exe scripts\bootstrap_qwen_image.py --yes
.venv\Scripts\python.exe scripts\bootstrap_autorig.py --yes
.venv\Scripts\python.exe scripts\bootstrap_gltfpack.py --yes
```

Questi comandi scaricano i modelli: Pixal3D circa 8,4 GB, Qwen circa 13,4 GB e
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

## Differenze rispetto alla release originale

- Il setup SkinTokens usa `.venv/Scripts/python.exe` su Windows, invece del percorso
  Linux. Gli altri passaggi di installazione rimangono quelli originali.
- I due file `Avvia-AssetFurnace` facilitano l'avvio da Esplora file.

Modelli, ambienti Python, output e credenziali non vengono pubblicati nel repository.
TRELLIS.2 e Hunyuan3D-2.1 non sono configurati in questa installazione Windows.
