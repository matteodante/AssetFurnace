# Funzione 1: lotti e accesso per agenti (progetto)

Data: 15 ottobre 2026. Stato: **approvato e implementato** (15/10/2026). Uso: [`docs/batches.md`](../batches.md).

Decisioni dell'utente: ridipintura spenta nei lotti (solo come opzione esplicita `"finish": {"paint": true}`); righe fallite riprovate solo con `--retry-failed`; le uscite arrivano alla coda di revisione della funzione 7 tramite `outputs.json`.

In breve: un lotto è un elenco di righe; ogni riga diventa un **run manifest** come quelli di oggi e passa per gli
stessi codici di generazione e provenienza. Il lotto aggiunge solo tre cose: la lettura dell'elenco, uno stato per
riga che permette di riprendere, e una catena di passi con nomi stabili. Il server MCP è una porta sottile sopra queste
stesse funzioni.

## 1. La specifica di lotto

Due formati, stesso contenuto: JSON (completo) e CSV (solo le righe, comodo da un foglio di calcolo).

### JSON

```json
{
  "schema_version": 1,
  "name": "pilota-oggetti",
  "use_case": "game",
  "distribution": "private",
  "commercial_intent": true,
  "license_policy": {"allow_conditional": true},
  "output": {"directory": "../output/batches"},
  "defaults": {
    "backend": "pixal3d",
    "kind": "prop",
    "faces": 5000,
    "profile": "unreal",
    "steps": ["generate", "finish"],
    "parameters": {"seed": 42}
  },
  "rows": [
    {"id": "SM_Barrel_01", "images": ["refs/barrel.png"], "faces": 3000},
    {"id": "SK_Spider_01", "images": ["refs/spider.png", "refs/spider_side.png"],
     "kind": "creature", "faces": 20000, "steps": ["generate", "finish", "materials"]},
    {"id": "SM_Crate_02", "asset": "../output/old/crate.glb", "steps": ["finish"]}
  ]
}
```

- `rows` può anche essere un percorso: `"rows": "pilota-oggetti.csv"`.
- I percorsi si risolvono **rispetto al file della specifica**, come nei manifest di oggi.
- `use_case`, `distribution`, `commercial_intent`, `license_policy`: stessi campi e stessi controlli del run manifest
  (`validate_run_policy`). Valgono per tutto il lotto.

### CSV

```csv
id,images,kind,faces,profile,steps,backend,seed
SM_Barrel_01,refs/barrel.png,prop,3000,unreal,generate;finish,pixal3d,42
SK_Spider_01,refs/spider.png;refs/spider_side.png,creature,20000,unreal,generate;finish;materials,,
```

- Più valori in una cella si separano con `;`. Una cella vuota prende il valore predefinito.
- Un CSV da solo non ha i campi d'intento: valgono quelli di un'esecuzione singola senza manifest (`showcase`,
  `private`, non commerciale), cambiabili da riga di comando (`--use-case game --distribution private --commercial`).

### Campi di una riga

| Campo | Obbligatorio | Significato |
|---|---|---|
| `id` | sì | Id esterno, unico nel lotto. Lettere, cifre, `_ . -`: diventa il nome della cartella e dei file |
| `images` | sì, se c'è `generate` | La prima è l'ingresso del generatore; le altre sono riferimenti per i passi successivi (controlli, materiali). Non si scartano in silenzio: restano nel manifest della riga |
| `asset` | sì, se non c'è `generate` | Un GLB già fatto, per finire o esportare modelli esistenti |
| `kind` | no (`prop`) | `prop`, `creature`, `equipment` (oggetto, creatura, parte di equipaggiamento) |
| `faces` | no | Budget di facce, passato alla finitura (`--faces`, fra 1.000 e 200.000) |
| `profile` | no | Profilo d'uscita (funzione 5); oggi solo registrato |
| `steps` | no (`generate;finish`) | Passi da eseguire, nell'ordine dato |
| `backend` | no (`pixal3d`) | Il generatore |
| `parameters` / `seed` | no | Parametri del generatore, come nel manifest |

**Tutta la specifica si controlla prima della prima riga**: id doppi, immagini mancanti, passi sconosciuti (un nome
sbagliato è un errore, non un passo saltato), budget fuori misura, licenza. Un lotto sbagliato non parte, invece di
fermarsi alla riga 7.

## 2. Cosa scrive su disco

```
<output.directory>/<name>/
  spec.json               copia della specifica risolta, con il suo sha256
  batch-state.json        lo stato (sotto)
  rows/<id>/
    manifest.json         run manifest v1 della riga (generate)
    generate/conditional/<stem>__pixal3d__commercial-conditional__<run>.glb
    generate/conditional/<stem>__...provenance.json
    finish/<id>.glb, <id>.provenance.json, <id>.retopo-repaint.json
    logs/<passo>.log
```

## 3. Stato e ripresa

`batch-state.json` si riscrive per intero a ogni cambio, in modo atomico (file temporaneo e poi `os.replace`): un
processo ucciso a metà scrittura lascia la versione precedente, mai un file a metà.

Per ogni riga: `status` (`queued`, `running`, `done`, `failed`), `reason` se fallita, l'impronta della riga (sha256
dei suoi campi risolti e dei file d'ingresso), e per ogni passo `done` / `skipped` / `failed` con i file prodotti e il
loro sha256.

**Regola di ripresa**, quando si rilancia la stessa specifica:

- `done` con la stessa impronta: si salta.
- `running` (il processo è morto lì) o `queued`: si esegue, **dal primo passo non finito**. Un passo è finito solo se
  i suoi file esistono ancora e hanno lo sha256 registrato; così una generazione da 4 minuti non si ripete se è
  caduta la finitura.
- `failed`: resta fallita e il lotto va avanti. `--retry-failed` la riprova. Motivo: una riga che fallisce sempre (per
  esempio un'immagine rotta) non deve rubare GPU a ogni ripresa.
- Riga cambiata nella specifica (impronta diversa): torna in coda; i file vecchi restano su disco.
- Intento o licenza del lotto cambiati: la ripresa si **rifiuta**, come fa `finish_props.py --resume` quando cambiano
  le impostazioni. Si cambia nome al lotto o si toglie la cartella.

Il lucchetto è uno solo per cartella dei lotti (`<output.directory>/.batch.lock`: nome del lotto, pid, token), creato
in modo atomico: una GPU, un lotto, e di due invii simultanei ne parte uno solo. Chi avvia il runner (server MCP, studio)
lo prenota e passa il token al runner, che lo rileva. Un lucchetto di un processo morto si ignora. (Cambiato dopo la
review di Codex sulla PR #2: prima c'era un `batch.lock` per lotto, controllato e scritto in due passi.)

Codice d'uscita: 0 se tutte le righe sono `done`, 1 se qualcuna è `failed`, 2 se la specifica è sbagliata.

## 4. Registro dei passi

Nomi stabili: `generate`, `finish`, `export_unreal`, `checks`, `materials`.

Ogni passo è un modulo `image_to_3dlab/batch_steps/<nome>.py` con:

```python
def run(ctx: StepContext) -> StepResult: ...
def preflight(rows: list[BatchRow]) -> list[str]: ...   # facoltativo: cosa manca, prima di partire
```

- `StepContext`: la riga, la sua cartella, l'asset corrente (il GLB uscito dal passo precedente), il suo sidecar di
  provenienza, l'intento del lotto, un `log`.
- `StepResult`: il nuovo asset (o nessuno, per un passo che controlla soltanto), i file scritti, un piccolo record
  per lo stato, `ok` e `reason` (un controllo fallito ferma la riga con il suo motivo).

Il registro cerca il modulo per nome. **Se il modulo non esiste**, il passo si salta con un avviso (nello stato e su
stderr) e la riga va avanti. Se il modulo esiste ma fallisce l'import per un altro motivo, è un errore vero: un passo
rotto non deve sembrare un passo assente.

Io scrivo `generate.py` e `finish.py`. **Come si agganciano le funzioni 5, 7 e 8 dopo il merge**: chi integra aggiunge
un file adattatore, per esempio `batch_steps/export_unreal.py`, che chiama il codice della funzione 5. Nessuna modifica
a `batch.py`. Lascio il contratto anche nelle note condivise delle sessioni.

## 5. Manifest e provenienza

- **Ogni riga con `generate` diventa un run manifest v1** (`rows/<id>/manifest.json`) e si esegue con la **stessa
  funzione** di `pipeline.py --run-manifest`. Un'esecuzione singola di quella riga è quindi letteralmente
  `pipeline.py --run-manifest rows/<id>/manifest.json`. I campi del lotto (id, tipo, budget, profilo, riferimenti)
  stanno nel manifest in un blocco `batch`; il loader accetta già campi in più.
- **`pixal3d` entra nel `--run-manifest` di `cli.py`**. Oggi la CLI conosce solo SF3D, TRELLIS.2 e Hunyuan, mentre
  Pixal3D è l'unico che funziona su questo PC. Il backend chiama `scripts/pixal3d_generate.py` in un sottoprocesso,
  come fa lo studio, poi `finalize_output` scrive il `.provenance.json` di oggi. Il record di Pixal3D (`<stem>.json`) e
  la cartella della camera (`<stem>.svviews/`) si spostano accanto al GLB finale, perché Pixel Match li cerca lì.
- **Lo schema della provenienza non cambia** (un cambio di schema è una modifica incompatibile). Il legame con il lotto
  passa da `source_manifest`, che registra già percorso e sha256 del manifest.
- **`finish`** chiama `scripts/retopo_repaint.py` con `--faces` dalla riga. Il suo GLB riceve un `.provenance.json`
  derivato con la stessa forma di quello delle prop (`props_api.provenance_record`): licenza ereditata dal sidecar del
  modello d'origine, record d'origine incluso, parametri del passo.
- **La ridipintura di `finish` è spenta nei lotti**: usa i pesi Hunyuan 2.1 Paint (licenza Tencent, non valida nell'UE),
  esclusi per il gioco; fuori da Apple Silicon è già spenta. Il passo lo registra nel sidecar.

## 6. Nessun download di pesi

Prima della prima riga girano i `preflight` di tutti i passi chiesti: Pixal3D pronto (`readiness()` di
`pixal3d_generate.py`: eseguibile compilato e 9 file di pesi), Blender trovato per `finish`. Se manca qualcosa il lotto
**non parte** e dice cosa lanciare (per esempio `python scripts/bootstrap_pixal3d.py`, 8,4 GB). Il lotto non chiama mai
un bootstrap. Doppia sicurezza: `pixal3d_generate.py` rifiuta già di partire senza pesi.

Backend senza un controllo dei pesi che non scarichi nulla (SF3D e TRELLIS.2 scaricano da Hugging Face al primo uso)
per ora **non sono ammessi nei lotti**: errore chiaro in validazione. Si aggiungono quando hanno un loro controllo.

## 7. Server MCP

- **SDK**: il pacchetto ufficiale `mcp` (modelcontextprotocol/python-sdk, MIT), versione 2.3.0 del 2 ottobre 2026.
  Nella 2.x `FastMCP` si chiama `MCPServer` (`from mcp.server.mcpserver import MCPServer`): **verificato** con una prova
  in un ambiente a parte, dove `list_tools()` e `call_tool()` funzionano in processo, quindi i test non hanno bisogno di
  un client.
- **Trasporto**: stdio. Avvio: `python -m image_to_3dlab.mcp_server`, da mettere nel `.mcp.json` della repo di
  il gioco.
- **Strumenti**:

| Strumento | Cosa fa |
|---|---|
| `validate_batch(spec)` | Controlla la specifica e i preflight, senza eseguire nulla |
| `submit_batch(spec, retry_failed=False)` | Valida, poi lancia `pipeline.py --run-batch` in un processo staccato e risponde subito con cartella e pid. Un lotto dura ore: una chiamata non può restare aperta |
| `batch_status(batch)` | Riassunto (quante righe in coda, fatte, fallite) e righe con motivo |
| `list_outputs(batch, row_id=None)` | Per riga, i file finali e i loro sidecar, con percorsi assoluti: l'agente è sulla stessa macchina e li legge da lì ("ritirare i file") |
| `cancel_batch(batch)` | Ferma il processo (riusa `image_to_3dlab.processes`); la riga in corso torna in coda alla ripresa |
| `list_steps()` | I passi registrati e quali sono disponibili |

- Gli strumenti sono funzioni Python normali; `MCPServer` le espone e basta. Si provano senza MCP, più un test che passa
  da `call_tool`.
- **Una GPU, un lotto**: `submit_batch` rifiuta se un altro lotto è già in esecuzione (lucchetto globale in
  `<output>/batches/`).
- `mcp` è **facoltativo**: non entra in `requirements.txt`, il modulo dice come installarlo se manca. Entra in
  `requirements-dev.txt` per i test; il test MCP si salta se il pacchetto non c'è.

## 8. Cosa riuso (verificato) e cosa scarto

**Riuso**, letti nel codice il 15/10/2026:

- `provenance.py`: `load_run_manifest`, `validate_run_policy`, `finalize_output`.
- `cli.py`: il percorso `--run-manifest`, a cui aggiungo `pixal3d`.
- `scripts/pixal3d_generate.py`: generazione, `readiness()`, rifiuto senza pesi.
- `scripts/retopo_repaint.py`: la finitura dello studio, con il suo `preflight` e `--resume`.
- `image_to_3dlab/blender.py` (`find_blender`), `image_to_3dlab/processes.py` (gruppi di processi, per fermare).
- La forma del record derivato di `viewer/props_api.py` e la regola di `finish_props.py --resume` (rifiutare una
  ripresa con impostazioni cambiate). Ne copio la forma, non le importo: `image_to_3dlab` non deve dipendere da
  `viewer/`.

**Scarto**:

- **ComfyUI come orchestratore**: perde provenienza, controllo delle licenze e regola sui download.
- **L'API HTTP dello studio come motore del lotto**: richiede il server acceso, ragiona per lavoro singolo, non per
  lotto, e legherebbe il lotto all'interfaccia.
- **Code di lavoro (Celery, RQ) o un database**: una GPU, un processo alla volta. Un JSON regge alcune migliaia di righe
  (circa 1 MB per 4.000); se diventa lento si passa a SQLite senza cambiare il formato della specifica.
- **`mcp` 1.x con `FastMCP`**: la 2.x è quella attuale.

## 9. Rischi

| Rischio | Cosa faccio |
|---|---|
| Pixal3D su CUDA può non essere deterministico: stesso seme, file diversi | L'uguaglianza byte per byte si prova con il backend finto. Con quello vero si garantiscono stessi parametri, seme e manifest, registrati |
| La GPU è condivisa con lo studio e con altre sessioni | Il lucchetto copre solo i lotti. Lo scrivo nella documentazione; un coordinamento con lo studio è un passo successivo |
| Lo studio non mostra le uscite dei lotti (cerca `output/<run>/<run>.glb`) | Fuori da questa funzione; lo segno come seguito |
| Il contratto dei passi può non calzare alle funzioni 5, 7 e 8 | Contratto piccolo (contesto in, risultato out) e adattatori: si cambia l'adattatore, non il lotto |
| `mcp` porta molte dipendenze (pydantic 2.12, starlette, uvicorn, pywin32) | Facoltativo; prima di metterlo nei requisiti di sviluppo controllo `pip check` nel `.venv` |
| Le impostazioni di `finish` nel lotto divergono da quelle dello studio | Si usano i valori predefiniti del worker, cambiando solo `--faces` e la ridipintura; tutto finisce nel sidecar |
| Percorsi lunghi su Windows | Id corti obbligati dalla validazione; cartelle per riga, non nomi lunghi |

## 10. Come provo i criteri di accettazione (backend finto)

Nessuna generazione vera. Nei test un generatore finto (`fake`) registrato con un suo profilo di licenza (nel test, non
nel codice consegnato) scrive un piccolo GLB **deterministico** ricavato dallo sha256 dell'immagine e dal seme. I passi
`finish` e gli altri sono sostituiti da passi finti nel registro. Le immagini sono PNG minuscole create dal test.

| Criterio | Test |
|---|---|
| 10 righe senza interventi | Lotto JSON di 10 righe: tutte `done`, 10 GLB e 10 `.provenance.json`, codice d'uscita 0 |
| Interrotto a metà, riparte dalla riga giusta | Il finto lancia `KeyboardInterrupt` alla riga 6: le righe 1-5 sono `done`, la 6 `running`. Alla ripresa il contatore delle chiamate mostra che 1-5 non ripartono e che 6-10 girano. Un secondo test parte da uno stato scritto a mano come dopo un `kill` (riga in `running`, generazione già fatta): riparte da `finish` |
| Uscite uguali a 10 esecuzioni singole | Le stesse 10 righe lanciate una per una con `cli.main(["--run-manifest", ...])`: stesso sha256 dei GLB e stessa provenienza, tolti `run_id`, date e percorsi |

Altri test: JSON e CSV danno le stesse righe; percorsi relativi alla specifica; errori di validazione prima di qualsiasi
riga; passo assente saltato con avviso; import rotto di un passo presente = errore; riga fallita con motivo e lotto che
continua; `--retry-failed`; ripresa rifiutata se cambia l'intento; preflight che ferma il lotto senza chiamare alcun
bootstrap; il backend `pixal3d` della CLI con il sottoprocesso sostituito; gli strumenti MCP, in diretta e via
`call_tool`.

Prova a mano facoltativa, solo se il coordinatore la permette: righe con `asset` e solo `finish` sui GLB già in
`output/`. Usa Blender, non i generatori.

## File che tocco

`image_to_3dlab/batch.py`, `image_to_3dlab/batch_steps/` (pacchetto mio), `image_to_3dlab/mcp_server.py`, le aggiunte a
`cli.py` (`--run-batch` e backend `pixal3d`), i test relativi, `requirements-dev.txt` (solo `mcp`), le mie righe in
`CHANGELOG.md`, un esempio in `manifests/` (`example-batch.json` e `.csv`, da aggiungere all'eccezione del
`.gitignore`) e la documentazione d'uso in `docs/batches.md`. Nessuno script nuovo in `scripts/`, quindi niente da
aggiungere a `scripts/README.md`.

## Seguiti

Punti della review di Codex sulla PR #2 rimandati con la regola dell'utente (P2 che non sono errori visibili:
risultato sbagliato, dati persi, crash, licenza sbagliata).

- Il piano dei passi di una ripresa considera finito un passo i cui file esistono, senza ricontrollarne l'hash: se un
  file generato cambia sul posto, il generatore riparte senza il controllo preliminare dei suoi pesi (che comunque
  rifiuta di scaricarli). [Thread](https://github.com/matteodante/AssetFurnace/pull/2#discussion_r4288609326)

Respinti con motivo nel thread (non da fare): `output.directory` di un manifest relativa al manifest invece che alla
cartella corrente, come fanno tutti i backend di `--run-manifest`
([thread](https://github.com/matteodante/AssetFurnace/pull/2#discussion_r4285752400)); chiave di revisione con il
nome del lotto, contro il contratto della funzione 7
([thread](https://github.com/matteodante/AssetFurnace/pull/2#discussion_r4287559111)); copia dei GLB sorgente delle
righe di sola finitura ([thread](https://github.com/matteodante/AssetFurnace/pull/2#discussion_r4287559143), poi
risolto in modo diverso: una riga già fatta sopravvive allo spostamento della sua sorgente).
