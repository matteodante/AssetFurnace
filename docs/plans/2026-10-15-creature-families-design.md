# Funzione 6: famiglie di creature (progetto)

Data: 15 ottobre 2026. Stato: **approvato; PR 1 implementata** (riconoscimento, controllo, bipede digitigrado e
quadrupede, studio). Le differenze rispetto al progetto sono nella sezione 11.
Piano di riferimento: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 6.

In una riga: dopo il rig automatico, AssetFurnace **riconosce la famiglia** della creatura dallo scheletro (umanoide,
bipede digitigrado, quadrupede, molte zampe, alata, serpente, pianta), dà un **nome a ogni catena di ossa** (zampe,
ali, coda, collo) e ci mette sopra **due clip di prova** fatte con curve procedurali (movimento e attacco), controllate
da misure automatiche di deformazione. Nello studio si sceglie o si conferma la famiglia e si guarda la clip.

## 1. Cosa c'è già (letto nel codice il 15/10/2026)

- **SkinTokens** gira su Windows, nel suo ambiente (`vendor/SkinTokens/.venv`, torch cu130, `bpy` 5.0.1, flash
  attention 2), lanciato da `viewer/autorig_api.py` (`rig_commands`, `run_job`): il lupo mannaro è stato rigato in
  51 s (`output/compare-20261015/werewolf/rig.log`). Il checkpoint usato,
  `articulation_xl_quantization_256_token_4/grpo_1400.ckpt`, è addestrato su **Articulation-XL2.0**, cioè scheletri di
  ogni tipo (animali, insetti, oggetti), non solo umanoidi. Le ossa escono senza nome (`bone_0…bone_N`).
- **`scripts/rig_check.py`** legge lo scheletro dal GLB senza Blender e dice solo "umanoide sì/no" (con il mappatore
  di `kimodo_retarget.py`). Oggi tutto ciò che non è umanoide riceve "rigged, not animated".
- **Quadrupedi**: `scripts/blender_quadruped_pipeline.py` (metarig Rigify adattato a mano, pesi via proxy voxel, trotto
  e passo procedurali); curve pure in `scripts/quadruped_gait.py` (fasi e appoggio per zampa, sollevamento, coda,
  testa), `rigify_walk_pose.py` e `attack_pose.py`. Funziona, ma parte da un metarig sistemato da una persona e da nomi
  Rigify; un quadrupede rigato da SkinTokens oggi non riceve nessuna clip.
- **Generici e riusabili**: `clean_skin_weights.py` (toglie i pesi delle ossa lontane, vale per ogni famiglia), le
  funzioni di catena e di posa di `kimodo_retarget.py` (`_chain`, `_limb`, `pose_matrices`, conversioni di
  quaternioni), il legame con un proxy voxel di `blender_bind_rig.py`.
- **Umanoidi**: 21 clip Kimodo in `image_to_3dlab/motions/`, retarget con `kimodo_retarget.py`.
- **Profilo Unreal** (funzione 5, ramo `claude/feat-unreal-profile-32aa4b`, non ancora su main): rinomina le ossa come
  Manny quando lo scheletro è umanoide; per le altre creature tiene i nomi originali e lascia il lavoro a questa
  funzione.
- **Controlli** (funzione 7, su main): il registro prevede già controlli con `needs=("rig",)`; le compenetrazioni in
  posa sono rimandate a "un secondo passo".

### Il lupo mannaro: perché le gambe si incrociano

Ho letto lo scheletro di `werewolf_rigged.glb` (46 ossa). Ogni gamba ha **tre ossa**: anca → ginocchio (in avanti) →
garretto (indietro, alto da terra il 18% della gamba, circa 19 cm su 220) → punta. Nell'uomo la caviglia sta al 5%
circa. È una gamba digitigrada: il "piede" vero è il metatarso, lungo e inclinato, e il garretto sta in alto.

La causa che ipotizzo (da confermare nella PR 4, in Unreal):
il profilo della funzione 5 la rinomina come Manny (`thigh`, `calf`, `foot`, `ball`): così il `foot` di Manny, che
nell'uomo è una caviglia a pochi centimetri da terra e un piede piatto, finisce sul garretto. Il retarget copia le
**rotazioni** della coscia e del polpaccio umani su ossa con orientamenti di riposo diversi, e senza un obiettivo IK
sul punto che tocca terra le ginocchia ruotano verso l'interno e i piedi passano uno davanti all'altro. Le due gambe
del lupo, poi, non sono simmetriche (ginocchia a circa 26 e 23 cm dall'asse) e il verso dei piedi è girato di 17°
rispetto al corpo.

## 2. Prima riusa: i rig automatici e le loro licenze

Verificato su GitHub e Hugging Face il 15/10/2026, **file per file**, compresi i pesi che il codice scarica da solo.

| Strumento | Codice | Pesi propri | Pesi che scarica in più | Dimensione | Dove gira | Esito |
|---|---|---|---|---|---|---|
| **SkinTokens** (VAST-AI) | MIT | MIT (`VAST-AI/SkinTokens`) | Qwen3-0.6B, Apache-2.0 | 1,13 + 0,49 GB, **già installati** | Windows | **Si usa per primo** |
| **UniRig** (VAST-AI) | MIT | MIT (`VAST-AI/UniRig`) | solo la configurazione di `facebook/opt-350m` (nessun peso) | scheletro 1,44 GB + pelle 4,38 GB = **5,8 GB** | WSL2 (`spconv`, `torch_scatter`, flash-attn: niente ruote per Windows) | Riserva, solo se SkinTokens sbaglia una famiglia |
| **Puppeteer** (Seed3D) — scheletro | Apache-2.0 (codice di Michelangelo incluso: **GPL-3.0**) | Apache-2.0 (`Seed3D/Puppeteer`) | `Maikou/Michelangelo` shapevae-256, **LGPL-3.0** | 4,40 + 3,93 GB = **8,3 GB** | WSL2, e va aggiornato: chiede CUDA 11.8 e torch 2.1, che **non supportano la RTX 5070 Ti** (Blackwell vuole CUDA 12.8) | Seconda riserva, solo lo scheletro |
| **Puppeteer** — pelle | Apache-2.0 | Apache-2.0 | `mikaelaangel/partfield-ckpt` (PartField), **NVIDIA License: solo uso non commerciale** | — | — | **Escluso** |
| **Puppeteer** — animazione da video | Apache-2.0 | — | CoTracker3 e Video-Depth-Anything-Large, **CC-BY-NC-4.0** | — | — | **Escluso** |

Note:
- È la stessa trappola di OmniPart: Puppeteer si presenta come Apache-2.0, ma la pelle carica PartField (NVIDIA, non
  commerciale) e l'animazione due modelli CC-BY-NC. Lo scheletro da solo è pulito (Apache + LGPL; il codice GPL resta
  uno strumento interno e non passa negli asset).
- Il dataset di addestramento di tutti e tre è Articulation-XL2.0 (CC-BY-4.0): riguarda chi lo ridistribuisce, non chi
  usa i pesi.
- **Download previsti per la strada principale: nessuno.** SkinTokens, Qwen-Image, Pixal3D e Blender 5.2 sono già
  installati; le curve sono codice nostro. Tutto gira su **Windows**; WSL2 serve solo alle riserve.
- **Nessun download senza conferma**: UniRig e Puppeteer si scaricano solo se una famiglia fallisce con SkinTokens, e
  solo dopo averlo chiesto con nome, percorso e dimensione (regola della repo).

## 3. Cosa provo per ogni famiglia

Una creatura di prova per famiglia, fatta con la catena di AssetFurnace (Qwen-Image → Pixal3D → finitura), nello stile
del progetto (colori saturi, forme stilizzate, sagome leggibili). Poi SkinTokens. Poi le clip.

| Famiglia | Creatura di prova | Cosa deve riconoscere lo scheletro | Clip di prova | Se SkinTokens fallisce |
|---|---|---|---|---|
| **Bipede digitigrado** | Il **lupo mannaro esistente** | 2 gambe a tre segmenti con il garretto alto | Camminata digitigrada + artigliata | Correggo la mappa delle ossa, non serve un altro rig |
| **Molte zampe** | Ragno gigante (8 zampe, addome grosso) | 6-8 catene che scendono a terra da un corpo centrale | Camminata a onde alternate (tripode/tetrapode) + morso con le zampe anteriori alzate | UniRig → Rigify (catene `limbs.simple_tentacle`) |
| **Alata** | Draghetto (quattro zampe, due ali, coda) | 4 zampe + 2 catene laterali larghe (ali) + coda | Volo sul posto (battito d'ali, coda che segue) + soffio/morso a terra | UniRig → metarig Rigify "Bird" adattato |
| **Serpente** | Basilisco senza zampe (corpo lungo, cresta) | Una catena lunga di molte ossa, con la testa a un capo | Strisciata a onda (serpentina) + scatto in avanti | UniRig → catena Rigify `spines.basic_tail` lungo la curva del corpo |
| **Pianta** | Fiore carnivoro (stelo, testa a bocca, radici) | Una catena che sale dal suolo, foglie/radici come catene corte | Oscillazione (idle) + scatto della testa con la bocca | Catene semplici: lo stelo è un tentacolo |

Le altre famiglie del gioco ricadono in queste: demoni e chimere sono bipedi o quadrupedi con ali o code (catene in
più sulle famiglie esistenti); insetti volanti sono "molte zampe" con ali.

### Riconoscere la famiglia (Python puro, senza Blender)

`image_to_3dlab/creature_family.py` legge lo scheletro come `rig_check.py` e lo divide in catene dalla radice:

- **gambe**: catene che finiscono vicino al suolo, sotto il corpo;
- **ali**: catene laterali, larghe, che finiscono sopra o all'altezza del corpo e sono più lunghe del busto;
- **coda / collo-testa**: catene lungo l'asse del corpo, dietro e davanti;
- **digitigrado**: gamba con l'ultima articolazione alta da terra (oltre ~12% della lunghezza della gamba; nell'uomo è
  ~5%) e il segmento di mezzo che va indietro;
- **serpente**: una catena che contiene quasi tutto lo scheletro, lunga più di ~5 volte la sezione;
- **pianta**: radice a terra, catena principale verso l'alto, nessuna gamba.

Esce un dizionario `{famiglia, fiducia, ruoli: {osso: ruolo}}`, salvato accanto al GLB come `<nome>.family.json`. La
scelta della persona nello studio prevale sul riconoscimento e va nello stesso file. Il messaggio di `rig_check.py`
cambia da "humanoid only" a "riconosciuto: ragno, 8 zampe" o simile.

### Clip di prova (curve procedurali)

Ho scelto le curve procedurali perché funzionano su **qualsiasi** scheletro uscito da SkinTokens, una volta dati i
ruoli. I pacchetti gratuiti hanno scheletri loro: andrebbero ritargettati, e per ragni, serpenti e piante non c'è un
retarget standard.

- **Curve pure** (testabili, senza `bpy`), in `scripts/creature_motion.py`:
  - **molte zampe**: generalizzo le fasi del passo dei quadrupedi a N zampe (onda metacronale: ogni zampa parte con
    un ritardo; con 6 e 8 zampe esce il tripode o il tetrapode). Riuso il calcolo di appoggio, sollevamento e
    velocità di `quadruped_gait.py` (`limb_phase`, `is_planted`, `foot_offset`), dove l'ipotesi delle quattro zampe
    sta solo nelle tabelle `LIMBS` e `FOOT_IK`;
  - **ali**: battito asimmetrico (giù veloce, su lento), con la punta in ritardo sulla base;
  - **serpente**: onda che scorre lungo la catena (ampiezza che cresce verso la coda, testa stabilizzata);
  - **pianta**: oscillazione lenta che si smorza verso l'alto, scatto con anticipo e rimbalzo;
  - **bipede digitigrado**: passo a due gambe con IK sulla punta e il metatarso che resta inclinato;
  - **attacchi**: riuso la forma "carica, colpo, ritorno" di `attack_pose.py` applicata al ruolo giusto (zampe
    anteriori, testa, mascelle se ci sono).
- **Lato Blender**, `scripts/creature_clips.py` (*implementato*: nel Python di SkinTokens con il suo `bpy`, come
  `kimodo_retarget.py`, non con il Blender 5.2 di sistema): applica le curve alle ossa per ruolo, con IK sulle zampe
  che toccano terra, e scrive le clip in posto (nessuno spostamento della radice), su CPU.
- **Quadrupedi rigati da SkinTokens**: con i ruoli, le stesse curve di `quadruped_gait.py` danno anche a loro un trotto
  di prova, quasi gratis. Non sostituisce il flusso Rigify di `docs/quadruped-pipeline.md`, che resta per la qualità
  di produzione.

Le clip seguono la convenzione che lo studio già legge: un GLB per clip accanto al rigato,
`<base>_<famiglia>-<movimento>.glb`, per esempio `wolf_digitigrade-walk.glb` e `wolf_digitigrade-attack.glb`
(`viewer/assets_api.py`, `_rigs`). Il trattino evita ogni scontro con le mosse preimpostate, che non ne hanno.

### Lupo mannaro: due strade, entrambe

1. **Clip propria**, come per le altre famiglie: la camminata digitigrada sul suo scheletro, senza Manny.
2. **Retarget da Manny corretto**, che è quello che serve in gioco (riusare le animazioni umane): il profilo della
   funzione 5 riceve dalla famiglia la mappa delle ossa giusta per una gamba digitigrada, cioè:
   - catena gamba `thigh → calf → foot` dove `foot` è il **garretto** e `ball` la **punta**;
   - **obiettivo IK sulla punta** (l'osso che tocca terra), non sulla caviglia, con il polo del ginocchio in avanti;
   - nella scheda `.unreal.json` le impostazioni del retargeter per quelle catene (IK attivo sulle gambe, traslazione
     del bacino scalata sull'altezza dell'anca).
   Lo script d'import della funzione 5 le applica. Se l'IK Retargeter di UE 5.8 non basta, il ripiego è uno scheletro
   umanoide "a gamba lunga" (si aggiunge un osso fittizio per il metatarso), ma lo provo solo se il primo fallisce.

Le mosse Kimodo dello studio sul lupo hanno con buona probabilità lo stesso difetto (anche lì si copiano rotazioni
umane). Le sistemo solo se la PR 4 conferma la causa: è un seguito, non parte di questa funzione.

## 4. Controlli: "senza deformazioni visibili" misurato

Un controllo nuovo, `rig.deformation`, nel registro della funzione 7 (`needs=("rig",)`, una riga di aggancio).
Per ogni clip, su 24 fotogrammi campione, con Blender senza interfaccia:

- **allungamento dei lati**: rapporto fra lunghezza in posa e a riposo; fail se l'1% dei lati supera ×2 o scende
  sotto ×0,4;
- **triangoli rovesciati** rispetto al riposo (normale invertita): fail oltre lo 0,5%;
- **pelle che si attraversa**: `BVHTree.overlap` fra parti lontane nello scheletro (zampa con zampa, ala con corpo):
  fail oltre una soglia da tarare;
- **piedi**: per le gambe, nessun piede sotto il suolo oltre 1 cm e, per il lupo, **i piedi non passano mai dall'altra
  parte dell'asse** (la distanza laterale fra i due piedi non cambia segno). Questo è il controllo che avrebbe
  fermato il difetto visto in Unreal.

Le soglie sono **provvisorie** e si tarano su queste cinque creature. Restano i controlli a occhio: la girata a 360°
e la coda di revisione della funzione 7.

## 5. Nello studio

Non c'è una pagina di rig a sé: rig e animazione sono passi del pannello Steps. Il codice nuovo sta in un modulo mio,
`viewer/studio/creature-family.js` (selettore e testi), più una funzione nel server, `viewer/creature_family_api.py`.
Nei file condivisi solo righe di aggancio:

- `viewer/studio/steps.js`, passo `rigged`: la domanda "Is this a humanoid?" con i due bottoni diventa un selettore
  **Famiglia**: *Riconosci da sola* (predefinita), Umanoide, Bipede digitigrado, Quadrupede, Molte zampe, Alata,
  Serpente, Pianta. In `startRig` una riga aggiunge `family` al modulo (oggi la risposta "umanoide" non arriva
  nemmeno al server).
- `viewer/generate_api.py` (`_create_animate_rig`): legge `family` e la passa al lavoro di rig.
- `viewer/autorig_api.py` (`rig_commands`): dopo SkinTokens e la pulizia dei pesi, un comando in più che riconosce la
  famiglia (o usa quella scelta), scrive `<base>.family.json` e le due clip di prova. Le clip compaiono da sole come
  chip nell'anteprima, perché lo studio le trova già per nome.
- `viewer/studio/steps.js`, passo `animated`: per le famiglie non umanoidi, invece di "preset moves are for humanoids
  only", il testo dice la famiglia riconosciuta e quali clip di prova ci sono; le mosse Kimodo restano per gli
  umanoidi.
- **Anteprima**: nessun codice nuovo. Il visore three.js (`viewer/studio/viewer.js`, `animation/player.js`) suona già
  le clip dei GLB; si sceglie la clip dalle chip. Il controllo `rig.deformation` appare nella revisione della funzione
  7 come gli altri.

## 6. Consegna: PR piccole

| PR | Contenuto | Creature |
|---|---|---|
| 1 | Riconoscimento della famiglia, `rig.deformation`, curve e clip del bipede digitigrado e del quadrupede, selettore e anteprima nello studio | Lupo mannaro (esiste già) |
| 2 | Molte zampe + serpente | Ragno, basilisco |
| 3 | Alata + pianta | Draghetto, fiore carnivoro |
| 4 | Mappa delle ossa digitigrade per il profilo Unreal e prova del retarget di Manny in UE 5.8 | Lupo mannaro |

Ogni PR: test prima del codice, riga in `CHANGELOG.md`, script nuovi in `scripts/README.md`, review di Codex con la
regola di stop (P1 sempre, P2 solo se errori veri visibili, massimo 3 giri, gli altri in "Seguiti"). Non unisco io.
Solo la PR 4 dipende dalla funzione 5 (PR #6): parte dopo il suo merge. Le altre non toccano il profilo.

## 7. GPU e tempi

Tutto il lavoro su GPU (generazione delle 4 creature, SkinTokens, eventuale UniRig) prende prima il lucchetto
`C:\Users\Matteo\AppData\Local\Temp\assetfurnace-gpu.lock` (pid e nome della sessione) e lo libera alla fine.
Stima per creatura: immagine ~1 min, Pixal3D ~4-5 min, finitura ~4 min, SkinTokens ~1 min. Le clip e i controlli
girano in Blender su CPU.

## 8. Alternative scartate

- **Puppeteer completo**: pelle e animazione caricano pesi non commerciali (sopra).
- **Pacchetti di animazioni gratuiti** (per esempio Quaternius, CC0): licenza chiara, ma scheletri loro; per ragni,
  serpenti e piante il retarget non esiste. Li tengo come riferimento visivo, non come sorgente.
- **Mixamo**: solo umanoidi, licenza Adobe non ridistribuibile, niente gambe digitigrade.
- **Modelli Rigify scritti a mano per ogni famiglia, da subito**: è il piano B; partire da lì vorrebbe dire adattare a
  mano un metarig per ogni mostro, cioè 592 volte.
- **Animazione da testo (Kimodo) per le creature**: addestrato su movimento umano.

## 9. Rischi

- **SkinTokens su forme estreme**: otto zampe sottili e vicine, o ali sottili come carta, possono uscire fuse o
  con pesi che sanguinano da una zampa all'altra. È proprio quello che `rig.deformation` deve vedere. Rimedio in
  ordine: rigenerare con un altro seme, UniRig, Rigify.
- **Riconoscimento sbagliato**: le euristiche sbagliano su forme ibride (chimere). Per questo nello studio la scelta
  è della persona e il riconoscimento è solo una proposta.
- **Clip procedurali "meccaniche"**: bastano come prova di rig, non come animazioni finali del gioco. Le animazioni di
  produzione per i mostri sono un lavoro a parte (e lo dico qui per non prometterle).
- **IK Retargeter di UE 5.8 sul lupo**: provato solo che riconosce lo scheletro; l'IK sulla punta va verificato.
  Ripiego scritto sopra.
- **Pixal3D su zampe sottili**: il ragno può uscire con zampe fuse al corpo. Si scrive un prompt con zampe ben
  staccate e una vista che le mostra; se non basta, si prova un secondo seme prima di cambiare strada.

## 10. Come provo l'accettazione

Accettazione dal piano: una creatura per famiglia rigata e animata con una clip di prova senza deformazioni visibili.

1. Cinque creature (lupo mannaro + 4 generate), ognuna con la sua provenienza.
2. SkinTokens su ciascuna; `creature_family.py` deve riconoscere la famiglia giusta su tutte e cinque (e i ruoli: 8
   zampe sul ragno, 2 ali sul drago).
3. Due clip per creatura; `rig.deformation` verde su tutte.
4. Il lupo: in Blender i piedi non si incrociano con la clip propria; in Unreal 5.8, con lo script d'import della
   funzione 5, la camminata di Manny ritargettata non incrocia i piedi (stessa misura, letta con
   `AnimPoseExtensions`) e c'è un'immagine a metà passo.
5. Un'immagine per creatura a metà clip e un video corto in `output/creature-families/`, e ogni creatura nella coda
   di revisione per l'approvazione dell'utente.
6. Test: `PYTHONPATH=. pytest -q` e `ruff check` verdi sui file toccati; i test Blender con `--factory-startup`.

## 11. Com'è andata la PR 1 (15/10/2026)

**Su cosa ho cambiato idea, misurando.**
- **Digitigrado si riconosce dalla pianta, non dalle articolazioni.** SkinTokens mette il garretto del lupo e la
  caviglia del cavaliere (con gli stivali) quasi alla stessa altezza: 18% e 16% della gamba. La mesh invece li separa
  nettamente: misurando quanto la pianta poggia dietro la caviglia, in lunghezze di piede, il cavaliere e gli altri
  quattro umanoidi rigati in `output/` stanno fra −0,53 e −0,74 (il tallone), il lupo fra −0,03 e −0,07. Soglia −0,25.
  Senza mesh, solo un garretto oltre il 25% della gamba dice digitigrado.
- **Il fronte viene dalle anche, non dai piedi.** I piedi del lupo sono girati di 17° rispetto al corpo (uno più aperto
  dell'altro). Dalle anche (perpendicolare alla linea fra le due), il fronte è esatto; i piedi dicono solo da che
  parte.
- **Ipotesi nuova sulle gambe incrociate in Unreal**, da verificare nella PR 4: il profilo della funzione 5 e
  `kimodo_retarget.py` orientano il personaggio dai piedi (`character_frame`). Sul lupo questo lo gira di 17°; se il
  retarget mette i piedi con obiettivi IK nel mondo mentre le anche sono girate, i piedi passano uno davanti
  all'altro. Basterebbe orientarlo dalle anche, come fa `creature_family.py`.
- **Le soglie di `rig.deformation`**: le mosse Kimodo già usate sugli umanoidi generati misurano 0,25–1,2% di lati
  stirati e 0,3–1,7% di triangoli rovesciati (ascelle e anche che si piegano). Soglie: avviso a 0,5%, errore a 1,5%
  (lati) e 1,0% (triangoli), e **solo avviso** finché il pilota non le tara.
- **La pelle che si attraversa** (`BVHTree.overlap`) non è nella PR 1: va nei seguiti. Bastano per ora lati, triangoli e
  piedi incrociati.

**Risultati sul lupo mannaro** (rig esistente, nessuna GPU per le clip: 4 secondi):

| Clip | Lati stirati | Triangoli rovesciati | Piedi incrociati |
|---|---|---|---|
| `digitigrade-walk` | 0,05% | 0,09% | no |
| `digitigrade-attack` | 0,47% | 0,46% | no |

Le immagini a metà clip (di fianco e di fronte) mostrano piedi fermi in appoggio, gambe che si alternano senza
incrociarsi, braccia alzate senza strappi alle spalle. Il ginocchio della gamba in volo rientra un po' verso il centro:
lo segnalo, non è una deformazione.

**Seguiti**
- Pelle che si attraversa (`BVHTree.overlap` fra parti lontane), quando servirà davvero.
- Orientare dalle anche anche `kimodo_retarget.character_frame` e il profilo Unreal (PR 4, dopo la #6 e la #7).
- La coda del lupo è nella mesh ma non nello scheletro di SkinTokens: segue il bacino. Con la famiglia giusta si
  potrebbe chiedere a SkinTokens un altro tentativo (altro seme) quando manca una coda visibile.

