# Funzione 6: famiglie di creature (progetto)

**Ricerca sugli approcci più recenti (16/10/2026)**: [`2026-10-16-creature-families-research.md`](2026-10-16-creature-families-research.md).

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
- `rig.deformation` misura anche le clip accanto al rig, ma l'esito resta legato all'hash del solo rig: se una clip
  cambia dopo un'approvazione, la coda di revisione non lo vede (terzo giro di review di Codex). Va deciso con chi
  cura la funzione 7: includere le clip misurate nella versione dell'asset, o rivedere ogni clip a sé.
- La coda del lupo è nella mesh ma non nello scheletro di SkinTokens: segue il bacino. Con la famiglia giusta si
  potrebbe chiedere a SkinTokens un altro tentativo (altro seme) quando manca una coda visibile.

## 12. Com'è andata la PR 2: ragni e serpenti (15/10/2026)

Creature generate con la catena (Qwen-Image → Pixal3D → finitura con `--skip-paint`, perché il ridipinto gira solo su
Apple Silicon) e rigate dallo studio con *Recognise it for me*.

**Cosa ho imparato dalle immagini.** Per un modello da animare la camera va **all'altezza del suolo**: il primo ragno,
visto dall'alto, è uscito da Pixal3D inclinato (ricostruisce nel riferimento della camera) e il suolo non era più
orizzontale. E un serpente **avvolto su se stesso** non si rigga: SkinTokens gli ha dato 21 ossa su un solo giro di
spira. Disteso, di fianco, ha avuto una catena pulita di 13 ossa. Qwen-Image disegna il basilisco con le zampe (quello
della tradizione): per la famiglia serpente il prompt dice "serpente, senza zampe".

**Cosa ho cambiato nel riconoscimento**, misurando sui rig veri:
- una zampa deve solo arrivare **vicino** al suolo (entro il 30% dell'altezza a cui si attacca): SkinTokens ferma le
  catene delle zampe di un ragno ad altezze diverse;
- trovate tre zampe, contano anche le catene lunghe che vanno **di lato e in basso** (le altre zampe del ragno);
- una zampa deve **scendere** (dislivello almeno un quarto della sua lunghezza): un serpente disteso ha il ventre a
  terra e non è una gamba;
- con cinque o più zampe il fronte è quello del modello generato (la camera): zampe che si irraggiano non dicono dove
  sta la testa; un serpente guarda lungo il corpo, verso la testa;
- **zampe gemelle**: SkinTokens ha messo due catene dentro la stessa zampa del ragno (punte a 1 cm). Muoverle in fase
  diversa strappava la pelle fra le due (0,95% di lati stirati); ora si muovono insieme.

**Risultati**

| Creatura | Famiglia riconosciuta | Clip | Lati stirati | Triangoli rovesciati |
|---|---|---|---|---|
| Ragno (seme 23, camera a terra) | multileg, 6 zampe, fiducia alta | walk | 0,37% | 0,42% |
| | | attack | 0,55% | 0,52% |
| Serpente (seme 8, disteso) | serpent, fiducia alta | slither | 0,00% | 0,01% |
| | | attack | 0,00% | 0,02% |

L'attacco del ragno resta appena sopra la soglia di avviso (0,5%): al culmine dell'impennata si vede una membrana
chiara tesa fra due zampe destre che la mesh di Pixal3D ha fuso. È un difetto della mesh più che della clip; ho già
ridotto l'impennata (da 12° a 7°). Il ragno ha 6 zampe riconosciute su 8: due sono alzate in avanti nella posa del
modello e risultano "ali" (una catena laterale lunga che sale). Per un ragno non cambia le clip, ma l'etichetta è
sbagliata.

**Seguiti della PR 2**
- Zampe alzate di un ragno scambiate per ali: con cinque o più zampe, una catena laterale lunga attaccata allo stesso
  mozzo andrebbe trattata come zampa.
- Una coda lunga tenuta di lato e verso il basso, su un quadrupede, può passare per una quinta zampa (terzo giro di
  review di Codex): serve una regola che riconosca le code fra le catene laterali, non solo fra quelle all'indietro.
- Un "obiettivo creatura" nelle ricette dei prompt (`image_to_3dlab/prompt_recipes.json`), con camera a terra, corpo
  intero e niente spire, invece della T-pose dei personaggi. Oggi lo studio aggiunge la T-pose a ogni personaggio.

## 13. Com'è andata la PR 3: draghetto e pianta (16/10/2026)

Creature dalla catena (Qwen-Image → Pixal3D → finitura `--skip-paint`), camera all'altezza del suolo, rigate dallo
studio con *Recognise it for me*. SkinTokens ha dato a entrambe scheletri buoni; il riconoscimento invece ha sbagliato
la prima volta, in modi che ora sono regole e test:

- **Draghetto**, preso per "multileg" con 5 zampe e un'ala: la **coda appoggiata a terra e girata di lato** passava per
  una quinta zampa (il caso che l'ultima review della PR 2 aveva rimandato), e l'**ala sinistra, alta e spinta in
  avanti**, era scelta come collo. Ora una catena che si allunga sul terreno molto più di quanto scenda, senza mai
  salire sopra l'attacco, è una coda (se è l'unica: tante sono le zampe di una creatura sdraiata), e la testa deve
  stare vicino alla mezzeria.
- **Fiore carnivoro**, preso per umanoide: SkinTokens l'ha **radicato nel fiore**, con lo stelo che scende a un
  ventaglio di radici. Ora una zampa si attacca ad almeno un quarto dell'altezza e a un'articolazione con almeno due
  rami, il controllo del trascinamento guarda tutto il ramo (un ventaglio di radici sparse a terra non è un piede) e lo
  stelo va dal ventaglio alla punta, qualunque sia la radice dello scheletro. La base resta ferma anche
  nell'orientamento, così le radici non ruotano.

Tutti i rig veri in `output/` (4 umanoidi, 2 lupi, ragno, serpente, draghetto, pianta) sono riconosciuti giusti.

| Creatura | Famiglia | Clip | Lati stirati | Triangoli rovesciati |
|---|---|---|---|---|
| Draghetto | winged, 4 zampe, 2 ali, fiducia alta | hover | 0,56% | 0,82% |
| | | attack | 0,43% | 0,15% |
| Fiore carnivoro | plant | sway | 0,00% | 0,00% |
| | | attack | 0,04% | 0,05% |

Il volo del draghetto resta in avviso: il battito piega molto le spalle delle ali. A vista le membrane si piegano
senza strappi; è il primo punto da guardare se si vuole un battito più ampio.

**Seguiti della PR 3**
- La coda del draghetto, girata di lato, non è riconosciuta come coda (resta nel tronco e segue il bacino): le code
  vanno cercate anche fra le catene che piegano di lato.
- `creature_clips.py` rilanciato su un rig già animato lascia le clip della famiglia precedente; nello studio non
  succede (ogni rig ha la sua cartella), da riga di comando sì.


## 14. Libreria di clip CC0 (approvata il 16/10/2026)

L'utente ha approvato la proposta della ricerca (`2026-10-16-creature-families-research.md`): clip fatte a mano e
CC0, adattate allo scheletro di SkinTokens con un retarget a catene, più GenMM (Apache-2.0) per le varianti. Le clip
procedurali restano: provano il rig e coprono le famiglie che la libreria non ha (le piante).

### Sorgenti

- **Mesh2Motion**, commit `653a969`: `LICENSE-CC0.MD` copre "tutti i modelli 3D, i rig e le animazioni", il codice è
  MIT ma non lo usiamo. Si scaricano solo i sei `static/animations/<creatura>-animations.glb` che servono, cioè
  uccello, drago, volpe, kaiju, serpente e ragno (3,9 MB in tutto). Ogni file ha il suo SHA-256 fissato. Restano fuori
  cavallo e squalo (per ora inutili), la cartella `CarnegieMellonAnimations` (licenza non CC0) e `models-variation`
  (licenza indicata file per file).
- **Quaternius**: CC0 dichiarato, ma senza un indirizzo stabile da cui scaricare (Google Drive, itch.io). Lo valuto in
  una PR a parte, eventualmente da uno zip che l'utente scarica a mano, controllato con lo SHA-256.
- **GenMM**: solo codice Apache-2.0, **senza pesi** (lavora per vicini più prossimi sulla clip stessa). Va in
  `vendor/GenMM` a un commit fissato, gira su CPU.

Il pacchetto è registrato nel catalogo (`viewer/backend_catalog.py`, una voce `kind="tool"`) e si installa con
`scripts/bootstrap_clip_library.py`, che dice cosa scarica e quanto pesa, chiede conferma e controlla ogni SHA-256
prima di tenere un file. Il pacchetto finisce in `vendor/clip-library/mesh2motion/`, insieme a un `SOURCE.json` con
URL, commit, licenza e hash.

### Licenze (registrate il 16/10/2026)

Rilette per ogni componente, pesi compresi, e con i file che vengono scaricati in automatico.

| Componente | Codice | Pesi e dati |
|---|---|---|
| Mesh2Motion (animazioni) | MIT, non usato | Animazioni CC0 (`LICENSE-CC0.MD` al commit `653a969`) |
| Quaternius | – | CC0 dichiarato, da registrare pacchetto per pacchetto al download |
| GenMM | Apache-2.0 (commit `aee9bec`) | Nessun peso e nessun download: lavora per vicini più prossimi sulla clip stessa |
| SkinTokens | MIT | Pesi MIT (`VAST-AI/SkinTokens` alla revisione fissata `79736ca`); di Qwen3-0.6B solo configurazione e tokenizer, Apache-2.0 |
| Kimodo (mosse preimpostate) | Apache-2.0 | `nvidia/Kimodo-SOMA-RP-v1.1`, NVIDIA Open Model License; encoder del testo LLM2Vec (MIT) su Meta-Llama-3-8B-Instruct (Llama 3 Community License), non eseguito: le mosse sono già generate |

Di GenMM si usano solo `GenMM.py`, `nearest_neighbor/` (che riprende Efficient-GPNN, Apache-2.0) e le trasformazioni.
L'add-on per Blender (`__init__.py`) è GPL-3 e non viene copiato. Le utilità BVH (`dataset/bvh/`) non dichiarano una
licenza e non servono: le clip le leggiamo noi dai GLB.

### Quale clip per quale famiglia

| Famiglia | Sorgente | Clip |
|---|---|---|
| quadruped | volpe | idle, walk, run, attack (morso), death |
| digitigrade | kaiju (bipede con coda, gambe digitigrade, braccia) | idle, walk, attack, hit, death |
| multileg | ragno (8 zampe; con 6 si salta una riga) | idle, walk, attack, hit, death |
| serpent | serpente | idle, slither (side winding), attack (morso), hit, death |
| winged | drago con 4 zampe, uccello con 2 o nessuna | idle, walk, fly, glide |
| plant, humanoid | nessuna | restano le procedurali e i preset |

Le clip si chiamano `<famiglia>-<movimento>`, come le procedurali; quando il nome è lo stesso vince la clip della
libreria. Ciascuna ha la sua provenienza: CC0 con sorgente, commit e hash, più le licenze del rig.

### Retarget a catene, in numpy

Il retarget è una funzione pura, senza Blender: restituisce gli stessi fotogrammi `(turn, root)` di
`creature_motion`, che `creature_clips.py` sa già scrivere. La parte in `bpy` non cambia. Si può provare con
scheletri sintetici.

1. **Ruoli**: per la sorgente, una mappa scritta a mano nel manifesto (catene per nome: zampe con lato e posizione,
   ali, coda, colonna, collo); per il bersaglio, quella che dà già `creature_family.classify`.
2. **Abbinamento delle catene**: per lato e per posizione, dall'anteriore alla posteriore. Se il numero di zampe è
   diverso, vale l'indice normalizzato: con 6 zampe sul ragno da 8 se ne salta una.
3. **Rotazioni**: per ogni osso della sorgente si calcola la rotazione rispetto al riposo, nel sistema del personaggio
   (sinistra, su, avanti). Ogni osso del bersaglio prende la rotazione del tratto di sorgente che sta alla stessa
   frazione di lunghezza della catena, convertita nel sistema del bersaglio. Le ossa senza catena seguono il padre.
4. **Bacino e piedi**: lo spostamento del bacino è scalato sul rapporto fra le lunghezze delle gambe. Poi ogni piede
   va dove lo porta la sorgente, scalato, e la gamba è risolta con la FABRIK che `creature_motion` usa già. Il piede
   che nella sorgente poggia a terra resta a terra.
5. **Controllo**: `rig.deformation` sulle clip vere, con le stesse soglie delle procedurali.

### PR

1. Questo progetto, il manifesto e il bootstrap con il catalogo, la correzione dell'F841 in
   `tests/test_kimodo_retarget.py`.
2. Il retarget a catene e il suo aggancio in `creature_clips.py`, provati sulle creature vere della funzione 6
   (lupo, ragno, serpente, draghetto), con la tabella delle deformazioni.
3. Le varianti con GenMM: per esempio attese diverse e camminate più lunghe senza ripetizioni.
4. Quaternius, se c'è un modo verificabile per scaricarlo.

## 15. Com'è andata la PR 2 della libreria: retarget a catene (16/10/2026)

`scripts/chain_retarget.py` porta una clip della libreria sul rig, in numpy, e restituisce i fotogrammi
`(turn, root)` di `creature_motion`. `creature_clips.py` scrive prima le clip della libreria e poi le procedurali che
la libreria non sostituisce. Le catene delle sorgenti sono scritte a mano in `image_to_3dlab/clip_library.py`; un
test le confronta con i GLB installati e riporta ogni clip sul suo stesso scheletro (errore sotto il 2%
della taglia).

Le prove sui rig veri hanno cambiato tre scelte del progetto della sezione 14:

- **Gli arti girano come il corpo, non prendono la forma della sorgente.** Imporre la forma della zampa del
  kaiju al lupo stirava la pelle già al primo fotogramma, all'1–5% dei lati. Ora ogni segmento riceve la rotazione
  dal riposo del segmento della sorgente alla stessa frazione, presa attorno alla direzione della catena (l'imbardata):
  così una zampa che punta di lato si solleva come una che punta in avanti, invece di spazzare.
- **La scala è la lunghezza delle zampe, non l'altezza dell'anca.** Il ragno sorgente è piatto, con l'anca a 0,11 e
  zampe da un metro e mezzo: scalato sull'anca, un rig tozzo faceva passi tre volte più lunghi delle sue zampe. In
  attesa, il ragno è sceso dal 3,07% allo 0,31% di lati stirati.
- **Piede a terra o sollevato.** Se il piede della sorgente è a terra, la punta del bersaglio fa lo stesso spostamento
  della sorgente, scalato, quindi non scivola. Se è sollevato, va dove la porta la rotazione trasferita. Fra il 2% e
  l'8% della lunghezza della zampa le due regole si mescolano.
- Le zampe sono ordinate dal piede, non dall'attacco (le zampe di un ragno partono tutte dallo stesso punto). Due
  catene nella stessa zampa della mesh (le "gemelle" delle clip procedurali) copiano le stesse rotazioni.

Misure su 24 fotogrammi per clip, con le soglie di `rig.deformation` (avviso 0,5%, errore 1,5% per i lati stirati;
avviso 0,5%, errore 1,0% per i triangoli rovesciati). In fondo, la stessa clip misurata sul modello della sorgente:

| Creatura | Clip | Lati stirati | Triangoli rovesciati | Sorgente sul suo modello |
|---|---|---|---|---|
| Lupo (kaiju) | idle | 0,08% | 0,05% | 0,23% / 0,73% |
| | walk | 0,07% | 0,08% | 0,23% / 0,65% |
| | attack | 0,69% | 1,43% | 0,52% / 0,86% |
| | hit | 1,41% | 0,41% | 0,31% / 0,78% |
| | death | 2,51% | 1,70% | 1,17% / 2,11% |
| Ragno (ragno) | idle | 0,32% | 0,30% | 0% / 0% |
| | walk | 1,58% | 1,17% | 0,08% / 0,25% |
| | attack | 2,11% | 1,45% | 0% / 0% |
| | hit | 2,14% | 1,72% | 0,16% / 0% |
| | death | 3,08% | 2,04% | 0,28% / 0,56% |
| Serpente (serpente) | idle | 0,49% | 0,44% | 0,39% / 0% |
| | slither | 0,30% | 0,21% | 0,48% / 0,35% |
| | attack | 0,31% | 0,33% | 0,29% / 0% |
| | hit | 0,53% | 0,73% | 0,24% / 0,71% |
| | death | 0,35% | 0,34% | 0,19% / 0% |
| Draghetto (drago) | idle | 0,97% | 2,00% | 0,50% / 1,52% |
| | walk | 1,25% | 2,02% | 0,75% / 1,69% |
| | fly | 1,16% | 1,72% | 0,32% / 0,63% |
| | glide | 0,56% | 0,44% | 0,02% / 0% |

Lettura:
- Le clip di movimento di lupo e serpente sono pulite come o più delle sorgenti. Le procedurali restano accanto per
  le mosse che la libreria non ha (il draghetto tiene `hover` e `attack`).
- Attacco, colpo e morte del lupo e il drago sono vicini a ciò che fanno le sorgenti sul loro modello. Le pose sono
  estreme già lì: la morte del kaiju rovescia il 2,1% dei triangoli del suo modello.
- **Il ragno resta il caso difficile, e il motivo è il rig, non il retarget.** Ha catene a cui il riconoscimento non
  ha dato un ruolo: una zampa intera a destra (`bone_60`–`bone_66`) e un moncone fra due zampe a sinistra. Restano
  rigide mentre le vicine si muovono, e la pelle in mezzo si strappa. Farle seguire la zampa più vicina ha
  peggiorato le cose: prendeva anche testa e cheliceri, quindi l'ho tolto.

**Seguiti della PR 2 della libreria**
- Riconoscere tutte le zampe del ragno, comprese quelle che oggi restano senza ruolo; poi rimisurare.
- Le clip con pose estreme (attacco, colpo, morte) superano le soglie anche sulle sorgenti: valutare soglie per tipo
  di clip, o un controllo che confronti con la sorgente invece che con un numero fisso.
- Il collo del bersaglio: oggi la testa gira solo con l'ultimo segmento del collo.
