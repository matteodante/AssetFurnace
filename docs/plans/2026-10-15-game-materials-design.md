# Funzione 8, materiali pronti per il gioco: progetto

> **Superato dalla [v2](2026-10-15-game-materials-v2-design.md)** (15/10/2026): l'utente ha chiesto di rifare il metodo
> (zone per materiale, niente luce cotta all'origine, controlli imparati). Qui restano la storia e i Seguiti.

Data: 15 ottobre 2026. Stato: **approvato dall'utente e implementato** (ramo `claude/feat-game-materials-62a00b`).
Cosa è cambiato implementando, rispetto alla proposta qui sotto: sezione [Esito](#esito-dellimplementazione) in fondo.

Cosa esce, per un GLB generato: una maschera di tinta a 3 zone, un controllo della luce cotta, normal e rugosità dove
mancano, una texture ORM. Tutto in `image_to_3dlab/materials.py` (una funzione Python) più uno script
`scripts/game_materials.py`. Il passo `materials` di `cli.py` lo aggancia dopo il merge la sessione della funzione 1.

## Cosa ho misurato prima di proporre

Prove esplorative su 5 GLB già in `output/` (nessuna generazione), con uno script usa e getta fuori dalla repo:

| Asset | Albedo | Zone di colore (3 gruppi) | Luce cotta (rapporto lato chiaro / lato scuro) |
|---|---|---|---|
| Guerriero 18k (finito) | scuro (L* medio 16) | ben separate: distanza minima ΔE 27, aree 26/26/48 % | 1,25 |
| Guerriero grezzo Pixal3D | scuro | separate: ΔE 11 | 1,10 |
| Ragazza 40k (finita) | chiaro | separate: ΔE 10 | 1,28 |
| Ragazza giovane grezza | chiaro | — | 1,15 |
| Testuggine | **quasi bianco**, senza colore | **nessuna zona**: ΔE 3,6 | — |

Prova inversa: ho aggiunto ai campioni del guerriero e delle ragazze una luce finta (una lambertiana con luce ambiente
dal 100 % al 20 %). Il controllo ritrova il rapporto vero: ambiente 0,4 (rapporto vero 2,5) → misurato 2,45–3,5;
ambiente 0,2 (vero 5) → 5,1–8,5; e ritrova anche la direzione della luce. Gli asset veri, che non hanno luce cotta,
stanno fra 1,10 e 1,28: è il "rumore di fondo" dato dai colori veri (stivali scuri in basso, capelli chiari in alto).

Altro trovato: Pixal3D scrive già una texture metallo-rugosità **che varia** (rugosità con deviazione 36–52 su 255,
metallo alto sull'armatura del guerriero). Va tenuta, non rifatta: "dove mancano" vuol dire davvero dove mancano.

## 1. Maschera di tinta a 3 canali

**Da dove vengono le zone.** Dalle regioni di colore della superficie, non da tagli geometrici:

1. Si campiona la superficie (circa 60.000 punti, pesati per area) e per ogni punto si legge l'albedo.
2. Raggruppamento k-means nello spazio colore CIELAB, con la luminosità pesata la metà della tinta, così un'ombra non
   crea una zona nuova. Si prova k da 2 a 6 e si fondono i gruppi più vicini di ΔE 10.
3. Le 3 zone più grandi diventano R, G e B, in ordine di area. Le altre (occhi, piccoli dettagli) restano a 0, cioè non
   si tingono.
4. La maschera si disegna **nello spazio delle UV, texel per texel**: ogni texel va alla zona col colore più vicino,
   poi un filtro di maggioranza (5 px) toglie i puntini e i bordi si ammorbidiscono di 1–2 px. I bordi seguono così il
   disegno vero (dove finisce la cinghia, inizia la tunica), cosa che un taglio geometrico non sa fare.
5. La maschera si allarga di 8 px nei margini fra le isole UV, per non avere cuciture con le mipmap.

Le regioni geometriche di `blender_split_regions.py` (testa, corpo, coda lungo un asse) non vanno bene come zone di
tinta: tagliano a metà il disegno. Restano un'opzione: con `--zones materials` ogni slot di materiale (quelli creati da
`blender_split_regions.py`) diventa una zona, per chi vuole tingere testa e coda separate.

**Asset senza zone chiare** (come la testuggine): non si inventano. Il controllo `tint_zones` dice quante zone ha
trovato:
- 3 zone con ΔE ≥ 10 fra loro e ognuna ≥ 5 % dell'area: `pass`;
- 1 o 2 zone: `warn`, la maschera esce con i canali mancanti a 0;
- nessuna zona (albedo uniforme): `fail`, nessuna maschera, l'asset torna indietro.

**Formula di tinta** (da usare uguale in Blender e in Unreal). In luce lineare, per la zona *i* con maschera *m*,
colore medio originale *c* e tinta scelta *t*:

```
g   = luminanza(albedo) / luminanza(c)       dettaglio della zona, 1 in media
out = albedo + m · g · (t − c)               poi limitato a 0..1
```

Con `t = c` l'asset resta **identico** all'originale: i parametri di partenza sono i colori medi delle zone, scritti
nel file accanto. Cambiando `t` si sposta il colore medio e il dettaglio di luminosità resta. Scartata la
moltiplicazione canale per canale (`albedo · t / c`): su una zona rossa tinta di blu moltiplica per 10 il rumore del
canale blu.

## 2. Controllo della luce cotta

Il controllo rimanda indietro e non corregge. Lavora per zona (le stesse della maschera), perché ogni materiale ha la
sua luminosità. Restituisce tre record nel formato comune dei controlli, così la funzione 7 li registra uno per uno:

```json
{"check": "baked_light_direction", "status": "pass", "value": 1.25, "threshold": {"warn": 1.5, "fail": 2.0},
 "message": "lit side 1.25x brighter than shadow side (worst zone R)"}
```

| Controllo | Cosa misura | Come | Soglie (warn / fail) |
|---|---|---|---|
| `baked_light_direction` | Ombre e luci di una luce direzionale dipinte nell'albedo | Per ogni zona, regressione del logaritmo della luminanza lineare sulla normale della superficie: `log Y = a + g · n`. Il rapporto `exp(2·|g|)` è quanto il lato illuminato è più chiaro del lato in ombra. Si riporta la zona peggiore, pesata per area | 1,5 / 2,0 |
| `baked_light_occlusion` | Occlusione ambientale (pieghe scure) dipinta nell'albedo | Solo se c'è l'AO cotta da `blender_bake_ao.py`: regressione `log Y = a + s · log AO` per zona. `s = 0` albedo pulito, `s = 1` AO interamente dipinta | 0,3 / 0,6 |
| `baked_light_highlights` | Riflessi speculari dipinti | Quota della superficie di una zona più chiara di 2,5 volte la sua mediana e quasi senza saturazione | 0,5 % / 2 % |

Le soglie partono dalle misure sopra: gli asset veri stanno a 1,10–1,28, una luce dipinta con ambiente 0,6 dà già
1,6–2,2. Si ritarano sul pilota con le valutazioni dell'utente. La "luminosità non uniforme" del piano rientra nel
primo controllo: una luce dipinta è proprio una luminosità che segue l'orientamento della superficie.

Se il controllo fallisce, gli altri passi si fermano: dall'albedo con luce cotta uscirebbero zone e normal sbagliate
(un'ombra dipinta diventerebbe un solco).

## 3. Normal e rugosità ricavate dal colore

**Si riusa**:
- l'idea di Materialize (Bounding Box Software): altezza dall'albedo per bande di frequenza, normal dall'altezza. Il
  codice **non si copia**: Materialize è GPL-3, AssetFurnace è Apache-2.0. Verificato su GitHub il 15/10/2026;
- da `scripts/surface_detail.py`: la convenzione della normal glTF (+Y in alto, (128,128,255) piatta), la rugosità
  dalla luminanza (più scuro = più ruvido), la maschera lucida per gli occhi, l'ordine dei canali ORM;
- numpy e scipy (BSD) per sfocature e gradienti.

**Si costruisce** (in `materials.py`, perché il pacchetto non può importare da `scripts/`; `surface_detail.py` resta
com'è e più avanti può chiamare il pacchetto):
- altezza a più scale: somma pesata di 4 bande (differenze fra sfocature gaussiane di 1, 2, 4, 8, 16 px) della
  luminanza, invece della banda singola di oggi;
- **sfocature mascherate per isola UV**: si sfoca dividendo per la copertura, così il nero dei margini non entra nelle
  isole e non disegna un bordo in rilievo lungo ogni cucitura;
- rugosità: un valore di base per zona (una zona è un materiale) più la variazione della luminanza, con ampiezza
  piccola.

**Dove mancano** significa:
- normal: solo se il materiale non ha `normalTexture`;
- rugosità: solo se manca la texture metallo-rugosità o se il suo canale G è costante (deviazione < 2 su 255, come
  per TRELLIS); con Pixal3D di solito c'è e si tiene;
- metallo: si tiene il canale B esistente; se manca, 0 (o un valore per zona da parametro).

## 4. ORM e come viaggia col glTF

- **Formato**: una PNG RGB lineare (non sRGB). R = occlusione, G = rugosità, B = metallo. È lo stesso schema di glTF,
  dove una sola immagine serve sia da `occlusionTexture` sia da `metallicRoughnessTexture`, ed è lo schema ORM di
  Unreal.
- **Occlusione**: si cuoce con `scripts/blender_bake_ao.py` (già presente, Blender senza finestra, sulle UV esistenti).
  Se non c'è, R = 255 (nessuna occlusione) e il materiale non dichiara `occlusionTexture`.
- **Nel GLB**: il file si modifica **nel JSON, senza riesportare la geometria** (`image_to_3dlab/glb_turn.py` ha già
  `parse_glb` e `build_glb`). Riesportare con trimesh perderebbe scheletro e animazioni dei GLB riggati. Si
  aggiungono le immagini nuove; la maschera si dichiara in `materials[i].extras.assetfurnace.tintMask`
  (`{"index": <texture>, "texCoord": 0}`) insieme alle zone.
- **Accanto al GLB**, perché non tutti i programmi leggono gli `extras`:

```
<asset>.glb                      con normal, ORM e maschera dentro
<asset>.materials.json           zone (canale, colore medio lineare, area), file, esito dei controlli, formula
T_<asset>_Tint.png               R, G, B = zone (nome deciso con il profilo Unreal)
T_<asset>_N.png                  se ricavata
T_<asset>_ORM.png
```

  I nomi `T_` vanno concordati con la sessione del profilo Unreal (funzione 5): li sceglie lei, io espongo il
  parametro.
- La provenienza: il `.provenance.json` dell'asset riceve una voce per il passo, senza modelli (niente AI).

## Alternative scartate

- **Togliere la luce con l'AI** (delighting): strumenti immaturi o con licenze strette, già escluso nel piano.
- **Codice di Materialize, AwesomeBump, GeneradorMapsPbr, Bulk Texture Creator**: GPL-3, non si copia in una repo
  Apache-2.0. **DeepBump** e pypbr con Depth Anything: usano reti neurali, il piano chiede "senza AI".
- **Zone da segmentazione 3D** (SAMPart3D e simili): codice di ricerca, Linux con CUDA. Si rivaluta se il colore non
  basta sul pilota.
- **Zone da tagli geometrici** come scelta predefinita: tagliano a metà il disegno. Restano come opzione `materials`.
- **KHR_materials_variants**: varianti fisse, non colori da parametro.
- **Tinta canale per canale**: amplifica il rumore (vedi sopra).

## Rischi

- **Albedo molto scuri** (il guerriero ha L* medio 16): la tinta parte da poca luminanza e i colori chiari vengono
  spenti. Si vede nel render di prova; se serve, un parametro di "luminosità della zona" in più.
- **Colore vero confuso con luce**: un asset scuro in basso e chiaro in alto (stivali, capelli) dà già 1,1–1,3. Le
  soglie lasciano margine; un falso allarme diventa un `warn`, non un `fail`.
- **Normal inventate**: un segno dipinto diventa un solco (lo dice già `surface_detail.py`). Forza bassa, e solo dove
  mancano.
- **Extras ignorati da Interchange**: per questo il file accanto è la fonte sicura. Lo verifica la sessione Unreal.
- **Nessun mostro fra i GLB esistenti**: la prova si fa sul guerriero 18k (umanoide, 3 zone nette). Il mostro vero
  arriva col pilota.

## Come provo i criteri di accettazione

1. **Test prima del codice**, su fixture piccole fatte nel test (una sfera UV a 3 bande di colore):
   - la maschera ha 3 canali che coincidono con le bande;
   - con `t = c` l'uscita è identica all'albedo;
   - albedo piatto → `pass`; luce lambertiana cotta con ambiente 0,4 → `fail` con la direzione giusta; macchia
     bianca → `baked_light_highlights` in `fail`;
   - albedo uniforme → `tint_zones` in `fail`;
   - normal piatta su albedo piatto, nessun bordo in rilievo lungo le cuciture UV;
   - ORM con i canali al posto giusto; il GLB riscritto conserva scheletro e animazioni (fixture riggata minima).
2. **Asset veri**: il controllo sui 5 GLB di `output/` (valori registrati nel piano) e sul guerriero con luce finta.
3. **Render in Blender** (Blender 5.2 senza finestra, Cycles su CPU per non pesare sulla GPU condivisa), con la
   formula di tinta costruita in nodi dal file accanto:
   - prima: colori originali, luce frontale come nel concept;
   - dopo: le 3 zone ricolorate da parametri, stessa luce;
   - dopo, con una luce diversa (laterale bassa, da dietro): nessuna ombra cotta deve "combattere" la luce nuova;
   - per confronto, la versione con luce finta che il controllo boccia.

   Un foglio unico con i 4 render, allegato al rapporto.
4. **Unreal**: la prova con i parametri del materiale la fa la sessione del profilo Unreal dopo il merge.

## Domande per l'approvazione

1. Va bene che le zone vengano dal colore (con `materials` come alternativa), e che un asset senza zone torni indietro
   invece di ricevere zone geometriche automatiche?
2. Soglie iniziali della luce cotta: 1,5 avviso, 2,0 rifiuto. Va bene partire da qui e ritararle sul pilota?
3. Va bene la formula di tinta "a scostamento" qui sopra? Unreal deve usare la stessa.

## Esito dell'implementazione

Codice: `image_to_3dlab/materials.py` (`make_game_materials`, `run_checks`), `scripts/game_materials.py`,
`scripts/blender_tint_preview.py`. Test: `tests/test_materials.py`, `tests/test_blender_tint_preview.py`.

**Cambiato rispetto alla proposta, con il motivo:**

| Punto | Proposta | Ora | Perché |
|---|---|---|---|
| Soglie `baked_light_direction` | 1,5 / 2,0 | **1,3 / 1,45** | Una luce cotta apposta nell'albedo del guerriero (ambiente 35 %) misura solo 1,77 (1,55 prima di usare le normali dei vertici): ombre proiettate e colori mescolati diluiscono quello che una sfera mostra pulito (2,7). Con 1,5 / 2,0 passava con un avviso. Gli albedo puliti stanno fra 1,01 e 1,26. **Da confermare con l'utente**; si torna indietro cambiando una riga (`LIGHT_RATIO`) |
| Gruppi di colore | k-means fino a 6, fusione sotto ΔE 10 | fino a **12**, fusione sotto 10 **o se sono lo stesso pigmento** (stessa tinta ±12°, stesso rapporto saturazione/luminosità ±20 %) | Con 6 la pelle del guerriero finiva nella stoffa rossa e si tingeva di verde; con 12 la stoffa rossa si spezzava in "rosso in luce" e "rosso in ombra". Un'ombra cambia luminosità e saturazione insieme, non la tinta |
| Gruppi per i controlli della luce | luminosità a peso 0,25 | gli **stessi** delle zone (peso 0,5) | A 0,25 nero e grigio dell'armatura si fondevano e la loro differenza vera sembrava luce (1,73 su un asset pulito) |
| `baked_light_highlights` | quota di punti chiari e senza colore | conta solo se i punti sono **rivolti nella stessa direzione** (≥ 70 % entro 45° dalla loro normale media; rampa da 40 % a 70 %) | La stoffa color crema accanto al viola scuro di una ragazza dava il 3,4 % (rifiuto falso). Un riflesso dipinto viene da una luce sola e guarda da una parte; un materiale chiaro guarda ovunque |
| `tint_zones` | 1–2 zone avviso, 0 rifiuto | 1 zona (albedo uniforme o un solo pigmento) **rifiuto**, 2 avviso, 3 o più passa | Una zona sola è "nessuna zona da separare" |

**Misure sui GLB in `output/`** (valori finali):

| Asset | Zone | Luce (rapporto) | Riflessi |
|---|---|---|---|
| Guerriero grezzo Pixal3D | 3 | 1,04 | 0 |
| Guerriero 9k / 18k finiti | 3 / 3 | 1,26 / 1,22 | 0 / 0 |
| Guerriero 18k con luce cotta apposta | 2 | **1,77, rifiutato** | 0,11 % |
| Ragazza grezza → finita 40k | 1 → 2 | 1,02 → **1,56, rifiutata** | 0 |
| Altra ragazza grezza → finita 100k | 3 → 2 | 1,02 → 1,09 | 0 |
| Ragazza giovane grezza | 3 | 1,11 | 0 |
| Testuggine (albedo bianco) | **1, rifiutata** | 1,01 | 0 |

La finitura che proietta la foto del concept sull'asset (passo `photo` di `retopo_repaint.py`) porta con sé la luce del
concept: la ragazza 40k passa da 1,02 a 1,56. È proprio il caso che il controllo deve rimandare indietro.

**Prova in Blender** (`blender_tint_preview.py`, Cycles su CPU, guerriero 18k: nel parco non c'è un mostro): originale
con la luce del concept, zone ricolorate da parametri, le stesse due con una luce bassa laterale, e la versione con luce
cotta apposta. La stoffa passa dal rosso al verde, il cuoio di guanti, stivali e cintura all'oro, l'emblema resta
dorato e la pelle non si tinge. Con la luce laterale non compaiono ombre dipinte; nella versione con luce cotta sì, e il
controllo la rifiuta.

**Limiti visti nella prova:**
- Zone quasi nere (il guerriero ha una zona a L* 4): la tinta le cambia poco, perché la formula conserva la luminosità.
  Per una zona così servirà un parametro di luminosità della zona, oppure tingerla in Unreal con un colore più scuro.
- Qualche texel rossastro della pelle (guance, labbra) cade nella zona della stoffa rossa e si tinge. Si corregge con
  una maschera fatta a mano (`feature_mask.py`) o con un ritocco della maschera.
- Il controllo della luce è tarato su 10 asset; il pilota lo ritara.

**Riusato**: `edit_glb.EditGLB` (lettura validata del GLB, accessor e trasformazioni dei nodi), `glb_turn.build_glb`,
la convenzione della normal di `surface_detail.py` (stessi valori, verificato da un test), `blender_bake_ao.py` per
l'occlusione, l'idea di Materialize per l'altezza a bande (solo l'idea: è GPL-3).

**Nello studio** (richiesto dall'utente dopo l'approvazione): nel visore di un asset della Libreria, il pannello
"Game materials" (`viewer/studio/materials.js`, `viewer/materials_api.py`) avvia il passo sull'asset scelto, mostra
l'esito dei controlli e le anteprime delle texture, sovrappone le 3 zone al modello e le ricolora dal vivo con tre
selettori. La ricolorazione è uno shader che esegue la stessa formula a scostamento in luce lineare; un test la
confronta con `apply_tint` di Python. I risultati vanno in `output/materials/<modello>__<hash>/`, uno per modello di
partenza, e rifare il passo li sostituisce. Righe di aggancio nei file condivisi: `generate_api.py` (3),
`assets_api.py` (la cartella `materials` non è un modello nuovo), `studio.html` (2), `app.js` (5), `viewer.js` (2).

La maschera si chiama `T_<Nome>_Tint.png`, come deciso con il profilo Unreal.

**Dalla review di Codex sulla PR** (4 giri, tutti i punti verificati e corretti con un test): il passo legge l'albedo
come lo disegna il renderer (solo la scena mostrata, ripetizione e trasformazione della texture, trasparenze, colori e
normali dei vertici, strip e fan), conserva occlusione già presente, provenienza e scheletro (nello studio parte dal
modello riggato), rifiuta le combinazioni che non può impacchettare senza errori (mappe su altre UV, AO con trasformazione,
materiali unlit) e dice quali materiali senza texture restano come sono.

**Nel registro dei controlli della funzione 7** (`image_to_3dlab/checks/material_checks.py`): `material.tint_zones`,
`material.baked_light_direction` e `material.baked_light_highlights`, con una sola analisi per GLB. Le zone di tinta si
chiedono solo nei profili `creature` ed `equipment_part`: un sasso o un barile non devono essere ricolorabili. Il
controllo dell'occlusione resta allo script, perché richiede l'AO cotta che l'`Asset` del registro non porta.

**Fuori dal mio perimetro, da fare dopo il merge**: il passo `materials` in `cli.py` (sessione della funzione 1); la
registrazione dei controlli nella coda (funzione 7); il Material Function di Unreal che rifà la formula dal
`.materials.json` (funzione 5).

## Seguiti

Punti P2 della review di Codex rimandati con la regola dell'utente (si correggono solo gli errori che un utente
vedrebbe), uno per riga:

- Alfa dei colori dei vertici (`COLOR_0` a 4 componenti) nei materiali MASK e BLEND: oggi conta solo l'alfa della texture e del fattore. La pipeline non produce ritagli dipinti sui vertici. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4287300576)
- Anteprima nello studio: la maschera usa il filtro lineare anche quando l'albedo chiede "nearest". Tocca solo l'anteprima di asset in pixel art. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4287300593)
- Normal ricavata da texture ripetute (tiling): sfocature e derivate non trattano come vicini i bordi opposti. Gli atlanti generati non si ripetono. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4287300614)
- Colori dei vertici nelle mappe prodotte: oggi contano nei controlli ma non nella maschera, nella rugosità e nella normal. La pipeline non dipinge zone sui vertici. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972809)
- MIRRORED_REPEAT: un triangolo che attraversa una piega viene piegato solo nei vertici. Layout non prodotto dalla pipeline. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972821)
- Texture con una sorgente di fallback e una WebP/AVIF/Basis: si legge il fallback, di solito identico. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972842)
- Morph target con pesi di partenza diversi da zero: i controlli usano la forma di base. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972875)
- Riflessi confrontati con il vicinato della stessa zona di colore (caso: un emblema bianco su stoffa scura, rivolto da una parte). Cambia la sensibilità del controllo: da decidere con i dati del pilota. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972887)
- Albedo non quadrati: la forza della normal si scala sulla larghezza per entrambi gli assi. Gli atlanti generati sono quadrati. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972898)
- Con `--zones materials` i controlli della luce raggruppano ancora per colore e non per materiale. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4286972921)
- CLAMP_TO_EDGE: un triangolo tutto fuori dall'atlante non riempie i texel del bordo che il renderer usa. Layout non prodotto dalla pipeline. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4287644894)
- L'immagine AO data con `--ao` si apre solo dopo l'analisi: un file sbagliato fallisce tardi, non dà un risultato sbagliato. [Thread](https://github.com/matteodante/AssetFurnace/pull/3#discussion_r4287644934)
