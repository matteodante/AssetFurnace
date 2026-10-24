# Funzione 5: profilo d'uscita per Unreal (design)

Data: 15 ottobre 2026. Stato: **fatto**: approvato dall'utente e implementato lo stesso giorno; risultato in fondo.
Piano generale: il piano di produzione (privato).

In una riga: un GLB finito di AssetFurnace diventa un **GLB per Unreal** (nomi, centimetri, origine, LOD,
collisione, ORM, ossa con i nomi di Manny) più una **scheda JSON** con le impostazioni d'import. Unreal 5.8.3 lo
importa con Interchange, da riga di comando, senza ritocchi.

## Cosa ho provato (15/10/2026, su questo PC)

Tutte prove usa e getta, fuori dal codice consegnato: script in una cartella temporanea, progetto Unreal vuoto in
`output/unreal-probe/` (ignorata da Git). Asset già presenti in `output/`, nessuna generazione.

| Strumento | Prova | Esito |
|---|---|---|
| **Send to Unreal 2.4.3** (ultima release, novembre 2023) con Blender 5.2.2 | Abilitato headless con `--factory-startup`, modalità "Send to Disk", su un prop e sul guerriero rigato | **Non usabile così com'è.** Per il prop servono 3 aggiramenti: percorso Windows sotto i 260 caratteri, coda delle operazioni creata a mano (headless non c'è), `lod_regex` predefinita (`(?i)…`) rifiutata da Python 3.11+. Il personaggio animato si rompe nell'FBX: Send to Unreal sostituisce funzioni interne dell'esportatore FBX di Blender 3.x e in 5.2 manca `add_keyframe`. L'FBX del prop che è uscito **Unreal 5.8 non riesce ad aprirlo**. Il codice non cambia dal 2024; l'attività recente del repository è documentazione e dipendenze del sito |
| **Interchange glTF** in UE 5.8.3, `UnrealEditor-Cmd -run=pythonscript` | Import di prop, guerriero animato e varianti | **Funziona, 1–2 minuti a giro.** Metri → centimetri e Y-su → Z-su da solo (prop: 15 × 32 × 17,5 cm come in Blender). Crea Skeleton, PhysicsAsset, AnimSequence (4 s) e un'istanza di materiale glTF con le texture al loro posto (colore sRGB, normal come normal map, rugosità/metallo lineare) |
| Interchange, LOD per nome | Nodi `LOD0_SM_Prop01`, `LOD1_…`, `LOD2_…` in un GLB | **Un solo Static Mesh con 3 LOD** (4.648 / 2.499 / 997 triangoli) |
| Interchange, collisione per nome | Nodo `UBX_SM_Ubx` accanto a `SM_Ubx` | **1 box di collisione**, nessun mesh in più. Senza nodo `UBX_`/`UCX_` Interchange genera comunque un convesso. `UCX_` importato senza errori, ma l'API Python non espone i vertici del convesso: da verificare con un altro metodo |
| Interchange, assi | Tre cubi spostati verso −Y, +X, +Z di Blender | Blender (x, y, z) → Unreal (x, −y, z). Il fronte glTF (−Y in Blender) diventa **+Y in Unreal, il verso di Manny** |
| Interchange, compressione | `prop_01_LOD0.web.glb` scritto da gltfpack | **Rifiutato**: `EXT_meshopt_compression` ed `EXT_texture_webp` non supportati |
| FBX nativo di Blender 5.2 in Interchange | Prop e guerriero, texture incorporate | Mesh e scheletro sì, ma **texture non importate** (payload non valido), animazione rifiutata (lunghezza non allineata ai fotogrammi), avviso sui gruppi di smoothing. Peggio del glTF su tutto |
| **IK Rig** di UE 5.8 (riconoscimento automatico dello scheletro) | `apply_auto_generated_retarget_definition()` su Manny e sul guerriero | Ossa `bone_0…bone_33` di SkinTokens: **non riconosciuto**. Stesse ossa rinominate come Manny (`root`, `pelvis`, `spine_01`, `thigh_l`…, usando il mappatore già in `scripts/kimodo_retarget.py`): **riconosciuto**, radice `pelvis`. Il retarget di `MM_Idle` da Manny al guerriero gira senza errori; il risultato non è ancora stato guardato |
| LOD degli scheletrici in Unreal | `SkeletalMeshEditorSubsystem.regenerate_lod(mesh, 3)` headless | **Funziona**: 10.949 → 3.294 → 1.808 vertici |
| **gltfpack 1.3** (già in `vendor/`) | `-noq -si 0.5 -sv` sul prop e sul guerriero | Esce glTF senza estensioni, pelle e animazione intatte; ma sul guerriero riduce solo del 13% (8.984 → 7.806 triangoli): le cuciture UV lo fermano, come già misurato in `finish_props.py` |
| **glTF-Transform 4.5.1** (npm, MIT) | `validate` e `inspect` sui GLB di prova | Funziona; `validate` usa il validatore Khronos. Utile come controllo prima di Unreal, ma richiede Node |

Altre cose emerse:
- Il guerriero esce da SkinTokens **alto 90 cm e con l'origine al centro** (piedi a −45 cm in Unreal). La misura vera
  non si può dedurre: il profilo deve riceverla (altezza o lato più lungo in cm).
- I GLB animati contengono ancora l'icosfera di servizio di SkinTokens (`Icosphere`, 2 m): `kimodo_retarget.py` e
  `clean_skin_weights.py` la tolgono già, il profilo farà lo stesso.
- Nel progetto vuoto Interchange attiva Nanite sugli Static Mesh: è un'impostazione del progetto, non del file.

## Cosa si riusa

- **Interchange glTF di Unreal** come unica strada d'import: niente plugin, niente editor aperto.
- **Le convenzioni di Send to Unreal**, non il suo codice: prefissi `SM_`/`SK_`, LOD per nome, collisione `UBX_`/`UCX_`,
  osso `root` all'origine.
- **Esportatore glTF di Blender 5.2** e il Blender headless che AssetFurnace già trova (`image_to_3dlab/blender.py`).
- **`map_skintokens_to_soma`** di `kimodo_retarget.py` per riconoscere bacino, colonna, braccia, gambe e dita.
- **LOD cotti da `finish_props.py`** per gli oggetti, quando ci sono (meglio di qualunque semplificazione, già misurato).
- **`blender_bake_ao.py`** per il canale R (occlusione) dell'ORM, quando serve cuocerla.
- **Riduzione LOD di Unreal** per gli scheletrici, dallo script d'import.
- **glTF-Transform `validate`**, solo se Node è installato (controllo facoltativo; la funzione 7 lo può condividere).

## Cosa si costruisce

1. `image_to_3dlab/unreal_profile.py`, Python puro e testabile:
   - nomi: `SM_`/`SK_` per i mesh, `T_<Nome>_BaseColor|N|ORM` per le texture, `MI_<Nome>` per i materiali
     (Interchange crea **istanze** di materiale, che in Unreal si chiamano `MI_`; i `M_` master stanno nel progetto
     del gioco);
   - mappa ossa SOMA → Manny e piano di rinomina (ossa extra: `<osso_padre>_extraN`), `root` aggiunto;
   - fattore di scala dalla misura richiesta, e correzione delle traslazioni nelle animazioni quando si scala;
   - impacchettamento ORM: R = occlusione (o bianco), G = rugosità, B = metallo, cioè lo stesso ordine del glTF:
     una sola immagine usata sia come occlusione sia come metallo-rugosità diventa un solo `T_<Nome>_ORM`;
   - scheda `<Nome>.unreal.json` (tipo, misura, LOD, collisione, scheletro, impostazioni d'import) e copia della
     provenienza con le licenze di origine;
   - controllo dell'uscita: nessuna estensione richiesta, nomi giusti, nodi di collisione ben formati.
2. `scripts/blender_export_unreal.py`, solo le chiamate `bpy`: toglie l'icosfera, scala, mette l'origine ai piedi
   (personaggi) o al centro della base (oggetti), gira il personaggio verso −Y di Blender (fronte ricavato dai piedi,
   come fa `kimodo_retarget.py`), rinomina ossa, oggetti, materiali e immagini, aggiunge i LOD come nodi `LODn_`, la
   collisione `UBX_` (predefinita) o `UCX_` (convesso con pochi vertici), esporta il GLB **senza compressione**.
3. `scripts/export_unreal.py`, la riga di comando, e la funzione `export_unreal(...)` che la sessione della funzione 1
   aggancia come passo `export_unreal`. Argomenti: `--kind prop|creature|equipment`, `--name`, `--height-cm` o
   `--size-cm`, `--lods`, `--collision box|convex|unreal`, `--skeleton-from` (per l'equipaggiamento: lo scheletro del
   corpo), `--out`.
4. `scripts/unreal_import_check.py`, che gira dentro Unreal: importa con Interchange seguendo la scheda (LOD degli
   scheletrici, equipaggiamento sullo Skeleton già esistente) e scrive un rapporto con le misure di accettazione.
   È generico; il gioco può copiarlo o chiamarlo.
5. Test prima del codice per tutte le parti pure (nomi, mappa ossa, scala, ORM, scheda, controlli, argomenti).

**Maschere di tinta** (funzione 8): il glTF non ha un posto standard per una terza texture, e Interchange importa solo
le immagini usate. Il profilo le porta accanto al GLB come `T_<Nome>_Tint.png`, elencate nella scheda; lo script
d'import le importa come texture lineari. Il nome del file va concordato con la sessione della funzione 8.

## Alternative scartate

- **Send to Unreal**, anche con patch: si rompe a ogni versione di Blender perché sostituisce funzioni interne
  dell'FBX, ha bisogno dell'editor aperto per importare, e il suo FBX non si apre in UE 5.8.
- **FBX come formato d'uscita**: con Interchange 5.8 perde le texture e rifiuta l'animazione. Si può riprendere più
  avanti se servisse, non ora.
- **Compressione con gltfpack o glTF-Transform**: Interchange non legge meshopt né WebP. Unreal comprime comunque per
  conto suo.
- **gltfpack per i LOD**: riduce poco dove ci sono cuciture e, forzato, macchia le texture (misurato in
  `finish_props.py`).
- **Rifare lo scheletro sulla gerarchia esatta di Manny**: pesante e con perdite; basta rinominare perché Unreal lo
  riconosca e l'IK Retargeter si occupa delle proporzioni.

## Rischi

- **Animazioni scalate**: scalando l'armatura le traslazioni delle chiavi (il bacino che sale e scende) non seguono da
  sole. Va corretto e testato.
- **Creature non umanoidi**: il mappatore accetta solo umanoidi. Per gli altri si tengono i nomi originali puliti, senza
  compatibilità con Manny (è il lavoro della funzione 6); il fronte si dà con un argomento.
- **Qualità del retarget**: provato solo che gira. Va guardato una volta nell'editor; da riga di comando si può al più
  misurare (per esempio l'altezza delle mani rispetto a Manny).
- **Mani a due dita** di SkinTokens: rinominate come medio e indice; il riconoscimento è provato su un solo guerriero.
- **Collisione `UCX_`** via glTF non verificata fino in fondo; la `UBX_` sì, e resta la predefinita.
- **Percorsi lunghi su Windows**: Send to Unreal è caduto proprio su questo; i nomi d'uscita vanno tenuti corti.
- **Equipaggiamento di prova**: in `output/` non c'è un pezzo d'armatura vero. Propongo di ricavarlo dal guerriero
  (le facce pesate sulla testa o su una spalla, con lo stesso scheletro), che è proprio il caso della funzione 4. Se il
  coordinatore preferisce un pezzo generato, va chiesto e generato da chi ha la GPU.
- **Misura vera**: va data in ingresso. La specifica di lotto della funzione 1 deve avere una colonna per l'altezza o
  il lato più lungo in cm.

## Come provo l'accettazione

1. Tre asset già in `output/`, nessuna generazione:
   - **oggetto**: `prop_01` del foglio "testuggine romana", con i suoi 3 LOD cotti;
   - **creatura**: `SK_BrennholdWarrior_9k` rigato da SkinTokens, con l'animazione idle;
   - **equipaggiamento**: un pezzo ricavato dal guerriero, sullo stesso scheletro.
2. `scripts/export_unreal.py` su ciascuno: GLB, scheda, provenienza.
3. `UnrealEditor-Cmd` sul progetto vuoto in `output/unreal-probe/` con `scripts/unreal_import_check.py`, nessun passo a
   mano. Il rapporto deve dire:
   - misure uguali a quelle richieste (±1%), base a z = 0 (±0,5 cm);
   - personaggio rivolto verso +Y, come Manny;
   - nomi con i prefissi giusti;
   - oggetto con 3 LOD e 1 box di collisione; personaggio con almeno 3 LOD;
   - una sola texture ORM lineare per materiale;
   - scheletro riconosciuto dall'IK Rig e retarget di `MM_Idle` riuscito;
   - equipaggiamento importato sullo Skeleton del guerriero, senza crearne uno nuovo.
4. `PYTHONPATH=. pytest -q` e `ruff check .` verdi; il risultato e cosa si è riusato vanno nel piano generale.

## Risultato (15/10/2026)

**Accettazione superata su UE 5.8.3**, da riga di comando, in un progetto vuoto, senza passi a mano
(`scripts/unreal_import_check.py`, circa due minuti e mezzo). Prova finale con tre asset fatti apposta dalla catena vera
(Qwen-Image → Pixal3D → finish → SkinTokens), in `output/compare-20261015/`:

| Asset | Esito in Unreal |
|---|---|
| Cavaliere `SK_Knight` (40k facce, 34 ossa) | 185,0 cm, piedi e `root` a z = 0, rivolto verso +Y come Manny, 3 LOD, scheletro riconosciuto dall'IK Rig. Idle di Manny: braccia giù a 0,305 dell'altezza (Manny 0,326). Camminata di Manny: passo largo 0,701 (Manny 0,666) |
| Lupo mannaro `SK_Werewolf` (30k facce, 46 ossa) | 220,0 cm, piedi e `root` a z = 0, verso +Y, 3 LOD, scheletro riconosciuto. Idle: 0,271 (Manny 0,326). Camminata: 0,687 (Manny 0,666) |
| Forziere `SM_Chest` (8k facce) | 80,0 cm sul lato più lungo, base a z = 0, 1 box di collisione, ORM lineare |
| Equipaggiamento `SK_WarriorHelm` | testa e spalle tagliate dal guerriero di prova: sullo Skeleton del guerriero, nessuno Skeleton nuovo, indossato con Leader Pose |

Le immagini (in ogni cartella d'uscita): `unreal_check.png` con Manny, i personaggi in idle e l'oggetto alla luce del
giorno, `unreal_check_walk.png` a metà camminata, `unreal_check_close.png` in primo piano per ogni asset.

Un difetto trovato solo dalla camminata: il profilo aggiungeva l'osso `root` all'origine del file sorgente, che per
SkinTokens è a metà corpo; dopo averlo messo a terra il `root` restava a 93 cm. L'idle non lo mostra, ma le camminate di
Manny bloccano il root motion sulla posa di riferimento e Unreal alzava il personaggio di tanto. Ora il `root` si aggiunge
dopo aver messo il modello a terra; un test con Blender vero e un controllo in Unreal lo tengono lì.

Cose imparate strada facendo, che valgono anche per l'import nel gioco:
- le immagini si prendono dal viewport dell'editor in Simulate con `-NoTextureStreaming`: una SceneCapture in Simulate
  rende una volta sola e parte con l'esposizione scura, e senza quel flag le texture restano alle versioni ridotte;
- cambiare animazione a sessione Simulate avviata lascia la posa di riferimento: per la camminata si chiude la sessione,
  si imposta l'animazione sugli attori e se ne apre un'altra;
- le misure del retarget si leggono direttamente dalle animazioni con `AnimPoseExtensions`, senza rendering;
- in modalità commandlet il registro degli asset non ha ancora visto il contenuto: gli asset di una prova precedente non
  si sostituiscono, per cui lo script svuota da disco la cartella di ogni asset prima di avviare Unreal;
- un IK Retargeter creato da Python in UE 5.8 **non ha operazioni** finché non si chiama `add_default_ops()`: copia solo
  il bacino e il personaggio resta in T-pose. Serve anche `auto_align_all_bones()` (SkinTokens è in T-pose, Manny in A-pose);
- Interchange chiama le animazioni glTF `<mesh><animazione>`: lo script d'import le rinomina `AS_…`;
- l'override della pipeline per riusare uno Skeleton passa da `InterchangeManager.import_asset` con una copia di
  `DefaultGLTFAssetsPipeline` più `DefaultGLTFPipeline`; su `AssetImportTask.options` viene ignorato;
- nel mondo dell'editor le animazioni non avanzano: per vederle serve Simulate, e il livello del template usa World
  Partition, quindi gli attori aggiunti vanno resi "sempre caricati";
- in Python `unreal.Rotator(a, b, c)` è (roll, pitch, yaw): meglio sempre gli argomenti per nome.

Con la funzione 8: il profilo prende da solo `T_<asset>_Tint.png` e `<asset>.materials.json` lasciati accanto al
modello, li porta nella cartella d'uscita e lo script d'import mette la maschera in Unreal come dato lineare (provato
sul guerriero passato dalla funzione 8: tutti i controlli verdi, maschera compresa). Il tipo accetta anche i nomi
`object` ed `equipment_part` usati dai lotti e dai controlli.

Nello studio: pannello **Export for Unreal** su un asset della Libreria (tipo, misura vera, corpo per l'equipaggiamento,
mosse, verifica facoltativa in Unreal), con i file da scaricare, lo zip, "Show folder" in locale e, se richiesto, i
controlli e l'immagine di Unreal.

## Cosa resta

- **Occlusione**: il canale R dell'ORM viene dalla mappa di occlusione del modello quando c'è (la funzione 8 la mette
  quando ha un'AO); senza, è bianco. Cuocerla con `scripts/blender_bake_ao.py` dentro il profilo è il passo successivo (va fatta per
  ogni LOD, ognuno ha il suo atlante).
- **Collisione `UCX_`** (convessa) scritta ma non verificata in Unreal: l'API Python non espone i vertici del convesso.
  La `UBX_` predefinita è verificata.
- **Creature non umanoidi**: il codice tiene i nomi originali e accetta il fronte come argomento, ma non c'è ancora una
  creatura non umanoide rigata da provare (funzione 6).
- **Equipaggiamento rigido** (armi su un socket): oggi si esporta come oggetto `SM_`; l'aggancio al socket è lato Unreal.
- **Nanite**: Interchange lo accende sugli Static Mesh del progetto vuoto. È una scelta del progetto del gioco, non del file.
- Su Windows 5 test della repo falliscono già su `main` (file letti senza `encoding="utf-8"`), e `ruff check .` segnala
  errori preesistenti fuori da questi file (soprattutto `hunyuan_mlx/`). I file di questa funzione sono puliti.

## Seguiti

Punti della review di Codex rimandati perché non danno un risultato sbagliato, dati persi o un crash:

- **Unreal fuori da Windows**: `scripts/unreal_import_check.py` cerca `UnrealEditor-Cmd.exe` in `Win64`. Su macOS e
  Linux il controllo si ferma con un errore, non dà un esito falso. Va aggiunto il nome dell'eseguibile per ogni
  piattaforma quando servirà ([PR #6](https://github.com/matteodante/AssetFurnace/pull/6#discussion_r4287542338)).
- **"Show folder" su Linux senza desktop**: lo studio lo offre anche quando non c'è `DISPLAY`, e il pulsante non apre
  nulla. I file restano scaricabili; va usato lo stesso controllo di `blender_edit_api.desktop_reason`
  ([PR #7](https://github.com/matteodante/AssetFurnace/pull/7#discussion_r4287526165)).
- **Zip in memoria**: lo studio costruisce lo zip dell'esportazione in memoria prima di inviarlo. Con file da
  centinaia di MB occupa memoria per tanto; va scritto su un file temporaneo e inviato a pezzi
  ([PR #7](https://github.com/matteodante/AssetFurnace/pull/7#discussion_r4287894329)).

Dall'utente, per la funzione 6 (famiglie di creature), dopo la prova in Unreal del 15/10/2026:

- **Lupo mannaro, gambe digitigrade**: con le animazioni umane di Manny le gambe si incrociano in modo innaturale in
  camminata. Serve un retarget o animazioni pensate per gambe digitigrade, non solo il profilo.
- **Lupo mannaro, albedo**: esce chiaro e poco contrastato già dalla generazione; in Unreal si vede uguale. È un
  problema del concept o della pittura, non dell'esportazione.
