# Pipeline di produzione per un gioco intero (primo utente: Auroconts)

Data: 15 ottobre 2026. Stato: **proposta**, niente è implementato.

Auroconts è un MMORPG su Unreal Engine 5.8 che rifà un gioco del 2012 con asset nuovi: stile stilizzato, sagome
riconoscibili rispetto all'originale, budget da PC medi. Il suo censimento conta 592 modelli di mostri, 866 di PNG, 427
forme di armatura (per 3 razze × 2 sessi), 413 di armi, 3.926 oggetti di mappa, circa 2.250 icone. Non si fanno uno alla
volta nello studio: serve una catena guidata da agenti, a lotti, con controlli uguali per tutti.

Regola dell'utente: **incrementale sì, semplificato no**. Si produce per frequenza d'uso, alla qualità finale; i
moltiplicatori (varianti di colore, riadattamento ai corpi) riproducono la varietà, non la tolgono.

Le funzioni qui sotto sono generiche: servono a qualunque gioco. Ciò che è specifico di Auroconts (lettori dei file
originali, tabella di corrispondenza, import in Unreal) resta nella repo di Auroconts (`docs/assets/pipeline.md`,
`tools/`).

## Le funzioni proposte

### 1. Lotti e accesso per agenti

- **Cosa**: una specifica di lotto (JSON o CSV) con una riga per asset: id esterno, immagini di riferimento, tipo
  (oggetto, creatura, parte di equipaggiamento), budget di facce, profilo d'uscita. `pipeline.py --run-batch lotto.json`
  la esegue riga per riga.
- **Comportamento**: riprende dall'ultima riga riuscita; stato per riga (in coda, fatto, fallito con motivo); un
  `.provenance.json` per ogni uscita come oggi; nessun download di pesi senza conferma (regola della repo).
- **Poi**: un server MCP con le stesse operazioni (inviare un lotto, leggere lo stato, ritirare i file), così un agente
  in un'altra repo guida tutto.
- **Accettazione**: un lotto di 10 prop gira senza interventi; interrotto a metà, riparte dalla riga giusta; le uscite
  coincidono con quelle di 10 esecuzioni singole con gli stessi parametri.

### 2. Dal riferimento al concept

- **Cosa**: generazione d'immagine guidata da un riferimento (sagoma, bordi o profondità di un render) con una palette
  imposta, per ottenere un oggetto **nuovo** che si riconosce dalla forma.
- **Accettazione**: su 5 riferimenti, la somiglianza delle sagome sta in una fascia configurabile (simile, non
  identica) e la palette resta entro una distanza data.
- **Rischio**: dipende dal controllo che offre il modello d'immagine locale (Qwen-Image). Prima una prova.

### 3. Bibbia dello stile

- **Cosa**: un insieme di immagini approvate, modelli di prompt per tipo di asset e un punteggio di coerenza (distanza
  di un nuovo concept dall'insieme approvato), mostrato nello studio e usabile come soglia nei lotti.
- **Poi, dopo una prova**: un LoRA dello stile per il modello d'immagine.
- **Accettazione**: il punteggio separa i concept che l'utente approva da quelli che rimanda, su un campione etichettato.

**Quanto costa un LoRA dello stile** (rimandato dall'utente il 15/10/2026). Non tanto in soldi, quanto in preparazione:
- Per Qwen-Image servono di base 24 GB di VRAM (le configurazioni di ai-toolkit sono pensate per 24 GB); la RTX 5070 Ti
  ne ha 16, quindi in locale solo con scaricamento aggressivo in RAM, non garantito.
- In alternativa una GPU a noleggio da 24–48 GB per qualche ora a corsa: probabilmente pochi euro a tentativo, da
  verificare con un preventivo prima di spendere.
- Il costo vero è il set di immagini: 20–50 concept già approvati dall'utente, più qualche giro di prova.
- È opzionale: prima si prova senza (prompt fissi, Qwen-Image-Edit-2511 con immagini di riferimento, ControlNet). Il LoRA
  si fa solo se il pilota mostra che lo stile deriva.

### 4. Parti per slot e adattamento a un corpo base

Il "Part Composer" per il caso vincolato: corpi base fissi, slot fissi, uno scheletro.

1. Dal concept di un set, una tavola per slot (elmo, busto, spalle, guanti, gambe, stivali, arma) e le viste che servono.
2. Un modello 3D per parte (Pixal3D), con budget di facce per slot.
3. **Adattamento in Blender**, headless e testabile:
   - allineamento della parte alla zona del suo osso sul corpo base;
   - shrinkwrap con spessore;
   - controllo delle compenetrazioni in pose di prova;
   - pesi copiati dal corpo;
   - maschera della geometria coperta del corpo.
4. **Riadattamento** dello stesso pezzo agli altri corpi base (per Auroconts 3 razze × 2 sessi).
5. Oggetti rigidi (armi, elmi rigidi) pesati al 100% su un osso.

- **Alternative per dividere in parti** una mesh già generata, tutte con licenza MIT: PartCrafter, OmniPart, HoloPart.
  Da evitare Hunyuan3D-Part (licenza Tencent, esclude l'UE).
- **Accettazione**: un set completo su 6 corpi, senza ritocchi a mano, passa i controlli della funzione 7 in corsa,
  attacco e una posa estrema.
- **Rischio**: medio. Nei video recenti anche lo strumento migliore (AssetHub Part Composer) lascia ritocchi a mano;
  qui si scommette che i vincoli rendano l'adattamento deterministico. Si prova su un solo set prima di andare avanti.

### Quale modello per le parti (confronto del 15/10/2026)

I tre non sono alternative uguali: fanno lavori diversi.

| Modello | Entra | Esce | Controllo sulle parti | Per noi |
|---|---|---|---|---|
| **OmniPart** (MIT, SIGGRAPH Asia 2025) | Un'immagine **più una maschera di segmentazione 2D** | Parti 3D generate con un modello derivato da TRELLIS | **Sì**: le parti sono quelle disegnate nella maschera | Il più adatto: la maschera può essere quella degli slot (elmo, busto, guanti…) |
| **HoloPart** (MIT) | Una **mesh già fatta** con la sua segmentazione (per esempio da SAMPart3D o SAMesh) | Le stesse parti, **completate** anche dove erano nascoste | Sì, dalla segmentazione | Complemento: si genera il personaggio intero con Pixal3D, lo si divide per slot e HoloPart completa i pezzi |
| **PartCrafter** (MIT, NeurIPS 2025) | Un'immagine | Fino a 16 parti in un colpo | Poco: decide il modello come dividere | Il meno adatto per l'equipaggiamento; forse per gruppi di oggetti |

Nel confronto pubblicato con CubePart (dati degli autori, non verificati da noi) OmniPart con maschere SAM3 batte
PartCrafter e PartPacker. **Nessuno dei tre è provato su questo PC**: requisiti e Windows vanno verificati (PartCrafter ha
una guida per Windows in un fork; OmniPart e HoloPart sono pensati per Linux con CUDA).

Proposta: nella versione minima della funzione 4 si mettono a confronto, sullo stesso set di armatura, tre strade:
1. **OmniPart** con le maschere degli slot;
2. **Pixal3D intero + segmentazione per slot + HoloPart**;
3. **una immagine per parte + Pixal3D**, come fa AssetHub.

Vince quella che dopo l'adattamento al corpo dà la qualità migliore con meno ritocchi.

### 5. Profilo d'uscita per Unreal

- **Cosa**: un profilo di esportazione che produce file pronti per Unreal:
  - scheletro compatibile con quello del progetto, anche dopo il rig automatico (retarget di SkinTokens);
  - centimetri, assi, origini;
  - nomi `SK_` / `SM_` / `T_` / `M_`;
  - LOD;
  - collisione semplice per gli oggetti;
  - texture ORM (occlusione, rugosità, metallo);
  - **maschere di tinta** per le varianti di colore;
  - FBX o glTF.
- **Accettazione**: un oggetto, una creatura e una parte di equipaggiamento entrano in Unreal 5.8 con Interchange senza
  ritocchi, con scala e assi giusti.
- **Stato (15/10/2026): fatto**, superata l'accettazione su UE 5.8.3 (dettagli e misure in
  [2026-10-15-unreal-profile-design.md](2026-10-15-unreal-profile-design.md)). Si usa con `scripts/export_unreal.py`, con
  il pulsante "Export for Unreal" nello studio e, dopo il merge, con il passo `export_unreal` dei lotti; il controllo in
  Unreal è `scripts/unreal_import_check.py`.

### 6. Famiglie di creature

- **Cosa**: oltre ai quadrupedi già presenti, modelli di scheletro e rig per ragni, draghi e volanti, serpenti, piante,
  con pesi e pose di riferimento come per i quadrupedi.
- **Accettazione**: una creatura per famiglia rigata e animata con una clip di prova senza deformazioni visibili.

### 7. Controlli e coda di revisione

- **Controlli automatici**, uguali per tutti:
  - sagoma rispetto al riferimento (dentro la fascia);
  - misure;
  - facce, materiali, risoluzione e densità delle texture;
  - pulizia (facce invertite, buchi, parti volanti, UV);
  - compenetrazioni nelle pose di prova;
  - licenze dei componenti.
- **Revisione**: girata a 360°, confronto con il riferimento e una coda nello studio dove l'utente approva o rimanda con
  una nota. L'esito resta nel registro dell'asset.
- **Accettazione**: un asset difettoso costruito apposta viene fermato dal controllo giusto; un rimando torna in coda
  con la sua nota.

### 8. Materiali pronti per il gioco

- **Cosa**: per ogni asset generato:
  - una **maschera di tinta** a 3 canali (zone ricolorabili), ricavata dalle regioni del modello;
  - un controllo della **luce cotta** nell'albedo (luminosità e ombre troppo marcate), che rimanda indietro invece di
    correggere;
  - mappe normal e di rugosità ricavate dal colore dove mancano (approccio di Materialize, senza AI);
  - texture ORM impacchettate.
- **Perché**: un gioco intero usa pochi materiali master e migliaia di istanze; senza maschere di tinta e senza albedo
  pulito, gli asset generati non si ricolorano e non reggono la luce dinamica.
- **Accettazione**: un mostro esce con 3 zone ricolorabili che in Unreal cambiano colore da parametri, senza ombre cotte
  visibili sotto una luce diversa da quella del concept.

## Ruote già fatte: cosa si riusa e cosa si costruisce (verifica del 15/10/2026)

Prima di scrivere codice si cerca chi lo fa già bene. Stato e licenze verificati su GitHub il 15/10/2026.

| # | Proposta | Si riusa | Si costruisce | Note |
|---|---|---|---|---|
| 1 | Lotti e MCP | Manifest e CLI di AssetFurnace; SDK MCP ufficiale | Il lettore dei lotti, la ripresa e il server MCP: poco codice sopra ciò che c'è | ComfyUI (con [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack), MIT, fermo da dicembre 2025) sarebbe un'alternativa come orchestratore, ma ci farebbe perdere provenienza e regole di AssetFurnace |
| 2 | Dal riferimento al concept | **Qwen-Image 2.1 Fun ControlNet Union** (Alibaba PAI, settembre 2026: canny, profondità, lineart, posa e altri 4 controlli in un file) oppure [InstantX Qwen-Image ControlNet Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) | L'aggancio in AssetFurnace, che usa già Qwen-Image 2.1 | Licenze dei pesi da verificare; supporto ComfyUI della 2.1 ancora in revisione a fine settembre |
| 3 | Bibbia dello stile | [ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT, molto attivo) per un LoRA dello stile; Qwen-Image-Edit-2511 per la coerenza fra immagini | Il punteggio di coerenza (distanza fra embedding) e la sua soglia nei lotti | musubi-tuner non dichiara una licenza: da evitare |
| 4 | Parti e adattamento al corpo | **Generazione**: OmniPart, HoloPart, PartCrafter (MIT). **Segmentazione di una mesh intera**: [SAMPart3D](https://github.com/Pointcept/SAMPart3D) (MIT); PartField ha licenza da verificare. **Adattamento**: la tecnica standard con gli strumenti di Blender: gabbia low-poly con Shrinkwrap e piccolo scostamento, armatura spinta con Surface Deform o Mesh Deform, pesi copiati dal corpo (Data Transfer), massimo 4 influenze | Lo script che applica la tecnica in automatico e i suoi test | Non c'è un adattatore open source affidabile: i componenti aggiuntivi seri di Blender sono a pagamento o proprietari (RadFit, FitClick). L'Outfit Asset di Unreal è pensato per la stoffa simulata, non per gusci rigidi |
| 5 | Profilo Unreal | **Usato**: Interchange glTF di Unreal (unica strada d'import); l'esportatore glTF di Blender 5.2; il mappatore di ossa di `kimodo_retarget.py`; i LOD già cotti da `finish_props.py`; IK Rig e Retargeter di Unreal (con `add_default_ops`) per Manny. Di Send to Unreal solo le **convenzioni** (prefissi, `LODn_`, `UBX_`/`UCX_`, osso `root`). **Non usato**: il codice di Send to Unreal (fermo al 2023, con Blender 5.2 si rompe in tre punti e il suo FBX non si apre in UE 5.8); FBX (Interchange perde texture e animazioni); compressione di gltfpack e glTF-Transform (Interchange rifiuta meshopt e WebP); gltfpack per i LOD (riduce poco dove ci sono cuciture UV) | Il profilo (`image_to_3dlab/unreal_profile.py`, `scripts/export_unreal.py`, `scripts/blender_export_unreal.py`), il controllo in Unreal (`scripts/unreal_import_check.py`) e il pannello nello studio | Fatto il 15/10/2026: accettazione superata su UE 5.8.3 |
| 6 | Famiglie di creature | **SkinTokens**, già in AssetFurnace, è il successore di [UniRig](https://github.com/VAST-AI-Research/UniRig) (MIT), che rigga scheletri diversi, animali compresi; [Puppeteer](https://github.com/Seed3D/Puppeteer) (Apache-2.0) rigga e anima forme qualsiasi | Prima si provano su ragno, drago e pianta; i modelli Rigify per famiglia solo dove falliscono | — |
| 7 | Controlli e revisione | [glTF-Validator](https://github.com/KhronosGroup/glTF-Validator) (Apache-2.0); analisi delle mesh di Blender; statistiche di meshoptimizer; Data Validation di Unreal | I controlli specifici (sagoma, misure, compenetrazioni) e la coda nello studio | — |

### Lato Unreal (non in AssetFurnace)

- **Le parti seguono il corpo** con Set Leader Pose Component: un solo AnimBP.
- **Il corpo sotto l'armatura si nasconde con maschere**, non tagliando la mesh per ogni pezzo. Si cuoce una maschera
  (colori dei vertici, o vertici entro pochi millimetri dal guscio) e il materiale del corpo usa la maschera di
  opacità dei pezzi indossati.
- **Più corpi**: due mesh adattate per lo stesso scheletro, oppure **Mutable** (beta), con parametro del tipo di corpo,
  morph fra gli adattamenti e visibilità per sezione. In 5.8 c'è una segnalazione di artefatti quando si combina Clip
  Mesh With Mesh con il reshape: da provare prima di affidarcisi.
- Le correzioni rimaste si fanno con correttivi di posa solo sugli attacchi peggiori.

Fonte principale della tecnica di adattamento: la discussione
[Fitting armour/clothing to MetaHumans in UE 5.8.3](https://forums.unrealengine.com/t/fitting-armour-clothing-to-metahumans-in-ue-5-8-3-best-workflow-to-avoid-hours-of-manual-fitting-per-set/2839852)
sul forum di Epic (una risposta esperta, non una verifica ufficiale).

## Ordine

**Decisione dell'utente del 15/10/2026**: prima si sviluppano in AssetFurnace le funzioni di base (1, 5, 7, 8), poi si prova. Il **LoRA dello stile è rimandato**: si lavora con prompt fissi, immagini di riferimento e ControlNet.

1. Fondamenta, sviluppate prima di qualsiasi prova: 1, 5, 7, 8.
2. La 4 parte da una prova rapida con script usa e getta su un solo set di armatura; la funzione vera si costruisce solo sulla strada che vince.
3. Stile: 2 e 3.
4. 6, quando servono le creature non umanoidi.

## Quanto sono maturi gli strumenti proposti (verifica del 15/10/2026)

| Strumento | Maturità | Note |
|---|---|---|
| Unreal 5.8.3: GAS, Common UI, StateTree, PCG, World Partition, Niagara, Landscape, PBR | Stabile | Stato letto nei `.uplugin` del motore installato |
| Game Features, Data Registry, Iris, Motion Warping, Mutable | Beta | Mutable ha una segnalazione di artefatti in 5.8 con il clip combinato al reshape |
| Water, plugin MCP e toolset, toon di Substrate | Sperimentale | Si usano con cautela, il toon non si usa |
| MegaLights / Lumen Lite | Stabile in 5.8 / beta in 5.8 | — |
| Blender (Shrinkwrap, Surface Deform, Data Transfer) | Stabile, standard da anni | Base della tecnica di adattamento |
| glTF-Transform, meshoptimizer, glTF-Validator | Stabili, attivi a ottobre 2026 | — |
| ai-toolkit | Attivo (ottobre 2026) | — |
| Nakama Unreal SDK, Flow Graph | Attivi, con versioni per 5.8 | — |
| Send to Unreal | Mantenuto dalla comunità, attivo a settembre 2026 | Supporto a Blender 5 e UE 5.8 non dichiarato: da provare |
| UniRig / SkinTokens | Attivi (giugno 2026) | SkinTokens è già in AssetFurnace |
| Qwen-Image 2.1 Fun ControlNet Union | Molto recente (settembre 2026) | Supporto ComfyUI in revisione; licenza dei pesi da verificare |
| OmniPart, HoloPart, PartCrafter, SAMPart3D, Puppeteer | **Codice di ricerca** (2025) | Funzionano, ma nessuno li mantiene come prodotto; pensati per Linux con CUDA. Vanno provati prima di dipenderne |
| ComfyUI-3D-Pack | Fermo da dicembre 2025 | Non lo usiamo |
| Strumenti di delighting con AI | Immaturi o con licenze restrittive | Si previene invece di correggere |

In sintesi: la base (Unreal, Blender, formati, ottimizzazione) è matura; la generazione per parti e il controllo dello
stile sono la parte nuova e incerta. Per questo si passa dal pilota prima di produrre in grande.

## Pilota di Auroconts

5 mostri (umanoide, quadrupede, ragno, drago, pianta), 10 oggetti di mappa e un set di armatura su 6 corpi, ognuno fino
alla prova in gioco. Misura tempo per pezzo, percentuale approvata al primo giro e tipi di difetto.

## Vincoli tecnici e di licenza

- PC di sviluppo: Windows, RTX 5070 Ti 16 GB. **Pixal3D** funziona (prova di installazione in 186 s); il pacchetto
  include l'encoder DINOv3, quindi l'output va trattato come "commerciale con condizioni": da verificare prima di
  distribuire.
- TRELLIS.2 e Hunyuan3D-2.1 su NVIDIA sono solo Linux: eventualmente WSL2. Hunyuan resta escluso per la licenza (UE).
- GPU a noleggio solo per lotti grandi e su decisione dell'utente.
- Restano valide le regole della repo: test prima del codice, script solo se generali, nessun download senza conferma,
  provenienza su ogni file.

## Fonti

- Stefan 3D AI, "You Can Now Create Modular 3D Character Without Blender" (14/10/2026, `CFA8Pfw6OIs`, sponsorizzato):
  divisione in parti, generazione e assemblaggio automatico; ritocchi in Blender ancora necessari. Trascrizione in
  `journal/video-research/CFA8Pfw6OIs/`.
- Stefan 3D AI, "I Built a Modular Character with AI (Full Workflow)" (25/03/2026, `gZIxrX1n2D4`): il flusso a mano
  (allineamento con Elastic Grab, geometria coperta tolta, oggetti rigidi su un osso, Leader Pose in Unreal).
- Stefan 3D AI, "Gemini 4 Argon … 3D & Blender" (15/10/2026, `2hqcDHnyJaY`): per Blender guidato da un agente Opus 5.5
  rende meglio di Gemini 4, a un costo più alto.
- Garbaj, "How Character Customization Actually Works In Video Games" (14/11/2023, `rspqfIxRXNM`): blend shape, oggetti
  agganciati alle ossa, colori da shader.
- Modelli per parti: [PartCrafter](https://github.com/wgsxm/PartCrafter) (MIT),
  [OmniPart](https://github.com/HKU-MMLab/OmniPart) (MIT), [HoloPart](https://github.com/VAST-AI-Research/HoloPart)
  (MIT), [PartPacker](https://github.com/NVlabs/PartPacker) e [Roblox cube](https://github.com/Roblox/cube) (licenza da
  verificare), [Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) (da evitare).
