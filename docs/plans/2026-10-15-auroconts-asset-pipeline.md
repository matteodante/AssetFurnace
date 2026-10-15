# Pipeline di produzione per un gioco intero (primo utente: Auroconts)

Data: 15 ottobre 2026, aggiornato il 16 ottobre 2026. Stato: **in sviluppo**. Le funzioni 1, 5 e 7 sono su `main`; 2, 3,
6 e 8 in parte; la 4 è in prova. Lo stato di ogni funzione è nella tabella qui sotto e in [README](README.md).

Auroconts è un MMORPG su Unreal Engine 5.8 che rifà un gioco del 2012 con asset nuovi: stile stilizzato, sagome
riconoscibili rispetto all'originale, budget da PC medi. Il suo censimento conta 592 modelli di mostri, 866 di PNG, 427
forme di armatura (per 3 razze × 2 sessi), 413 di armi, 3.926 oggetti di mappa, circa 2.250 icone. Non si fanno uno alla
volta nello studio: serve una catena guidata da agenti, a lotti, con controlli uguali per tutti.

Regola dell'utente: **incrementale sì, semplificato no**. Si produce per frequenza d'uso, alla qualità finale; i
moltiplicatori (varianti di colore, riadattamento ai corpi) riproducono la varietà, non la tolgono.

Le funzioni qui sotto sono generiche: servono a qualunque gioco. Ciò che è specifico di Auroconts (lettori dei file
originali, tabella di corrispondenza, import in Unreal) resta nella repo di Auroconts (`docs/assets/pipeline.md`,
`tools/`).

## Stato al 16/10/2026

| # | Funzione | Progetto | Stato |
|---|---|---|---|
| 1 | Lotti e accesso per agenti | [batch-mcp](2026-10-15-batch-mcp-design.md) | **Su `main`** (PR #2): lotti, ripresa, server MCP, pagina Batches |
| 2 | Dal riferimento al concept | [reference-concept](2026-10-15-reference-concept-design.md) | Strada decisa: Qwen-Image 2.1 in modifica dentro `sd-cli`. File della visione su `main` (PR #9); prova dei 4 casi in corso, giudicata a occhio dall'utente |
| 3 | Bibbia dello stile | [style-bible](2026-10-15-style-bible-design.md) | PR 1 su `main` (PR #10: bibbia, prompt, riferimenti). PR 2 (gara fra giudici imparati) in corso. Lo stile si ricava dai riferimenti originali del progetto (vedi sotto) |
| 4 | Parti per slot | progetto della prova sul ramo `claude/spike-slot-parts-79e80d` | Prova delle tre strade in corso; rapporto in arrivo |
| 5 | Profilo Unreal | [unreal-profile](2026-10-15-unreal-profile-design.md) | **Su `main`** (PR #6, #7), accettazione superata su UE 5.8.3 |
| 6 | Famiglie di creature | [creature-families](2026-10-15-creature-families-design.md) | PR 1 su `main` (PR #8: riconoscimento, `rig.deformation`, digitigradi e quadrupedi). PR #13 (molte zampe, serpenti) in attesa della ricerca sugli approcci più recenti per l'animazione |
| 7 | Controlli e revisione | [checks-review](2026-10-15-checks-review-design.md) | **Su `main`** (PR #4) |
| 8 | Materiali pronti per il gioco | [v2](2026-10-15-game-materials-v2-design.md) (la [v1](2026-10-15-game-materials-design.md) è superata) | v2 approvata. PR 1 e 2 su `main` (PR #11 ORM e pezzo comune, PR #12 etichette). La PR #3 (v1) resta in bozza e si chiude |

### Regole trasversali decise dall'utente (15-16/10/2026)

- **"O fatto bene o niente"; incrementale sì, semplificato no.**
- **Niente controlli deterministici sui pixel.** Luminosità, contorni (Canny), saturazione, distanze di palette e
  raggruppamenti per colore "non funzionano bene" (parole dell'utente). I controlli di immagine sono modelli imparati
  (SigLIP 2, CSD, DINOv2, giudici visione-linguaggio, segmentatori). **Un controllo entra solo se supera la prova contro
  i giudizi dell'utente** (area ROC ≥ 0,90; almeno l'80% dei difettosi fermati; al massimo il 10% dei buoni). Se non la
  supera, non entra, nemmeno come "informativo". Le misure geometriche sulla mesh restano (facce, buchi, UV,
  allungamenti, compenetrazioni, IoU della sagoma su maschere ritagliate).
- **AssetFurnace è un programma a sé.** I motori stanno in `vendor/` (per le immagini `vendor/sdcpp/sd-cli`), i pesi nel
  catalogo con conferma esplicita. Niente ComfyUI.
- **Prima di proporre uno strumento si verifica che sia l'approccio più recente**, con la data di uscita e la licenza di
  codice e pesi (attenzione ai pesi non commerciali caricati in automatico, come PartField in OmniPart e Puppeteer).
- **Stile del progetto dai suoi riferimenti originali.** Per Auroconts, immagini del gioco originale (schermate e
  render) usate solo come riferimento interno: fuori da git, mai negli asset, con il controllo di copia. La bibbia vive
  nella repo del progetto (`Auroconts/local/style-bible/`), non qui.
- **GPU condivisa** (`%TEMP%\assetfurnace-gpu.lock`, con pid e nome della sessione): si prende solo mentre un lavoro
  calcola davvero, si libera dopo ogni lavoro, e dopo un lavoro di oltre 20 minuti si passa la mano se altri aspettano.
- **Review di Codex**: al massimo un giro per PR, ed è facoltativo. Si correggono solo i P1 veri, il resto va nei
  "Seguiti" del progetto. Poi test verdi in una copia pulita e merge del coordinatore.
- **LoRA dello stile rimandato.** Le immagini approvate della bibbia diventano il suo insieme di addestramento.

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

- **Cosa**: generazione d'immagine guidata da un riferimento (immagine, sagoma, profondità di un render), per ottenere un
  oggetto **nuovo** che si riconosce dalla forma. Serve anche a imporre la T-pose del corpo base, a disegnare i pezzi
  d'armatura sopra il corpo base e a tenere coerenti le viste.
- **Strada decisa (15/10/2026)**: **Qwen-Image 2.1 in modifica** dentro `sd-cli`, con più riferimenti (`-r`
  ripetibile) e il file della visione di Qwen3-VL-8B. Valutate e scartate: Fun ControlNet Union (non c'è in `sd-cli`;
  chiederebbe ComfyUI, fuori dalla struttura) e Qwen-Image-Edit-2511 (Apache-2.0, ma è un altro modello e cambierebbe lo
  stile; resta la riserva se la licenza di ricerca della 2.1 diventa un problema).
- **Accettazione**: quattro casi (mostro riconoscibile ma non copiato, T-pose esatta, pezzo d'armatura sopra il corpo,
  viste coerenti) giudicati a occhio dall'utente. Il segnale di copia, se c'è, viene da un modello imparato (DINOv2).

### 3. Bibbia dello stile

- **Cosa**: una cartella per progetto con le immagini approvate e rifiutate, ciascuna con la sua nota; modelli di prompt
  per tipo di asset; 1-3 immagini della bibbia date come riferimento di stile quando si genera, tenendo il migliore di
  alcuni tentativi; un giudice imparato che dice se un concept è nello stile, prima del 3D.
- **Il giudice**: gara fra SigLIP 2, CSD (embedding di stile) e Qwen3-VL con domande sì/no scritte dall'utente. Vince
  chi supera la prova sui giudizi dell'utente; se nessuno la supera, restano bibbia e riferimenti, senza un controllo
  finto.
- **Primo giro di etichette (16/10/2026)**: 2 approvati su 25. Il prompt di stile generico ("colori saturi") non
  descriveva il gioco; ora la descrizione si ricava dalle immagini originali del progetto.
- **Poi, dopo una prova**: un LoRA dello stile per il modello d'immagine.
- **Accettazione**: il giudice separa i concept che l'utente approva da quelli che rimanda, su un campione etichettato.

**Quanto costa un LoRA dello stile** (rimandato dall'utente il 15/10/2026). Non tanto in soldi, quanto in preparazione:
- Per Qwen-Image servono di base 24 GB di VRAM (le configurazioni di ai-toolkit sono pensate per 24 GB); la RTX 5070 Ti
  ne ha 16, quindi in locale solo con scaricamento aggressivo in RAM, non garantito.
- In alternativa una GPU a noleggio da 24–48 GB per qualche ora a corsa: probabilmente pochi euro a tentativo, da
  verificare con un preventivo prima di spendere.
- Il costo vero è il set di immagini: 20–50 concept già approvati dall'utente, più qualche giro di prova.
- È opzionale: prima si prova senza (prompt fissi e immagini di riferimento in Qwen-Image 2.1). Il LoRA si fa solo se il
  pilota mostra che lo stile deriva anche con i riferimenti.

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

- **Alternative per dividere in parti** una mesh già generata: PartCrafter, OmniPart, HoloPart (codice MIT). Attenzione:
  OmniPart carica i pesi di PartField (NVIDIA, solo uso non commerciale); la prova lo usa alle condizioni scritte nel
  suo progetto. Da evitare Hunyuan3D-Part (licenza Tencent, esclude l'UE).
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
- **Stato (16/10/2026)**: riconoscimento della famiglia dallo scheletro, controllo `rig.deformation` (geometrico) e clip
  procedurali per digitigradi e quadrupedi su `main`. Le clip procedurali provano il rig e non sono animazioni da gioco:
  per le animazioni vere dei mostri (attesa, camminata, corsa, attacco, colpo, morte) è in corso una ricerca sugli
  approcci più recenti.

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

- **Cosa** (progetto v2, approvato il 16/10/2026; la v1 per colore e luminosità è superata): per ogni asset generato:
  - una **maschera di tinta** a 3 canali per **materiale**, non per colore: viste del modello in Blender, SAM 3 guidato
    dai nomi dei materiali dati da Qwen3.5-4B, voto riportato sulle UV;
  - **luce tolta all'origine** (Marigold-IID sul concept o sulle viste) e, se supera la prova, un controllo **imparato**
    della luce cotta;
  - normal cotte dalla geometria densa, rugosità e metallo di Pixal3D (dove mancano, stimati da un modello addestrato);
  - texture ORM impacchettate.
- **Prova**: 60 asset etichettati dall'utente nello studio; ogni pezzo entra solo se la supera.
- **Perché**: un gioco intero usa pochi materiali master e migliaia di istanze; senza maschere di tinta e senza albedo
  pulito, gli asset generati non si ricolorano e non reggono la luce dinamica.
- **Accettazione**: un mostro esce con 3 zone ricolorabili che in Unreal cambiano colore da parametri, senza ombre cotte
  visibili sotto una luce diversa da quella del concept.

## Ruote già fatte: cosa si riusa e cosa si costruisce (verifica del 15/10/2026)

Prima di scrivere codice si cerca chi lo fa già bene. Stato e licenze verificati su GitHub il 15/10/2026.

| # | Proposta | Si riusa | Si costruisce | Note |
|---|---|---|---|---|
| 1 | Lotti e MCP | Manifest e CLI di AssetFurnace; SDK MCP ufficiale | Il lettore dei lotti, la ripresa e il server MCP: poco codice sopra ciò che c'è | ComfyUI (con [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack), MIT, fermo da dicembre 2025) sarebbe un'alternativa come orchestratore, ma ci farebbe perdere provenienza e regole di AssetFurnace |
| 2 | Dal riferimento al concept | **Qwen-Image 2.1 in modifica** dentro `sd-cli` (già in AssetFurnace), con il file della visione di Qwen3-VL-8B | La funzione `generate(prompt, references, seed)` e la prova dei 4 casi | Aggiornato il 16/10/2026: Fun ControlNet Union scartato (non c'è in `sd-cli`, chiederebbe ComfyUI); Qwen-Image-Edit-2511 (Apache-2.0) tenuto come riserva |
| 3 | Bibbia dello stile | SigLIP 2 (Apache-2.0), CSD (CC-BY-4.0), Qwen3-VL-8B con llama.cpp (MIT); [ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT) per un futuro LoRA | Bibbia, prompt, riferimenti, gara fra giudici e soglia nei lotti | Aggiornato il 16/10/2026: niente controlli sui pixel; musubi-tuner non dichiara una licenza: da evitare |
| 4 | Parti e adattamento al corpo | **Generazione**: OmniPart, HoloPart, PartCrafter (MIT). **Segmentazione di una mesh intera**: [SAMPart3D](https://github.com/Pointcept/SAMPart3D) (MIT); PartField ha licenza da verificare. **Adattamento**: la tecnica standard con gli strumenti di Blender: gabbia low-poly con Shrinkwrap e piccolo scostamento, armatura spinta con Surface Deform o Mesh Deform, pesi copiati dal corpo (Data Transfer), massimo 4 influenze | Lo script che applica la tecnica in automatico e i suoi test | Non c'è un adattatore open source affidabile: i componenti aggiuntivi seri di Blender sono a pagamento o proprietari (RadFit, FitClick). L'Outfit Asset di Unreal è pensato per la stoffa simulata, non per gusci rigidi |
| 5 | Profilo Unreal | **Usato**: Interchange glTF di Unreal (unica strada d'import); l'esportatore glTF di Blender 5.2; il mappatore di ossa di `kimodo_retarget.py`; i LOD già cotti da `finish_props.py`; IK Rig e Retargeter di Unreal (con `add_default_ops`) per Manny. Di Send to Unreal solo le **convenzioni** (prefissi, `LODn_`, `UBX_`/`UCX_`, osso `root`). **Non usato**: il codice di Send to Unreal (fermo al 2023, con Blender 5.2 si rompe in tre punti e il suo FBX non si apre in UE 5.8); FBX (Interchange perde texture e animazioni); compressione di gltfpack e glTF-Transform (Interchange rifiuta meshopt e WebP); gltfpack per i LOD (riduce poco dove ci sono cuciture UV) | Il profilo (`image_to_3dlab/unreal_profile.py`, `scripts/export_unreal.py`, `scripts/blender_export_unreal.py`), il controllo in Unreal (`scripts/unreal_import_check.py`) e il pannello nello studio | Fatto il 15/10/2026: accettazione superata su UE 5.8.3 |
| 6 | Famiglie di creature | **SkinTokens**, già in AssetFurnace, è il successore di [UniRig](https://github.com/VAST-AI-Research/UniRig) (MIT), che rigga scheletri diversi, animali compresi; [Puppeteer](https://github.com/Seed3D/Puppeteer) (Apache-2.0) rigga e anima forme qualsiasi | Prima si provano su ragno, drago e pianta; i modelli Rigify per famiglia solo dove falliscono | Puppeteer: lo scheletro è pulito, ma pelle e animazione caricano pesi non commerciali (PartField, CC-BY-NC): esclusi |
| 7 | Controlli e revisione | [glTF-Validator](https://github.com/KhronosGroup/glTF-Validator) (Apache-2.0); analisi delle mesh di Blender; statistiche di meshoptimizer; Data Validation di Unreal | I controlli specifici (sagoma, misure, compenetrazioni) e la coda nello studio | Su `main` |
| 8 | Materiali | SAM 3 (SAM License, accettata dall'utente), Qwen3.5-4B (Apache-2.0), Marigold-IID v1.1 (OpenRAIL++-M), SigLIP 2; l'idea della tabella materiale → rugosità/metallo di MaterialSeg3D++ | Il pezzo comune (viste e riporto sulle UV), le etichette nello studio, il voto per materiale, il controllo imparato | Lavori del 2026 (MaterialSeg3D++, Material Magic Wand, MatWeaver) senza codice o pesi usabili; ripiego tutto Apache: SAM 2.1 + Qwen3.5-4B |

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

**Decisione dell'utente del 15/10/2026**: prima si sviluppano in AssetFurnace le funzioni di base (1, 5, 7, 8), poi si prova. Il **LoRA dello stile è rimandato**: si lavora con prompt fissi e immagini di riferimento (Qwen-Image 2.1 in modifica).

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
| Qwen-Image 2.1 in modifica (`sd-cli`) | Recente, supportata da stable-diffusion.cpp | Strada scelta per la funzione 2. Fun ControlNet Union (settembre 2026) scartato: non c'è in `sd-cli` |
| SAM 3 (novembre 2025), SAM 3.1 (marzo 2026) | Recenti; la 3.1 migliora solo il video | Si usa SAM 3 da `transformers`; la 3.1 ha errori aperti su Windows con le RTX serie 50 |
| Marigold-IID v1.1 (maggio 2025) | Il più recente con licenza usabile | Addestrato su foto e interni: va misurato su arte stilizzata |
| OmniPart, HoloPart, PartCrafter, SAMPart3D, Puppeteer | **Codice di ricerca** (2025) | Funzionano, ma nessuno li mantiene come prodotto; pensati per Linux con CUDA. Vanno provati prima di dipenderne |
| ComfyUI-3D-Pack | Fermo da dicembre 2025 | Non lo usiamo |
| Strumenti di delighting con AI | Quasi tutti con licenze non commerciali (verifica del 16/10/2026) | Si previene all'origine; Marigold-IID è l'eccezione usabile |

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
