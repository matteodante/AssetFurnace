# Funzione 8, materiali pronti per il gioco: progetto v2

Data: 15 ottobre 2026 (aggiornato lo stesso giorno: SAM 3.1, lavori del 2026, video, Qwen3.5, togliere la luce, date).
Stato: **approvato**. PR 1 e 2 unite (#11, #12); PR 4 (luce tolta) in #26; PR 6 (rugosità e metallo) non passa la prova, vedi la sezione 10.
Sostituisce il metodo di [`2026-10-15-game-materials-design.md`](2026-10-15-game-materials-design.md) (v1), di cui
tiene l'infrastruttura. La PR [#3](https://github.com/matteodante/AssetFurnace/pull/3) è in bozza e in pausa.

## Perché si rifà

Decisione dell'utente: "controlli deterministici su luminosità, contorni o saturazione non funzionano bene", e si fa
"qualcosa di fatto bene o non lo facciamo per niente". Nella v1 non reggono tre cose:

- **zone di tinta per colore**: k-means in CIELAB, poi aggiustato asset per asset con regole a mano (12 gruppi, tinta
  ±12°, saturazione ±20 %). Un'ombra o due materiali di colore simile la ingannano;
- **controlli della luce cotta**: regressioni sulla luminanza con soglie tarate su un solo guerriero con luce iniettata;
- **normal e rugosità dalla luminanza** dell'albedo (stile Materialize): inventano rilievo dove c'è solo colore.

Le nuove regole:
1. le zone si fanno **per materiale**;
2. la luce si toglie **all'origine**, e il controllo che resta è **imparato**;
3. il PBR viene dalla geometria o da un **modello addestrato**;
4. **ogni controllo entra solo se supera una prova contro giudizi umani** (sezione 6). Se non la supera, non entra.

## 1. Il pezzo comune: rendere, segmentare o stimare, riportare sulle UV

Le tre parti nuove fanno la stessa cosa: guardano l'asset da più punti di vista con un modello 2D e riportano il
risultato sull'atlante. Si costruisce una volta sola:

1. **Rendere in Blender** (senza finestra, già usato nella repo) 16 viste più sopra e sotto. L'albedo è mostrato
   **senza luce** (shader di emissione) e ogni vista esce con tre passate: colore, coordinate UV in float, normale.
2. Ogni vista passa per un **modello 2D** (segmentatore, scomposizione della luce, stima del PBR).
3. **Si riporta sulle UV**: la passata UV dice a quale texel corrisponde ogni pixel. Ogni texel accumula il risultato
   pesato per l'angolo di vista (una superficie vista di taglio conta poco). I texel mai visti prendono il valore della
   superficie vicina.

Si tiene dalla v1 tutto ciò che serve a questo: copertura UV con ripetizione e trasformazioni della texture,
numerazione delle isole, riempimento dei margini, solo la scena mostrata, trasparenze. Le parti pure (accumulo,
voto, riempimento) sono funzioni Python testate senza Blender.

## 2. Zone di tinta per materiale

Date verificate il 15/10/2026 su GitHub e Hugging Face. "Età" vuol dire tempo dall'uscita dei pesi a oggi.

**Scelta: SAM 3 vista per vista, guidato da una lista di materiali, con voto sulle UV.**
- **Il vocabolario**: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (27/02/2026, 8 mesi, Apache-2.0) elenca fino
  a 6 materiali guardando il concept, oppure si usa una lista fissa (stoffa, cuoio, metallo, pelle, pelliccia, legno,
  pietra, osso...). Qwen3.5 legge le immagini da solo: un "Qwen3.5-VL" separato non esiste. Il ragionamento va spento
  per risposte brevi.
- **La segmentazione**:
  - [SAM 3](https://huggingface.co/facebook/sam3) (Meta, novembre 2025, 11 mesi) segmenta "tutte le regioni di questo
    concetto" da una frase di testo.
  - Gira in `transformers` (`Sam3Model`, `Sam3Processor`): PyTorch puro, pesi in safetensors, quindi Windows senza la
    repo di Meta.
  - Si interroga con ogni materiale su ogni vista, e anche sul concept riproiettato con la camera nota di Pixal3D (la
    repo lo fa già nella finitura).
- **Il voto sulle UV**: ogni texel prende il materiale più votato, pesato per l'angolo di vista. Il metallo che Pixal3D
  scrive già vale come indizio in più per "metallo". Il voto si liscia dentro le isole UV, così cuciture e ombre non
  spezzano una zona.
- **Le zone**: le 3 classi di materiale più grandi diventano R, G e B. Resta la regola v1: 3 zone passa, 2 avvisa,
  1 rimanda indietro (solo nei profili `creature` ed `equipment_part`).
- **Il nome della zona** finisce nel file accanto al GLB (`zones[i].material: "leather"`): in Unreal si sa cosa si sta
  ricolorando.

**SAM 3.1** ([facebook/sam3.1](https://huggingface.co/facebook/sam3.1), 27/03/2026, 7 mesi), verificato:
- ha la stessa SAM License di SAM 3: il file `LICENSE` è identico byte per byte;
- l'accesso è approvato a mano, con un modulo che chiede nome, data di nascita, paese, affiliazione e ruolo;
- **migliora solo il video** (Object Multiplex, tracciamento di molti oggetti più veloce): per le singole immagini il
  codice di Meta carica ancora i pesi di SAM 3;
- **non è in `transformers`**;
- su Windows con le RTX serie 50 ha errori aperti: "No available kernel"
  ([#532](https://github.com/facebookresearch/sam3/issues/532)), Flash Attention forzata
  ([#523](https://github.com/facebookresearch/sam3/issues/523)) e NCCL assente
  ([#150](https://github.com/facebookresearch/sam3/issues/150)).

Per le viste singole quindi **SAM 3, non 3.1**: stesso risultato, un'approvazione in meno, e gira su Windows.

**Segmentazione come video, da provare nel pilota.** MatWeaver tratta le viste in orbita come un video, così una
regione tiene lo stesso numero da una vista all'altra:
- `transformers` lo permette con `Sam3VideoModel`: frasi di testo multiple, fotogrammi tenuti in RAM;
- svantaggi: un errore si trascina da una vista all'altra, le viste in orbita "saltano" molto, e i parametri di
  pulizia sono pensati per video lunghi (andrebbero ritarati per 18 fotogrammi);
- l'etichetta del materiale viene comunque dalla frase di testo, e il voto sulle UV unisce già le viste.

Quindi: prima vista per vista; il video come seconda passata, che entra solo se migliora le zone nella prova della
sezione 6.

**Ripiego, tutto Apache-2.0**: [SAM 2.1](https://github.com/facebookresearch/sam2) (24/09/2024, 2 anni) in modalità
automatica (regioni senza nome) più Qwen3.5-4B che dà un nome a ogni regione da una lista chiusa. Stesso voto.

**Lavori del 2026 che fanno questo compito**: nessuno è usabile oggi. Confermano però l'impostazione: viste multiple,
segmentazione, un VLM che sceglie il materiale, voto sulle UV.

| Lavoro | Uscita | Codice e pesi | Licenza | Per noi |
|---|---|---|---|---|
| MaterialSeg3D++ (rivista Electronics; dataset MIO++, 30 materiali con rugosità e metallo) | 28/08/2026 | Codice solo della versione 2024 ([repo](https://github.com/PROPHETE-pro/MaterialSeg3D), ultimo aggiornamento 30/06/2025); pesi e dataset su Google Drive | Nessuna dichiarata | Non usabile. La tabella materiale → rugosità/metallo è un'idea da tenere per i valori di partenza (sezione 4) |
| Material Magic Wand ([2603.17370](https://arxiv.org/abs/2603.17370), CVPR 2026) | 18/03/2026 | La repo ha solo il README; un `checkpoint.pt` da 90 MB senza scheda | Nessuna | Non eseguibile. Lavora su mesh senza texture |
| Home3D 1.0 / MatWeaver ([2606.27923](https://arxiv.org/abs/2606.27923)) | 29/06/2026 | Né codice né pesi | — | Solo articolo. Segue SAM 2 con prompt a punti, non a testo; Qwen3-VL sceglie il materiale da una libreria |
| PartMat ([2608.01825](https://arxiv.org/abs/2608.01825)) | 03/08/2026 | Nessuno | — | Solo articolo |
| Gen2Physics ([2608.23869](https://arxiv.org/abs/2608.23869)) | 24/08/2026 | Nessuno | — | Solo articolo |

**Scartati**:

| Strumento | Uscita | Motivo |
|---|---|---|
| DMS46 (Apple, 46 classi di materiale) | 2022, repo attiva 09/2026 | Dataset CC-BY-NC |
| SAMa (Adobe) | 11/2024, repo 03/2026 | Licenza di ricerca Adobe, non commerciale |
| Materialistic (Adobe) | 2023 | Nessun file di licenza, ferma |
| MaterialSeg3D | 04/2024 | Nessuna licenza, Linux, CUDA 11.3 |
| PartField (NVIDIA) | 04/2025 | Non commerciale; dà parti, non materiali |
| SAMPart3D, Find3D, PartSLIP++ | 2024-2025 | Danno parti, non materiali; Linux, ricerca |
| Hunyuan3D-Part / P3-SAM | — | Tencent, escluso |
| Mappa ID dei materiali da Qwen-Image-Edit | — | Non verificata; il concept di Qwen-Image 2.1 è a licenza di ricerca |
| Make-it-Real | 07/2024 | Richiede GPT-4V in cloud |

## 3. Niente luce cotta all'origine

### A. Togliere la luce

**Scelta: [Marigold-IID v1.1](https://github.com/prs-eth/Marigold)** (prs-eth). I pesi Lighting sono del
16/12/2024, gli Appearance del 21/10/2024, entrambi aggiornati il 20/05/2025: circa 17 mesi. Licenza OpenRAIL++-M, uso
commerciale permesso con limiti d'uso.
- **Lighting** dà albedo, ombreggiatura diffusa e residuo; **Appearance** dà albedo, rugosità e metallo.
- È addestrato su interni sintetici e foto, quindi l'età e il tipo di dati sono un rischio (sezione 8). Ma la verifica
  del 15/10/2026 non ha trovato niente di più recente che lo batta con una licenza usabile.
- Si confrontano due punti dove usarlo:
  - **sul concept, prima del 3D**: Pixal3D riceve l'albedo e la finitura proietta l'albedo. Va misurato se la forma di
    Pixal3D peggiora, perché è addestrato su immagini illuminate;
  - **sulle viste della texture, dopo il 3D**, riportate sulle UV con il pezzo comune: toglie anche la luce che
    Pixal3D aggiunge di suo. L'idea di Geo-ID (accordo fra le viste, sotto) si può ricostruire attorno a Marigold.

**Candidato nuovo per le viste multiple: [UniVidX](https://github.com/houyuanchen111/UniVidX)** (02/05/2026, 5 mesi,
Apache-2.0 per codice, pesi e base Wan2.1).
- Un video RGB diventa albedo, irraggiamento e normale: un giro di render in orbita è esattamente un video.
- È un LoRA da 0,8 GB su una base da 14 miliardi di parametri: su 16 GB serve fp8 o lo scaricamento in RAM (da
  verificare).
- Codice di ricerca. Entra nel pilota come alternativa sulla texture, solo se gira sulla nostra scheda.

**Gratis, da fare comunque**: ricette di prompt con "flat even lighting, no cast shadows". Aiutano ma non bastano.

**Ripiego sul concept**: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (17/12/2025,
10 mesi, Apache-2.0) con il LoRA [QIE-2511-Studio-DeLight](https://huggingface.co/prithivMLmods/QIE-2511-Studio-DeLight)
(28/01/2026, 9 mesi, Apache-2.0, 0,24 GB). Rilumina in modo uniforme ma non dà un albedo vero, può cambiare i
dettagli, i dati di addestramento non sono dichiarati e la base da 20 miliardi va quantizzata.

**Scartati**:

| Strumento | Uscita | Motivo |
|---|---|---|
| LumiTex ([2511.19437](https://arxiv.org/abs/2511.19437), ICLR 2026) | 11/2025 | Non pulisce una texture: ne genera una nuova. Dipende da FLUX.1-dev (non commerciale) e dal renderer di Hunyuan3D-2.1 (Tencent) |
| Cosmos DiffusionRenderer (NVIDIA) | 06/2025 | Licenza commerciale, ma 56 GB, solo Linux, ideale 48 GB di VRAM. Resta un ripiego pesante in WSL2 |
| AlayaRenderer | 03-04/2026 | Derivato di DiffusionRenderer, addestrato su riprese di due giochi AAA (rischio sui diritti), pensato per scene |
| Geo-ID ([2603.13859](https://arxiv.org/abs/2603.13859)) | 06/2026 | Il codice include RGB↔X (non commerciale). L'idea dell'accordo fra viste si riusa con Marigold |
| FlowIID | 01/2026 | Solo una demo, 52 M di parametri, addestrato su interni |
| albedoLBM, CroCoDiLight | 09/2026, 02/2026 | CC BY-NC |
| IntrinsiX, OmniX, Lotus-2 | 2025 | Basati su FLUX.1-dev (non commerciale), oppure solo panorami o normali |
| RGB↔X (Adobe) | 05/2024 | Licenza di ricerca Adobe, non commerciale |
| Intrinsic di Careaga e Aksoy | 2023-2024 | "Academic use only", brevetto in corso |
| Intrinsic Image Diffusion, LumiNet | 2023-2024 | CC-BY-NC-SA |
| IntrinsicAnything | 04/2024, licenza cambiata il 15/09/2026 | Dal 15/09/2026 serve registrare l'uso aziendale con un modulo: lo decide l'utente, non io |
| Texturing di TRELLIS.2 | 12/2025 | MIT, ma Linux e almeno 24 GB |
| StableDelight | 08/2024 | Toglie solo i riflessi, non le ombre |
| Hunyuan3D-delight, MaterialMVP | — | Tencent, esclusi |
| Flux2-Klein-Delight | — | Base FLUX con licenza non commerciale |

### B. Un controllo imparato della luce cotta

Al posto delle regressioni sulla luminanza. Pronto all'uso con licenza commerciale non c'è niente: ReasonX (Adobe,
CVPR 2026) è un giudice che confronta coppie di punti e sarebbe il modello giusto, ma ha licenza non commerciale.
Si ricostruisce la stessa idea:
1. **Primo candidato: un piccolo classificatore addestrato su coppie sintetiche.**
   - Si parte da asset PBR puliti con licenza CC0 (Poly Haven, ambientCG).
   - In Blender si cuociono nell'albedo quantità casuali di ombreggiatura, occlusione, ombre portate e riflessi.
   - Il classificatore guarda le viste dell'albedo senza luce più normale e occlusione, così distingue la luce che segue
     la geometria dal colore dipinto davvero.
   - Base congelata: [SigLIP 2 base](https://huggingface.co/google/siglip2-base-patch16-224) (17/02/2025, Apache-2.0),
     la stessa della funzione 3 (un download condiviso), con una testa piccola addestrata qui.
   - Le statistiche di ombreggiatura di Marigold-IID entrano come caratteristiche in più.
2. **Secondo candidato**: un giudice alla ReasonX costruito su Qwen3.5-4B, che confronta coppie di punti dell'albedo
   ("questi due punti hanno lo stesso colore di base?").
3. **Terzo, senza addestramento**: quanta ombreggiatura trova Marigold-IID-Lighting nell'albedo, e quanto segue la
   geometria. Nessun articolo lo dimostra come rilevatore: va provato.

Entra quello che supera la prova della sezione 6. Se nessuno la supera, **non c'è controllo della luce cotta**: si conta
solo sulla luce tolta all'origine (A).

## 4. PBR

- **Normal**: si cuoce dalla mesh densa di Pixal3D sulla mesh ritopologizzata (`blender_bake_normals.py` nella
  finitura; esiste già ed è la geometria vera). Senza mesh densa (asset grezzo) **non si mette nessuna normal**: niente
  rilievo inventato.
  - Nessun modello imparato vince sulla geometria: Marigold-Normals v1.1 (10/2024), Lotus (01/2025) e StableNormal
    (06/2024) lavorano in spazio camera e scambiano il colore dipinto per rilievo.
- **Rugosità e metallo**: si tiene la mappa che Pixal3D scrive già (la finitura la cuoce). Dove manca:
  - Marigold-IID-Appearance v1.1 passa per il pezzo comune e si riporta sulle UV;
  - se non supera la prova, si usano valori di partenza per materiale, presi dalle zone della sezione 2, sull'esempio
    della tabella materiale → rugosità/metallo di MaterialSeg3D++ (metallo: metallo 1, rugosità 0,35; stoffa: 0,
    0,9...), validati con la stessa prova.
- **Scartati**:
  - RGB↔X e DSINE: non commerciali;
  - DeepBump (03/2025): GPL-3.0, e ricava il rilievo dall'albedo, la stessa trappola della v1;
  - Material Anything (06/2025): MIT, ma solo Linux; resta come ripiego pesante;
  - Material Palette, DreamMat: lenti, un modello o un'ottimizzazione per asset;
  - Paint3D, UniTEX, TEXGen: niente rugosità né metallo, e licenze deboli.

## 5. Cosa resta della v1 e cosa si butta

| Resta (testato) | Si butta |
|---|---|
| Modifica del GLB nel JSON senza riesportarlo (scheletro e animazioni conservati), `set_view_bytes`, texture aggiunte, `extras` | `cluster_colours`, `same_pigment`, `zone_features`: le zone per colore |
| Impacchettamento ORM, occlusione già presente conservata, riuso della texture solo se esclusiva | `light_direction_record`, `occlusion_record`, `highlight_record` e le loro soglie |
| Provenienza portata accanto al risultato, hash della sorgente, file accanto al GLB | `height_from_luminance`, `normal_from_albedo`, rugosità dalla luminanza |
| Formula di tinta a scostamento `out = albedo + m·g·(t − c)`, uguale in Blender, studio e Unreal | — |
| Lettura dell'albedo come il renderer (scena, ripetizione, trasformazioni, trasparenze, colori e normali dei vertici, strip e fan) | — |
| Copertura UV, isole, margini, maschera RGB, nomi dei file `T_<Nome>_Tint/ORM` | — |
| Anteprima in Blender, pannello dello studio, API, CLI, registro dei controlli (si ricollegano ai controlli nuovi) | — |

## 6. Prova di accettazione: contro giudizi umani, su più tipi di asset

**Insieme etichettato**: 60 asset generati, 20 umanoidi, 20 creature e 20 prop, presi dal pilota e da nuove
generazioni con la GPU condivisa (circa 15 minuti l'uno). L'utente li etichetta nello studio, nella coda di revisione
della funzione 7, con tre giudizi per asset:
- la luce è cotta sì o no;
- le zone sono giuste, oppure le corregge cliccando le regioni;
- i materiali, in un render sotto 3 luci, sembrano giusti o sbagliati.

Circa 10 asset sono etichettati due volte, per misurare quanto l'utente è d'accordo con sé stesso: è il tetto che un
modello può raggiungere. Le etichette restano in `output/` (contenuti nostri, fuori da git).

| Parte | Entra se |
|---|---|
| Controllo della luce cotta | Area ROC ≥ 0,90 (limite inferiore dell'intervallo bootstrap al 95 %) sul gruppo di prova; alla soglia scelta sul gruppo di taratura, almeno l'80 % dei difettosi fermati e al massimo il 10 % dei buoni. Divisione per asset, mai lo stesso asset in taratura e prova; risultati anche per tipo |
| Zone per materiale | IoU medio delle 3 zone con le correzioni dell'utente (abbinamento ottimo, R/G/B scambiabili) ≥ 0,75; ricolorazione "giusta" per l'utente su almeno l'80 % degli asset, "chiaramente sbagliata" al massimo sul 10 % |
| Luce tolta all'origine | Sotto una luce laterale, "nessuna ombra dipinta visibile" su almeno l'80 % degli asset, contro la base attuale; la forma di Pixal3D non peggiora (confronto cieco) |
| Rugosità e metallo stimati | Su 30 asset Poly Haven (CC0) con mappe vere: rugosità RMSE ≤ 0,15 e almeno il 20 % meglio del valore costante; metallo F1 ≥ 0,8 e metallo sbagliato su meno del 5 % dell'area; nel confronto cieco, preferito o pari almeno nel 60 % dei casi |

Il pezzo che non passa non entra. Lo dico chiaro: l'insieme etichettato costa **tempo all'utente** (60 asset per tre
giudizi, circa un'ora nello studio).

## 7. Download (nessuno senza conferma)

Ogni modello passa dal comando di installazione con nome, licenza e dimensione, come vuole la repo.

| File | Dimensione | Licenza | Serve per |
|---|---|---|---|
| `facebook/sam3` → `model.safetensors` (11/2025; HF, accesso da richiedere; per `transformers`) | 3,44 GB | SAM License (personalizzata: uso commerciale gratuito, vietato l'uso militare, **Meta può cambiarla**) | Zone per materiale |
| `Qwen/Qwen3.5-4B` (02/2026, bf16) | 9,3 GB | Apache-2.0 | Nomi dei materiali; ripieghi Qwen3-VL-8B-Instruct (10/2025) o Gemma-4-E4B-it (03/2026, Apache-2.0) |
| `prs-eth/marigold-iid-lighting-v1-1` (12/2024, aggiornato 05/2025; fp16: unet 1,73 + text encoder 0,68 + vae 0,17) | circa 2,6 GB | OpenRAIL++-M (uso commerciale permesso, con limiti d'uso da riportare nella provenienza) | Luce tolta; caratteristiche del controllo |
| `prs-eth/marigold-iid-appearance-v1-1` (10/2024, aggiornato 05/2025; fp16) | circa 2,6 GB | OpenRAIL++-M | Rugosità e metallo dove mancano |
| `google/siglip2-base-patch16-224` (02/2025; condiviso con la funzione 3) | 1,54 GB | Apache-2.0 | Base del classificatore |
| `facebook/sam2.1-hiera-large` (09/2024; solo per il ripiego) | 0,9 GB | Apache-2.0 | Ripiego delle zone |
| Asset CC0 di Poly Haven per addestramento e prova del PBR | da misurare, qualche GB | CC0 | Coppie sintetiche, verità di riferimento |

In tutto, con la prima scelta, circa **19 GB**. Da provare solo nel pilota e solo se gira su 16 GB: UniVidX (LoRA 0,8 GB su Wan2.1-T2V-14B, Apache-2.0, base da misurare). Librerie nuove: `transformers` e `diffusers` (Apache-2.0).
**Dove gira**: la repo di Meta per SAM 3/3.1 ha errori aperti su Windows e sulle RTX serie 50 (#532, #523, #150). Si
usa la versione in `transformers` (PyTorch puro) e, se serve, WSL2. Marigold in `diffusers` dovrebbe girare su Windows (da verificare all'inizio, prima
di qualsiasi lavoro lungo). Tutto passa dal lucchetto della GPU condivisa.

## 8. Rischi

- **Licenze**:
  - la SAM License è personalizzata e Meta può cambiarla: è l'utente a decidere se accettarla, altrimenti si usa il
    ripiego tutto Apache;
  - i limiti d'uso di OpenRAIL++-M viaggiano nella provenienza;
  - Qwen-Image 2.1 resta a licenza di ricerca, e il problema esiste già a monte;
  - Pixal3D è rilasciato da TencentARC con licenza MIT: non rientra nell'esclusione di Hunyuan, che riguarda la
    licenza Tencent, ma lo segnalo.
- **Età e dati**: Marigold-IID ha circa 17 mesi ed è addestrato su interni sintetici e foto; SAM 3 su foto. Su arte stilizzata il comportamento non è provato:
  è proprio ciò che misura la sezione 6.
- **Classificatore**: potrebbe imparare gli artefatti sintetici invece della luce cotta vera; l'insieme umano è l'unica
  prova che conta.
- **Tempo per asset**: 18 viste per 6 materiali con SAM 3, più Marigold, fanno qualche minuto in più a lotto.
- **Etichette**: se l'utente non ha il tempo di etichettare, nessun controllo può entrare. È la regola, non un ripiego.

## 9. Ordine di lavoro, PR piccole

1. **Il pezzo comune** (render, riporto sulle UV) e la **rimozione dei metodi buttati**. La normal viene solo dalla
   cottura, la rugosità e il metallo solo da Pixal3D. Nessun download.
2. **Etichettatura nello studio**: i tre giudizi nella coda della funzione 7 e un pannello per correggere le zone.
   Nessun download.
3. **Zone per materiale** (SAM 3 o il ripiego), dopo la conferma del download; prova della sezione 6.
4. **Luce tolta** (Marigold-IID-Lighting): confronto concept contro texture; prova della sezione 6.
5. **Controllo imparato** della luce cotta: dati sintetici, addestramento, prova contro l'insieme umano.
6. **Rugosità e metallo stimati** dove mancano: prova su Poly Haven.

Ogni PR può avere un giro di review di Codex, facoltativo; si correggono solo i P1 veri (regola di `AGENTS.md`).

**Ordine aggiornato il 16/10/2026.** SAM 3 resta bloccato finché l'utente non compila il modulo di accesso su Hugging
Face (dati personali: lo fa lui). Quindi le PR che non ne dipendono vengono prima: **4 → 6 → 5 → 3**. La PR 3 arriva
con un'interfaccia del segmentatore pronta a ricevere SAM 3 e il ripiego SAM 2.1 dietro la stessa interfaccia.

## 10. Stato del lavoro

### PR 4, luce tolta (16/10/2026)

Download confermati dall'utente il 16/10/2026 ("permesso per tutto"): Qwen3.5-4B, Marigold-IID v1.1, SigLIP 2.

- **Registro dei modelli** (`image_to_3dlab/material_models.py`): Marigold-IID Lighting e Appearance, Qwen3.5-4B e
  SAM 3, ognuno con revisione fissata, elenco dei file, dimensione e licenza. Di Marigold si scaricano solo i file
  fp16 (2,40 GB a modello invece di 9,6). Installazione con `scripts/bootstrap_materials.py`, che annuncia e chiede;
  SAM 3 viene saltato finché l'account non ha l'accesso.
- **Ambiente separato** (`vendor/materials/.venv`, 3,1 GB): `diffusers` richiede `huggingface-hub` sotto la 2, e
  l'ambiente del visore è alla 2.2. Scaricarlo lì avrebbe cambiato le dipendenze di tutti. Marigold gira quindi in un
  processo a parte (`scripts/materials_worker.py`), come SkinTokens. Dentro ci sono anche `transformers` 5.20 per
  Qwen3.5 e SAM 3.
- **Licenza**: i pesi sono CreativeML Open RAIL++-M (il codice di Marigold è Apache-2.0). Le restrizioni d'uso
  (allegato A) e l'obbligo di passarle a chi riceve i pesi finiscono nella provenienza di ogni risultato; sulle uscite
  il licenziante non rivendica diritti. Attenzione per la PR 5: un modello addestrato a imitare le uscite di Marigold
  sarebbe un suo "derivato" con le stesse restrizioni; usare le sue uscite come caratteristiche no.
- **Metodo**: Marigold risponde a 768 pixel, l'atlante è a 2K, quindi non si sostituisce la texture con il suo albedo.
  Ogni pixel dice di quanto il suo albedo è più chiaro o più scuro del colore mostrato (un guadagno di luminanza). I
  guadagni si lisciano **nella vista** (2 % del lato), dove la superficie è continua: Marigold legge anche il dettaglio
  dipinto fine (profili, incisioni) come ombreggiatura, e deve andare via solo la luce larga. Poi si riportano sulle UV
  con il pezzo comune, si lisciano poco dentro ogni isola, si prendono rispetto alla mediana (la luminosità complessiva
  del modello non è affidabile su arte stilizzata) e si limitano a 3 volte. La texture a piena risoluzione si
  moltiplica per il guadagno: tinta e dettaglio dipinto restano. Stesso metodo sul concept, prima del 3D
  (`delight_image`).
- **Prova vera** (cavaliere di `output/compare-20261015`, 18 viste a 768 px): Marigold 42,8 s sulla 5070 Ti, il render
  in Blender pochi secondi. Il riflesso dipinto sul pettorale e le luci di contorno su spalle e ginocchia si attenuano,
  profili e incisioni restano. Senza il liscio nella vista i guadagni toccavano i limiti (×1/3 e ×3) e appiattivano il
  dettaglio; con il liscio il 95° percentile scende a ×1,78. Il 5° resta a ×1/3: le zone più chiare dipinte vengono
  scurite molto, ed è proprio ciò che la prova della sezione 6 deve giudicare.
- **Correzione al pezzo comune** (PR 1): le camere delle viste ad anello ruotavano attorno al proprio asse
  (`to_track_quat("-Z", "Z")`, il "su" era l'asse stesso di mira), e il cavaliere compariva sdraiato in metà delle viste.
  Per il riporto sulle UV non contava; per un modello 2D sì. Ora la base della camera è costruita a mano con il "su" del
  mondo, e l'inquadratura segue il contorno del modello visto da ogni camera (il cavaliere passa dal 65 % a circa il
  95 % dell'altezza).
- **Lucchetto della GPU** in codice (`image_to_3dlab/gpu_lock.py`): preso solo attorno al processo di Marigold, non
  durante il render in Blender (CPU) né le installazioni.
- **Non ammesso**: il passo si usa solo se chiesto (`scripts/delight_materials.py`), e il file accanto al GLB lo
  dichiara (`admitted: false`). Entra quando supera la prova della sezione 6 (ombre dipinte sotto luce laterale),
  che richiede l'insieme etichettato.

### PR 6, rugosità e metallo stimati (16/10/2026): **non passa**

- **Cosa c'è**:
  - `image_to_3dlab/pbr_estimate.py` e `scripts/estimate_pbr.py`: Marigold-IID Appearance v1.1 sulle viste in orbita,
    con rugosità e metallo riportati sull'atlante come mappa glTF (G rugosità, B metallo). La mappa propria di un
    materiale (quella di Pixal3D) resta, salvo `--force`.
  - `image_to_3dlab/pbr_trial.py` e `scripts/pbr_trial.py`: la prova della sezione 6. Prende 30 modelli Poly Haven
    (CC0, licenza verificata su polyhaven.com/license): metà con il tag del metallo, senza piante né rocce, sempre gli
    stessi. Li impacchetta in GLB, toglie la mappa vera, la stima di nuovo e confronta sui texel coperti.
  - La costante di confronto per un asset è la rugosità media degli *altri* asset. Si riporta anche il miglior F1 del
    metallo con la soglia scelta a posteriori: è un limite ottimistico, non un criterio.
- **Viste illuminate**: sulle viste senza luce Appearance diceva "metallo" a metà su un bersaglio di sughero (2 asset:
  rugosità peggio della costante del 96 %, metallo sbagliato sul 75 % dell'area). Il modello legge i materiali dalle
  ombre, quindi le viste ora possono avere anche un'immagine sotto una luce da studio neutra: colore di base e normal
  map su un dielettrico con rugosità 0,5, luce dalla camera in alto a sinistra, cielo grigio tenue. Neutra, così
  nessun riflesso è inventato da noi (`render_views(..., lit=True)`).
- **Esito con le viste illuminate, 16 asset su 30**:

  | Criterio | Soglia | Risultato |
  |---|---|---|
  | Rugosità, RMSE | ≤ 0,15 | 0,247 |
  | Rugosità, meglio della costante | ≥ 20 % | −4 % (costante 0,238) |
  | Metallo, F1 a 0,5 | ≥ 0,8 | 0,37 (0,44 con la soglia migliore a posteriori, 0,32) |
  | Metallo sbagliato | < 5 % dell'area | 58 % |

  Il modello chiama metallo legno, plastica e sughero (ciotola, sedia, cartello) e manca più di metà del metallo del martello da guerra.
- **Prova fermata a 16 su 30**, perché l'esito era già deciso: anche con i 14 rimasti perfetti, alla fine il metallo
  resterebbe sbagliato su almeno il 27,7 % dell'area e la rugosità avrebbe RMSE di almeno 0,171. Proseguire avrebbe
  tenuto la GPU condivisa per un'ora e mezza senza cambiare la risposta.
- **Conseguenze**:
  - la rugosità e il metallo restano quelli di Pixal3D;
  - dove mancano, il ripiego della sezione 4 sono i valori di partenza per materiale, presi dalle zone (PR 3);
  - lo strumento non compare nella pagina di installazione;
  - i pesi di Appearance sono stati cancellati (liberati 1,7 GB; encoder del testo e VAE sono condivisi con Lighting);
  - i 30 asset impacchettati e il rapporto restano in `output/pbr-trial/`, per misurare un modello migliore allo
    stesso modo.
- **Limite della prova**: Poly Haven sono scansioni realistiche, il campo più vicino a quello di Marigold. Non passare
  qui dice già abbastanza per l'arte stilizzata.

## Seguiti

I punti rimandati della v1 restano nella sezione "Seguiti" di
[`2026-10-15-game-materials-design.md`](2026-10-15-game-materials-design.md); quelli legati ai metodi buttati cadono
con loro.
