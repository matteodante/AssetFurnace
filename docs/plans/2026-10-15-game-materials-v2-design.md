# Funzione 8, materiali pronti per il gioco: progetto v2

Data: 15 ottobre 2026. Stato: **proposta, in attesa di approvazione**. Niente di nuovo è implementato.
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

**Scelta: SAM 3 guidato da una lista di materiali.**
- **Il vocabolario**: Qwen3-VL (Apache-2.0) elenca fino a 6 materiali guardando il concept, oppure una lista fissa
  (stoffa, cuoio, metallo, pelle, pelliccia, legno, pietra, osso...).
- **La segmentazione**: [SAM 3](https://github.com/facebookresearch/sam3) (Meta, ultimo aggiornamento 14/10/2026)
  segmenta "tutte le regioni di questo concetto" da una frase di testo. Si interroga con ogni materiale, su ogni vista
  e anche sul concept, che si riproietta con la camera nota di Pixal3D (la repo lo fa già nella finitura).
- **Il voto sulle UV**: ogni texel prende il materiale più votato. Il metallo che Pixal3D scrive già vale come indizio
  in più per "metallo". Il voto si liscia dentro le isole UV, così cuciture e ombre non spezzano una zona.
- **Le zone**: le 3 classi di materiale più grandi diventano R, G e B. Resta la regola v1: 3 zone passa, 2 avvisa,
  1 rimanda indietro (solo nei profili `creature` ed `equipment_part`).
- **Il nome della zona** finisce nel file accanto al GLB (`zones[i].material: "leather"`): in Unreal si sa cosa si sta
  ricolorando.

**Ripiego, tutto Apache-2.0**: [SAM 2.1](https://github.com/facebookresearch/sam2) in modalità automatica (regioni
senza nome) più Qwen3-VL che dà un nome a ogni regione da una lista chiusa. Stesso voto. Più lento, ma ogni pezzo ha
una licenza Apache.

**Scartati**:

| Strumento | Motivo |
|---|---|
| DMS46 (Apple, 46 classi di materiale, la tassonomia giusta) | Licenza "personal", dati CC-BY-NC |
| SAMa (Adobe), Materialistic (Adobe) | Non commerciale; senza file di licenza |
| MaterialSeg3D | Senza licenza, Linux, CUDA 11.3 |
| PartField (NVIDIA) | Non commerciale; dà parti, non materiali |
| SAMPart3D, Find3D, PartSLIP++ | Danno parti, non materiali; Linux, ricerca |
| Hunyuan3D-Part / P3-SAM | Tencent, escluso |
| Mappa ID dei materiali generata da Qwen-Image-Edit | Non verificata; il concept di Qwen-Image 2.1 è comunque a licenza di ricerca |
| Make-it-Real | Richiede GPT-4V in cloud |

## 3. Niente luce cotta all'origine

### A. Togliere la luce

Due punti dove intervenire, da confrontare nel pilota:
- **Sul concept, prima del 3D**: [Marigold-IID-Lighting v1.1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)
  scompone l'immagine in albedo, ombreggiatura diffusa e residuo. Pixal3D riceve l'albedo, e la finitura proietta
  l'albedo invece della foto illuminata. Rischio: Pixal3D è addestrato su immagini illuminate, quindi va misurato se la
  forma peggiora.
- **Sulla texture, dopo il 3D**: la stessa scomposizione sulle viste del pezzo comune (sezione 1), riportata sulle UV.
  Così si toglie anche la luce che Pixal3D aggiunge di suo.

**Gratis, da fare comunque**: ricette di prompt con "flat even lighting, no cast shadows". Aiutano ma non bastano.

**Ripiego**: Qwen-Image-Edit-2511 (Apache-2.0) con il LoRA
[QIE-2511-Studio-DeLight](https://huggingface.co/prithivMLmods/QIE-2511-Studio-DeLight) (Apache-2.0, 0,24 GB) sul
concept. Rilumina in modo uniforme ma non dà un albedo vero, e i dati di addestramento del LoRA non sono dichiarati.

**Scartati**:

| Strumento | Motivo |
|---|---|
| RGB↔X (Adobe) | Licenza di ricerca Adobe, non commerciale |
| Intrinsic di Careaga e Aksoy | "Academic use only", brevetto in corso |
| Intrinsic Image Diffusion, LumiNet | CC-BY-NC-SA |
| IntrinsicAnything | Dal 15/09/2026 serve registrare l'uso aziendale con un modulo: da decidere all'utente, non lo scelgo io |
| Cosmos DiffusionRenderer (NVIDIA) | Licenza commerciale, ma 56 GB, solo Linux, ideale 48 GB di VRAM |
| Texturing di TRELLIS.2 | MIT, ma Linux e almeno 24 GB |
| StableDelight | Toglie solo i riflessi, non le ombre; base probabilmente SD2 |
| Hunyuan3D-delight, MaterialMVP | Tencent, esclusi |
| Flux2-Klein-Delight | Base FLUX con licenza non commerciale |

### B. Un controllo imparato della luce cotta

Al posto delle regressioni sulla luminanza:
1. **Primo candidato: un piccolo classificatore addestrato su coppie sintetiche.**
   - Si parte da asset PBR puliti con licenza CC0 (Poly Haven, ambientCG).
   - In Blender si cuociono nell'albedo quantità casuali di ombreggiatura, occlusione, ombre portate e riflessi.
   - Il classificatore guarda le viste dell'albedo senza luce più normale e occlusione, così distingue la luce che segue
     la geometria dal colore dipinto davvero.
   - Base congelata: SigLIP 2 base (Apache-2.0), la stessa scelta dalla funzione 3 (un download condiviso), con una
     testa piccola addestrata qui.
   - Le statistiche di ombreggiatura di Marigold-IID entrano come caratteristiche in più.
2. **Secondo candidato, senza addestramento**: quanta ombreggiatura trova Marigold-IID-Lighting nell'albedo, e quanto
   segue la geometria. Nessun articolo lo dimostra come rilevatore: va provato.

Entra quello che supera la prova della sezione 6. Se nessuno la supera, **non c'è controllo della luce cotta**: si conta
solo sulla luce tolta all'origine (A).

## 4. PBR

- **Normal**: si cuoce dalla mesh densa di Pixal3D sulla mesh ritopologizzata (`blender_bake_normals.py` nella
  finitura; esiste già ed è la geometria vera). Senza mesh densa (asset grezzo) si mette **nessuna normal**: niente
  rilievo inventato. Nessun modello imparato vince sulla geometria: quelli di immagine (Marigold-Normals, Lotus,
  StableNormal) lavorano in spazio camera e scambiano il colore dipinto per rilievo.
- **Rugosità e metallo**: si tiene la mappa che Pixal3D scrive già (la finitura la cuoce). Dove manca:
  - [Marigold-IID-Appearance v1.1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) (albedo, rugosità e
    metallo) passa per il pezzo comune e si riporta sulle UV;
  - se non supera la prova, si usano valori di partenza per materiale, presi dalle zone della sezione 2 (metallo:
    metallo 1, rugosità 0,35; stoffa: 0, 0,9...), validati con la stessa prova.
- **Scartati**:
  - RGB↔X e DSINE: non commerciali;
  - DeepBump: GPL-3.0, e ricava il rilievo dall'albedo, la stessa trappola della v1;
  - Material Anything: MIT, ma solo Linux, e i pesi probabilmente ereditano la licenza di Stable Diffusion; resta come
    ripiego pesante;
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
| `facebook/sam3` → `sam3.pt` (HF, accesso da richiedere) | 3,45 GB | SAM License (personalizzata: uso commerciale gratuito, vietato l'uso militare, **Meta può cambiarla**) | Zone per materiale |
| `Qwen/Qwen3-VL-4B-Instruct` | circa 8,9 GB | Apache-2.0 | Nomi dei materiali. Se la funzione 2 installa già Qwen3-VL-8B GGUF, si prova a riusarlo |
| `prs-eth/marigold-iid-lighting-v1-1` (fp16: unet 1,73 + text encoder 0,68 + vae 0,17) | circa 2,6 GB | OpenRAIL++-M (uso commerciale permesso, con limiti d'uso da riportare nella provenienza) | Luce tolta; caratteristiche del controllo |
| `prs-eth/marigold-iid-appearance-v1-1` (fp16) | circa 2,6 GB | OpenRAIL++-M | Rugosità e metallo dove mancano |
| `google/siglip2-base-patch16-224` (condiviso con la funzione 3) | 1,54 GB | Apache-2.0 | Base del classificatore |
| `facebook/sam2.1-hiera-large` (solo per il ripiego) | 0,9 GB | Apache-2.0 | Ripiego delle zone |
| Asset CC0 di Poly Haven per addestramento e prova del PBR | da misurare, qualche GB | CC0 | Coppie sintetiche, verità di riferimento |

In tutto, con la prima scelta, circa **19 GB**. Librerie nuove: `transformers` e `diffusers` (Apache-2.0).
**Dove gira**: SAM 3 nativo ha problemi aperti su Windows e sulle RTX serie 50. Si usa la versione in `transformers`
(PyTorch puro) e, se serve, WSL2. Marigold in `diffusers` dovrebbe girare su Windows (da verificare all'inizio, prima
di qualsiasi lavoro lungo). Tutto passa dal lucchetto della GPU condivisa.

## 8. Rischi

- **Licenze**:
  - la SAM License è personalizzata e Meta può cambiarla: è l'utente a decidere se accettarla, altrimenti si usa il
    ripiego tutto Apache;
  - i limiti d'uso di OpenRAIL++-M viaggiano nella provenienza;
  - Qwen-Image 2.1 resta a licenza di ricerca, e il problema esiste già a monte;
  - Pixal3D è rilasciato da TencentARC con licenza MIT: non rientra nell'esclusione di Hunyuan, che riguarda la
    licenza Tencent, ma lo segnalo.
- **Stile**: Marigold è addestrato su foto e interni, SAM 3 su foto. Su arte stilizzata il comportamento non è provato:
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

Ogni PR passa la review di Codex con le regole correnti (P1 sempre; P2 solo se è un errore visibile; al massimo 2 giri).

## Seguiti

I punti rimandati della v1 restano nella sezione "Seguiti" di
[`2026-10-15-game-materials-design.md`](2026-10-15-game-materials-design.md); quelli legati ai metodi buttati cadono
con loro.
