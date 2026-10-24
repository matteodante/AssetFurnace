# Funzione 3: bibbia dello stile (progetto)

Data: 15 ottobre 2026. Stato: **approvato dall'utente il 15/10/2026** (commit 559d749). **Concluso il 17/10/2026**:
bibbia, modelli di prompt e giudici restano su main; il giudice automatico è **rimandato**, come il LoRA, e i
riferimenti di stile non entrano nei concept nuovi (sezione 7.2).
Piano di riferimento: il piano di produzione (privato), funzione 3.
Il LoRA dello stile è **fuori** da questo lavoro (rimandato dall'utente il 15/10/2026). Anche il giudice automatico
(sezione 4) è rimandato, dal 17/10/2026.

**AssetFurnace è una repo a sé**: la bibbia è un formato generico, ogni progetto ha la sua e la tiene nella propria
repo. Dentro AssetFurnace non entrano immagini, nomi o stile di un progetto; gli esempi qui sotto sono inventati.

**Cosa cambia rispetto alla prima versione** (richieste dell'utente del 15/10/2026):
- tolto `style.palette`: niente controlli deterministici sui pixel (luminosità, saturazione, contorni);
- nuova sezione 3: **lo stile si ottiene all'origine**, dando alla generazione 1-3 immagini approvate come riferimento,
  poi tenendo il migliore di alcuni tentativi;
- la misura è **solo imparata**, e tre candidati fanno una gara sui giudizi dell'utente: entra solo chi la vince
  (sezione 4).

In breve: una cartella con le immagini approvate e rifiutate di un progetto e le loro note; modelli di prompt per tipo
di asset; riferimenti di stile presi dalla bibbia per generare; e un giudice imparato che dice quanto un concept nuovo è
nello stile. Il giudizio esce nel formato dei controlli della funzione 7, così lo studio lo mostra e i lotti lo usano
come soglia.

Perché serve, con un esempio già sul disco: il lupo mannaro di `output/compare-20261015/` è stato chiesto con
"stylized fantasy … hand-painted fantasy MMORPG monster" ed è uscito quasi realistico, grigio e spento. **Le parole del
prompt non bastano a garantire lo stile.**

## 1. Il formato della bibbia

Una bibbia è **una cartella** autosufficiente. Si copia, si mette sotto git nella repo del gioco, si apre da un altro PC.

```
my-game-style/
  bible.json
  images/
    3f9a1c2e-knight.png
    3f9a1c2e-knight.provenance.json     # copiata dall'originale: la licenza viaggia con l'immagine
  rejected/
    9c04e1b7-wolf.png                   # un concept rimandato, con la sua nota: insegna cosa NON è lo stile
  .cache/                               # rigenerabile, da ignorare in git
    embeddings-<modello>.npz
    judgements.json
```

`bible.json`:

```json
{
  "schema_version": 1,
  "name": "my-game",
  "description": "Hand-painted fantasy: saturated colours, slightly exaggerated proportions.",
  "style": {
    "wording": "stylised hand-painted fantasy game art, bold saturated colours, slightly exaggerated proportions, clean readable shapes",
    "criteria": [
      "Is it painted in the same way as the references, not photographic?",
      "Are the colours as saturated and bright as in the references?",
      "Are the proportions exaggerated the way they are in the references?"
    ]
  },
  "kinds": {
    "creature": {"wording": "oversized claws and head, vivid fur colours"}
  },
  "images": [
    {"file": "images/3f9a1c2e-knight.png", "sha256": "3f9a1c2e…", "kind": "character",
     "note": "Colours and proportions are exactly right.", "added": "2026-10-16T09:12:00+00:00",
     "source": "images/stylized-fantasy-knight-hand-painted-gam__20261015-192236/….png"}
  ],
  "rejected": [
    {"file": "rejected/9c04e1b7-wolf.png", "sha256": "9c04e1b7…", "kind": "creature",
     "note": "Almost photographic and grey: not our style.", "added": "2026-10-16T09:15:00+00:00"}
  ],
  "score": {"method": null, "warn": null, "fail": null}
}
```

- **Aggiungere** un'immagine (approvata o rifiutata): la si copia nella cartella con le prime 8 cifre del suo hash nel
  nome, si copia accanto la sua `.provenance.json` se c'è, si scrive la voce con la **nota obbligatoria**.
- **Togliere**: si cancella la voce e il file (con conferma nello studio).
- I **rifiutati** servono a due cose: la prova di accettazione cresce da sola con l'uso, e il giudice impara anche da
  cosa non va (sezione 4.4).
- `style.criteria`: le domande che il giudice visione-linguaggio fa su ogni concept (sezione 4.3). Le scrive l'utente,
  in inglese; ci sono tre domande di partenza generiche.
- `score.method` dice quale misura ha vinto la gara per questa bibbia; `warn` e `fail` sono le soglie tarate.
- **Più bibbie**: AssetFurnace tiene un elenco in `output/styles/bibles.json` (`[{"name", "path"}]`). Una bibbia nuova
  nasce in `output/styles/<nome>/`; una cartella esterna (per esempio nella repo del gioco, il posto normale) si
  **registra** con il suo percorso, senza copiarla. `output/` è fuori da git: AssetFurnace non conserva bibbie di
  progetti. Una è quella attiva nello studio; la CLI e i lotti la nominano con `--style-bible`.

Codice: `image_to_3dlab/style/bible.py` (leggere, validare, aggiungere, togliere; scrittura atomica del JSON), senza
dipendenze nuove. Controlla che ogni file elencato esista e abbia il suo hash.

## 2. I modelli di prompt per tipo di asset

Due livelli, ciascuno nel suo posto:

1. **`image_to_3dlab/prompt_recipes.json`** (generico): cosa serve al **3D**, non allo stile. Oggi ha `character` e
   `prop`; aggiungo `creature`, `equipment_part` e `map_object` con `wording`, `tips` e parole a rischio (`trouble`).
   I 5 tipi: personaggio (`character`), creatura (`creature`), oggetto (`prop`), pezzo di equipaggiamento
   (`equipment_part`), oggetto di mappa (`map_object`). Profili dei controlli della funzione 7: character e creature →
   `creature`, prop e map_object → `prop`, equipment_part → `equipment_part`.
2. **La bibbia** (specifica del progetto): `style.wording` più `kinds.<tipo>.wording`.

Ordine fisso, in una funzione pura testata,
`image_to_3dlab/style/prompts.py: compose(subject, kind, bible=None, recipes=None)`:

```
<soggetto>, <wording del tipo dalle ricette>, <wording dello stile>, <wording del tipo nella bibbia>
```

La usano la CLI (`scripts/style_bible.py prompt`), i lotti (una riga con `kind` e `subject` al posto dell'immagine) e lo studio. Con Qwen-Image a cfg 1.0,
il valore di base di AssetFurnace, il prompt negativo non ha effetto: lo stile va detto in positivo.

## 3. Lo stile all'origine: riferimenti dalla bibbia e il migliore di N

### Cosa dice la verifica (ottobre 2026)

- **Dare immagini di riferimento allo stesso modello che genera è pratica corrente.** Qwen-Image 2.1 accetta fino a 10
  riferimenti, Qwen-Image-Edit-2511 lavora con più immagini insieme, FLUX.2 klein ne accetta fino a 4. È il modo più
  rapido di cominciare, senza addestrare nulla.
- **Ma non basta per centinaia di immagini.** Il manuale di stile di ComfyUI (marzo 2026) dice che generare contenuti
  nuovi con un modello di modifica e un riferimento di stile "tends to be more hit or miss": dipende molto dal
  riferimento scelto. Per contenuti nuovi in uno stile fisso la strada affidabile resta il LoRA, che l'utente ha
  rimandato. Nessuna fonte confronta le due strade su un set di asset per un gioco: lo misuriamo noi.
- Conseguenza: i riferimenti alzano la probabilità di stare nello stile, **il giudice (sezione 4) dice quando non ci
  si è riusciti**. Insieme fanno il "migliore di N": si generano 2-4 tentativi e si tiene il migliore che passa la
  soglia. Se nessuno passa, la riga del lotto si ferma con il motivo.

**Esito (17/10/2026, sezione 7.2): niente riferimenti per i concept nuovi.** La modifica copia anche il contenuto dei
riferimenti (ogni creatura prendeva la testa della prima), e il braccio senza riferimenti è stato il più approvato.
`compose()` e `style_bible.py prompt` li lasciano spenti se non chiesti; restano per la funzione 2, che ridisegna un
oggetto preciso.

### Cosa costruisco io e cosa fa la funzione 2

**Motore deciso dall'utente**: solo Qwen-Image 2.1 in modifica dentro `sd-cli` (Qwen-Image-Edit-2511 è fuori dal
progetto). La **funzione 2** possiede il motore, il comando e l'aggancio in `viewer/image_api.py`, e offre una funzione
pura `generate(prompt, references, seed) -> Path`; `-r` è ripetibile: prima la mappa di forma, poi 1-3 riferimenti di
stile, in quest'ordine. Non tocco niente di questo. Io fornisco:

- `image_to_3dlab/style/refs.py: pick(bible, kind, k)`: sceglie 1-3 immagini approvate da usare come riferimento. Prima
  quelle dello stesso tipo; a parità, quelle che il giudice ritiene più tipiche della bibbia (le più vicine alle altre).
  Funzione pura, deterministica, testata.
- La frase che accompagna i riferimenti nel prompt ("Paint the subject in the same style as the reference images: same
  brushwork, colours and proportions"), in `compose()`.
- Il giudice, per scegliere il migliore fra N tentativi.

**Prova** (la conduco e la misuro io, con `generate` della funzione 2): stessi soggetti e semi, tre bracci: solo
prompt, prompt + 1 riferimento, prompt + 3 riferimenti. Si misura quanti escono nello stile secondo i giudizi
dell'utente, e quanto costa in tempo. Concordato con la sessione della funzione 2 il 15/10/2026.

**Licenze dei motori** (verificate sulle schede Hugging Face, 15/10/2026):

| Motore | Pesi | Riferimenti | Nota |
|---|---|---|---|
| Qwen-Image 2.1 (quello in uso) | **Qwen Research License: §2a solo uso non commerciale** (trovato dalla funzione 2) | fino a 10 | Va bene per le prove; per produrre un gioco commerciale va deciso a parte |
| Qwen-Image-Edit-2511 | Apache-2.0 | più immagini | **Fuori dal progetto per decisione dell'utente.** Resta la strada Apache-2.0 se la licenza della 2.1 diventa un problema (in `sd-cli` serve `--model-args qwen_image_zero_cond_t=true`) |
| FLUX.2 klein 4B | Apache-2.0 | fino a 4 | Non adottato. Circa 13 GB di VRAM, supportato da `sd-cli`; il 9B ha licenza non commerciale |
| FLUX 3 Image Klein (uscito oggi, 15/10/2026) | FLUX Kommunity License: commerciale gratis solo sotto 5 M$ di fatturato, con condizioni su filtri e ridistribuzione | sì | Da guardare, non da adottare ora |
| Qwen-Image-Style-Transfer (QwenStyle, LoRA su Edit-2509) | codice Apache-2.0, **licenza dei pesi non dichiarata** | stile + contenuto | Escluso finché i pesi non hanno una licenza: ripristinerebbe lo stile di un concept venuto male |

## 4. Misurare lo stile: una gara fra misure imparate

Niente controlli sui pixel. Tre candidati, tutti modelli addestrati; **ognuno entra solo se passa la prova sui giudizi
dell'utente** (sezione 7: area ROC ≥ 0,90 e il resto). Se ne passa più d'uno, si usa il più economico, e il più costoso
solo nella zona incerta fra `warn` e `fail` (a cascata).

### 4.1 SigLIP 2 (embedding generale)

Ogni immagine diventa un vettore; il punteggio è la somiglianza media con le k = 3 immagini approvate più vicine (prima
quelle dello stesso tipo). Le soglie di partenza si calcolano dalla bibbia stessa ("lascia fuori uno": quanto si
somigliano fra loro le immagini buone), poi si tarano sui giudizi. Il modello sa distinguere "foto" e "dipinto", ma il
soggetto pesa molto: un lupo realistico può somigliare a un lupo approvato.

### 4.2 CSD (embedding addestrato apposta per lo stile)

Contrastive Style Descriptor ("Measuring Style Similarity in Diffusion Models", 2024): un ViT-L partito da CLIP e
addestrato a separare lo stile dal contenuto. Nella letteratura del 2026 è ancora il riferimento: non ho trovato un
embedding di stile più recente con pesi aperti che lo batta in modo dimostrato. Un articolo di maggio 2026
([arXiv 2605.09030](https://arxiv.org/abs/2605.09030)) mostra che il coseno grezzo non va letto come punteggio assoluto
(area 0,883 nella verifica a coppie su 91 artisti) e che una lettura relativa (CSLS: si confronta la somiglianza con
quella dei vicini tipici) porta a 0,905. Uso CSLS. Avvertenza dal README del CSD: i pesi caricati non riproducono del
tutto i numeri dell'articolo. Lo dice la nostra prova.

### 4.3 Giudice visione-linguaggio con griglia di domande (Qwen3-VL-8B)

Al modello si danno 3 immagini approvate (da `refs.pick`) e il concept, e le domande di `style.criteria`, una alla
volta, con risposta sì/no. Non si legge la parola scritta ma la **probabilità del "sì"** (dai logit): un numero
continuo, che si può tarare e mettere su una curva ROC. Il punteggio è la media sulle domande, ripetuta con due terne
di riferimenti diverse e mediata. Il messaggio dice quale domanda è andata male ("colours: 0.12").

Perché così e non "dai un voto da 1 a 10": la ricerca del 2026 dice che i giudici visione-linguaggio **ordinano bene ma
danno voti assoluti poco affidabili** (accordo esatto con le persone su una scala a 5 punti circa un terzo delle volte,
[arXiv 2604.25235](https://arxiv.org/abs/2604.25235)), che ripetere il giudizio con semi o riferimenti diversi lo rende
più stabile, e che le immagini di riferimento possono spostare il giudizio. Domande sì/no, probabilità invece del voto,
due terne di riferimenti e soglie tarate sui giudizi dell'utente sono le contromisure.

Il modello: Qwen3-VL-8B-Instruct (Apache-2.0). **Il file del modello è già sul disco** (`Qwen3VL-8B-Instruct-Q4_K_M.gguf`,
5,03 GB, lo usa Qwen-Image come codificatore); manca la parte visiva (`mmproj`, 1,16 GB), la stessa che serve alla
strada edit della funzione 2: la aggiunge **la funzione 2** al catalogo (nel backend qwen-image); io la leggo da lì,
senza duplicarla. Gira con **llama.cpp** (MIT, supporta Qwen3-VL), in
`vendor/llama-cpp/` come `sd-cli` sta in `vendor/sdcpp/`. Usa la GPU: lucchetto condiviso. Tempo per concept da
misurare nella prova (stima: decine di secondi).

Successori considerati: Qwen3.5 ha visione integrata e pesi Apache-2.0, ma non c'è un GGUF ufficiale verificato per
llama.cpp e non abbiamo nulla già sul disco. Si cambia solo se Qwen3-VL non passa la prova.

### 4.4 Imparare dai giudizi dell'utente

Le decisioni dell'utente (approvato nella bibbia, rifiutato con nota) sono dati. Quando ci sono almeno 10 approvati e
10 rifiutati, si addestra un piccolo classificatore lineare (regressione logistica, in numpy) sugli embedding del
vincitore fra 4.1 e 4.2. Entra solo se, con validazione "lascia fuori uno", batte il punteggio dei vicini. Questo è il
modo in cui la misura migliora con l'uso. È una seconda PR, dopo la prova.

### 4.5 Contratto e registrazione

- `image_to_3dlab/style/score.py: judge_concept(image, bible, kind=None) -> dict`: un risultato `style.coherence` nel
  contratto della funzione 7 (`check`, `status`, `value`, `threshold`, `message`, `details` con metodo, modello e
  licenza, vicini o risposte alle domande). **Non serve un GLB**: è il cancello prima del 3D.
- Registrazione nei controlli della funzione 7: `style.coherence` con `needs=("reference_shape",)` legge il concept da
  `Asset.reference` quando è un'immagine. Nel profilo `default.json` è **spento** (`"enabled": false, "bible": null`);
  un lotto lo accende passando la bibbia negli `overrides`. Acceso senza bibbia → `fail` con il motivo.
- **Lotti (funzione 1)**: un passo `style` **prima** di `generate` (adattatore `image_to_3dlab/batch_steps/style.py`,
  `preflight` che verifica bibbia, motori e pesi senza scaricare). `fail` ferma la riga; `warn` la lascia andare e la
  segna. Il migliore di N vive qui quando la riga parte da un soggetto e non da un'immagine.
- **CLI**: `scripts/style_bible.py judge <bibbia> concept.png [--kind creature]` stampa il JSON; uscita 0 per
  pass/warn, 2 per fail.

## 5. Download e motori (ognuno solo dopo un sì esplicito)

Struttura di AssetFurnace: motori in `vendor/`, pesi nel catalogo (`viewer/backend_catalog.py`) con nome, dimensione e
licenza, e uno script di installazione (`scripts/bootstrap_style.py`) che chiede conferma; `--yes` per chi lo usa da
agente. Niente ComfyUI.

| Cosa | File | Dimensione | Licenza | Serve per |
|---|---|---|---|---|
| SigLIP 2 base/16 224 | `google/siglip2-base-patch16-224` → `model.safetensors` | 1,5 GB | Apache-2.0 (scheda; big_vision: "including models") | 4.1 |
| CSD ViT-L | `tomg-group-umd/CSD-ViT-L` → `pytorch_model.bin` | 2,44 GB | CC-BY-4.0 (attribuzione), derivato da OpenAI CLIP ViT-L/14 (MIT) | 4.2. È un file pickle: si carica con `torch.load(weights_only=True)` |
| Qwen3-VL-8B-Instruct Q4_K_M | `Qwen/Qwen3-VL-8B-Instruct-GGUF` | 5,03 GB, **già in cache** | Apache-2.0 | 4.3 |
| mmproj di Qwen3-VL-8B | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,16 GB, **catalogato dalla funzione 2** (non lo duplico) | Apache-2.0 | 4.3 |
| llama.cpp | release `b11669` (15/10/2026): Vulkan x64 | 30 MB; con CUDA 13.4: 147 MB + 403 MB di runtime | MIT | 4.3. Si parte con Vulkan; CUDA solo se la prova lo mostra troppo lento |
| `transformers` (pip) | — | piccolo | Apache-2.0 | Carica SigLIP 2 e CSD (torch c'è già) |

Per la gara: circa **4 GB** di pesi nuovi miei (SigLIP 2 e CSD) più il mmproj della funzione 2 e 30 MB di motore. Dopo la gara si tengono solo quelli del vincitore; il
catalogo permette di togliere gli altri.

**Esclusi**, con il motivo: DINOv3 (licenza con condizioni); Cell-DINO e XRay-DINO (non commerciali, nella stessa repo
di DINOv2); DINOv2 (Apache-2.0, ma guarda la forma più dello stile, e nell'articolo del CSD resta sotto il CSD); OpenAI CLIP
puro (la scheda esclude ogni uso distribuito); OpenCLIP su LAION (LAION-5B fu ritirato nel 2023 per contenuti
illegali); Qwen3-VL-Embedding-2B (Apache-2.0, gennaio 2026, ma nessun risultato pubblicato sullo stile); tutto ciò che
è Hunyuan/Tencent.

## 6. Nello studio: la pagina "Stile"

Un pulsante "Style" nella barra, come "Review", che apre un foglio sopra il visore (file miei:
`viewer/studio/style.js`, `viewer/studio/style-model.js` per la logica pura e testata, `viewer/style_api.py`).

- In alto: bibbia attiva, nuova bibbia, registra una cartella esterna.
- **La bibbia**: approvate e rifiutate, con nota e tipo; togli (con conferma); modifica nota.
- **Aggiungi**: da una scheda della Libreria (le immagini di `output/images/`) o da un caricamento; nota obbligatoria.
- **Concept recenti**: le ultime immagini generate con il giudizio (verde/giallo/rosso e il motivo). Da qui, un clic le
  approva o le rifiuta nella bibbia: così i giudizi dell'utente si accumulano mentre lavora.
- **Prompt**: anteprima del prompt composto per tipo, con i riferimenti che userebbe.

Righe di aggancio nei file condivisi, minime: `generate_api.py` (import e 2 righe di instradamento, come
`review_api`), `studio.html` (1 pulsante), `app.js` (import e 1 ascoltatore).

## 7. Come provo l'accettazione

"Su un campione etichettato il giudizio separa i concept che l'utente approverebbe da quelli che rimanderebbe." Le
etichette sono **dell'utente**, non mie.

1. **Bibbia di esempio**, fuori da git (`output/style-acceptance-20261016/bible/`): circa 24 concept generati con i
   modelli di prompt, nei 5 tipi; l'utente ne approva 16 o più dalla pagina Stile (5 minuti). Si aggiungono i concept e
   i modelli già in `output/`.
2. **Campione di prova**, con soggetti diversi: circa 40 concept fra quelli generati nei tre bracci della sezione 3,
   più alcuni costruiti per uscire dallo stile (fotorealistici, troppo scuri, colori spenti come il lupo mannaro, un
   altro stile) e negativi difficili (stesso soggetto e stesse parole di stile, ma cupi o spenti). L'utente li etichetta
   approvato/rimandato, senza vedere i punteggi.
3. **Numeri dichiarati** per ciascun candidato (SigLIP 2, CSD con CSLS, giudice Qwen3-VL) e per la cascata: area ROC;
   alla soglia scelta, quota dei rimandati fermati e dei buoni fermati per sbaglio; tempo per concept.
4. **Regola d'ingresso**: area ROC ≥ 0,90 e, alla soglia scelta, almeno l'80% dei rimandati in `warn` o `fail` con al
   massimo il 10% dei buoni. Chi non passa non entra. Se nessuno passa, lo scrivo e la funzione si ferma alla bibbia e ai
   riferimenti, senza un controllo che finge di funzionare.
5. **Stile all'origine**: per i tre bracci, quota approvata dall'utente e tempo per immagine.
6. Tutto resta in `output/style-acceptance-20261016/` con il rapporto; nel documento metto i numeri e gli errori
   peggiori di ogni candidato.

Avvertenza: 40 etichette danno un'area ROC con un margine d'errore ampio (circa ±0,05). Basta per scegliere ed
escludere; la misura vera si raffina con i giudizi del pilota (sezione 4.4).

### 7.1 Risultato della prova (16/10/2026)

**Avvertenza, scoperta dopo la gara:** il campione r1b è stato generato con la ricetta sbagliata: il modello base di
Qwen-Image 2.1 con i parametri del Turbo (8 passi, cfg 1,0). Molte immagini sono velate, con forme "fantasma". I numeri
qui sotto sono quindi falsati due volte, dalla qualità delle immagini oltre che dalle punte. La ricetta giusta è il
Turbo Q8 (`--sigmas qwen-image-2.1-turbo`, cfg 1,0, euler). Il secondo campione (r1c) usa quella, a 1024 px come
r1b (a 1536 il Turbo aggiunge cose non richieste attorno al soggetto), con gli stessi soggetti, semi e riferimenti,
così il confronto regge.

Bibbia del primo progetto pilota (fuori da git): 32 immagini di riferimento originali del progetto, solo per uso
interno, e 2 approvate. Campione: 75 concept, 25 soggetti nei 5 tipi × 3 bracci, etichettati dall'utente alla cieca
(l'id non diceva il braccio): **41 approvati, 34 rimandati**. Rapporto e punteggi in
`output/style-acceptance-20261016/race-r1b-*.json`.

| Candidato | Area ROC (95%) | Rimandati fermati | Buoni fermati | s/concept | Entra |
|---|---|---|---|---|---|
| CSD, k vicini | 0,43 (0,31-0,58) | 3% | 10% | 0,50 (CPU) | no |
| CSD, CSLS | 0,43 (0,30-0,56) | 6% | 7% | 0,50 (CPU) | no |
| SigLIP 2, k vicini | 0,43 (0,31-0,56) | 9% | 10% | 0,14 (CPU) | no |
| SigLIP 2, CSLS | 0,37 (0,25-0,50) | 6% | 7% | 0,14 (CPU) | no |
| Giudice Qwen3-VL-8B | 0,66 (0,54-0,78) | 35% | 10% | 1,66 (GPU) | no |

**Nessun candidato passa la regola d'ingresso.** Come detto sopra, la funzione per ora si ferma alla bibbia, ai modelli
di prompt e ai riferimenti; `style.coherence` (PR 3) non si fa finché una misura non passa.

Perché, dagli errori peggiori: i concept rimandati erano quasi tutti **nello stile giusto ma sbagliati nel contenuto**
(punte e armature su pozioni, carri, lupi; una bancarella diventata un guerriero), per colpa di una frase della
descrizione generale poi spostata nelle aggiunte di personaggio ed equipaggiamento. Il giudice dà 0,98-0,99 a
oggetti ed equipaggiamenti rimandati, perché alle domande sullo stile la risposta giusta è "sì". Gli embedding stanno
sotto il caso: i rimandati, pieni di armature, somigliano ai riferimenti più degli approvati. Sulle sole creature (1
approvata su 15) separano, ma è un'immagine sola.

Stile all'origine (punto 5), quota approvata per braccio: solo prompt 13/25, +1 riferimento 15/25, +3 riferimenti
13/25; per tipo (solo prompt/+1/+3) personaggi 4/4/2, creature 1/0/0, equipaggiamento 3/3/3, oggetti di mappa
1/4/4, oggetti 4/4/4. Un riferimento aiuta soprattutto gli oggetti di mappa; tre non aggiungono altro.

### 7.2 Secondo campione (r1c) ed esito della funzione (16-17/10/2026)

Stessi 25 soggetti, semi e riferimenti di r1b, con la ricetta Turbo di main (1024 px) e l'armatura tolta dalla
descrizione generale. 75 concept etichettati alla cieca dall'utente: **61 approvati**.

| Braccio | Approvati | Personaggi | Creature | Oggetti | Equipaggiamento | Oggetti di mappa |
|---|---|---|---|---|---|---|
| Solo prompt | 22/25 | 4/5 | 4/5 | 5/5 | 4/5 | 5/5 |
| +1 riferimento | 21/25 | 4/5 | 2/5 | 5/5 | 5/5 | 5/5 |
| +3 riferimenti | 18/25 | 3/5 | 0/5 | 5/5 | 5/5 | 5/5 |

Osservazioni dell'utente: gemme su quasi tutte le immagini, un lupo "modificato male", un drago simile al lupo,
immagini troppo simili fra loro. Le cause, verificate:

- **I riferimenti passano anche il contenuto.** Tutte le creature ricevono gli stessi 3 riferimenti (`refs.pick` per
  tipo), e la modifica di Qwen-Image ne copia la testa e il guscio: lupo, ragno, cinghiale e drago diventano varianti
  della stessa bestia. I personaggi con 3 riferimenti ne prendono cappuccio e viso. Più riferimenti, meno approvati.
- **Parole che diventano oggetti.** Ogni prompt conteneva colori scritti come materiali ("emerald accents") e i
  personaggi anche "gemstone inlays" e "spikes": il modello le disegna come gemme e punte su tutto.

Decisioni dell'utente (17/10/2026):

1. **Niente riferimenti** nei concept nuovi: lo stile lo porta il prompt. I riferimenti restano per la funzione 2.
2. **Descrizione riscritta**: fantasy colorato e solare da MMORPG classico, colori accesi e saturi, luce calda e piena
   (senza ombre portate, per la funzione 8), forme massicce e stilizzate con proporzioni un po' esagerate, pittura a
   mano con pennellata visibile. Nessun marchio. Nella parte generale i colori sono scritti come colori, senza parole
   che diventano oggetti (niente gemme, smeraldi, punte): compaiono solo se li chiede il soggetto. Armature solo nelle
   parti di personaggio ed equipaggiamento.
3. **Il giudice automatico è rimandato**, come il LoRA. Il codice della gara e dei tre giudici resta su main, spento:
   nessun controllo `style.coherence`, nessun nuovo giro di etichette. Il motivo: con una descrizione giusta e senza
   riferimenti 22 concept su 25 escono buoni, e gli scarti dipendono dal contenuto (cosa è disegnato), che i giudici di
   stile non vedono; nessuno dei tre passava la regola su r1b, e un giudice sul contenuto sarebbe un altro progetto.
4. Prova di controllo, giudicata a occhio da me (l'utente non vuole altre valutazioni): i 25 soggetti di r1c
   rigenerati solo con il prompt, stessi semi, e confrontati uno per uno con quelli di r1c.

Le tre versioni della descrizione nuova (fuori da git, nella bibbia del progetto pilota):

- **r1d**, con i colori elencati per nome ("rich red, golden yellow, royal blue and leaf green"): gemme, punte e
  somiglianze sparite, ma i nomi diventano una tavolozza dipinta su tutto (lupo, ragno e cinghiale arcobaleno, orco
  giallo, stivali da pagliaccio) e "leaf green" diventa foglie su pozione, guanto, libro e corazza. Anche i colori, non
  solo i materiali, sono parole che diventano oggetti.
- **r1e**, senza nomi di colori ("vivid saturated colours, each subject in its own natural colours made brighter"):
  lupo, cinghiale, orco, cavaliere, scudo, forziere, pozzo e carro nei loro colori, solo più accesi.
- **r1f** = r1e più "a few harmonious colours per subject", e il metallo dell'equipaggiamento non più "painted": a
  parità di seme cambia poco. È la descrizione che resta.

Esito su r1f, rispetto a r1c: **gemme** solo dove il soggetto le chiede (la spada "with a jewelled hilt", il libro "with
a gem on the cover"); **punte** nessuna, e nessuna armatura su animali e oggetti; **somiglianze** sparite: lupo, ragno,
cinghiale, drago e golem sono bestie diverse, la bancarella è finalmente una bancarella. Limiti rimasti: colori da
caramella su 4 soggetti su 25 senza un colore proprio (ragno, stivali spaiati, guanto, cartello), effetto dei "colori
accesi e saturi" chiesti; il drago è un po' tenero. Immagini in `output/style-acceptance-20261016/r4..r6/`.

## 8. Le PR

1. **Bibbia e prompt** (nessun modello): `style/bible.py`, `style/prompts.py`, `style/refs.py`, i tre tipi nuovi nelle
   ricette, CLI `scripts/style_bible.py new|register|use|add|reject|remove|note|list|prompt|refs` (uno script,
   come `check_asset.py`, invece di un sottocomando di `pipeline.py`: `cli.py` resta della funzione 1), test.
2. **Gara**: `scripts/bootstrap_style.py` con il catalogo, i tre candidati, la prova con i numeri nel documento.
3. ~~Il vincitore come controllo~~: rimandato con il giudice (sezione 7.2).
4. ~~Pagina Stile nello studio~~: rimandata; la bibbia si gestisce con `style_bible.py`.
5. ~~`batch_steps/style.py` con il migliore di N, classificatore della 4.4~~: rimandati con il giudice.
6. **Chiusura**: descrizione nuova nella bibbia del progetto, riferimenti spenti per i concept nuovi, esito in 7.2.

## 9. Rischi

- **Il soggetto pesa più dello stile** negli embedding generali: è il motivo per cui ci sono CSD e il giudice, e per
  cui la prova ha negativi con lo stesso soggetto.
- **Il giudice è instabile o lento**: due terne di riferimenti, probabilità invece dei voti, soglie tarate; il tempo si
  misura, e la cascata lo usa solo nei casi incerti.
- **Poche etichette**: margine ampio sull'area ROC (sopra). Le etichette crescono con l'uso della pagina Stile.
- **I riferimenti fanno copiare** (la generazione ricalca un'immagine della bibbia invece di prenderne lo stile): la
  funzione 2 ha tolto il controllo sui bordi (`concept.copy`); l'unico segnale possibile è quello imparato
  (somiglianza DINOv2) della funzione 2, e lo si usa qui solo se supera la sua prova. Altrimenti la copia la vede
  l'utente nella pagina Stile, dove il concept è accanto ai riferimenti usati.
- **Licenza di Qwen-Image 2.1** (non commerciale): riguarda la generazione, non questa funzione, ma i riferimenti di
  stile la usano. L'utente ha scelto la 2.1; Qwen-Image-Edit-2511 resta la strada Apache-2.0 di riserva.
- **CSD**: pesi in formato pickle e risultati non del tutto riprodotti dagli autori. Caricamento sicuro e prova nostra.

## Seguiti

(vuoto: ci finiscono le osservazioni di revisione rimandate)

## Domande per l'approvazione

1. Va bene la gara fra SigLIP 2, CSD e il giudice Qwen3-VL, con circa 4 GB di pesi (SigLIP 2 e CSD; il mmproj lo scarica la funzione 2) e llama.cpp Vulkan (30 MB)?
   Scarico solo dopo il tuo sì.
2. Etichette dell'utente: circa 5 minuti per approvare la bibbia di esempio e 10 per etichettare il campione, dalla
   pagina Stile o da un foglio. Va bene?
3. ~~La prova dei riferimenti di stile dentro la gara della funzione 2~~: risolta con la funzione 2 (motore suo,
   bracci e misura miei).

## Fonti (verificate il 15/10/2026)

- Schede Hugging Face: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
  [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
  [siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224),
  [CSD-ViT-L](https://huggingface.co/tomg-group-umd/CSD-ViT-L),
  [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF),
  [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B),
  [dinov2-base](https://huggingface.co/facebook/dinov2-base).
- GitHub: [big_vision](https://github.com/google-research/big_vision) (Apache-2.0, modelli compresi),
  [CSD](https://github.com/learn2phoenix/CSD) (MIT), [dinov2](https://github.com/facebookresearch/dinov2),
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`docs/flux2.md`, `docs/qwen_image_edit.md`),
  [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, `src/models/qwen3vl.cpp`),
  [Qwen-Image-Style-Transfer](https://github.com/witcherofresearch/Qwen-Image-Style-Transfer).
- [The Complete Style Transfer Handbook](https://blog.comfy.org/p/the-complete-style-transfer-handbook) (Comfy Org,
  19/03/2026); [arXiv 2605.09030](https://arxiv.org/abs/2605.09030) (CSD e CSLS);
  [arXiv 2604.25235](https://arxiv.org/abs/2604.25235) (giudici visione-linguaggio: ordinano ma non danno voti);
  [FLUX 3 Image Klein](https://bfl.ai/blog/flux-3-image-klein) e
  [VentureBeat](https://venturebeat.com/technology/black-forest-labs-opens-flux-3-image-klein-weights-with-free-commercial-use-limited-to-smaller-businesses)
  sulla licenza.
