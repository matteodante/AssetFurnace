# Funzione 6: approcci più recenti per animare, riggare e riconoscere le creature (ricerca, 16/10/2026)

Richiesta del coordinatore, a nome dell'utente: "o facciamo qualcosa di fatto bene o non lo facciamo per niente".
Prima di andare oltre la PR #13 ho cercato cosa esiste a ottobre 2026 su tre fronti. Per ogni candidato ho verificato:
- la licenza del codice;
- la licenza dei pesi, compresi quelli che il codice scarica da solo (letta dall'API di Hugging Face e dal codice);
- dove gira.

I repository sono stati clonati e letti, mai eseguiti; non ho scaricato pesi. Piano di riferimento:
[`2026-10-15-creature-families-design.md`](2026-10-15-creature-families-design.md).

**In una riga**: nessun modello aperto e commerciale oggi genera animazioni per scheletri qualsiasi. La strada matura è
una libreria di clip CC0 per famiglia (Mesh2Motion, Quaternius), adattate allo scheletro di SkinTokens attraverso i
ruoli che la #13 già riconosce. SkinTokens resta il rigger. Un giudice visione-linguaggio può aiutare a riconoscere la
famiglia, ma solo come voto in più, non per contare zampe.

## 1. Animazione delle creature

Ai mostri del gioco servono attesa, camminata, corsa, attacco, colpo subito e morte. Le clip procedurali della #8 e della
#13 provano il rig, non bastano per il gioco.

### Modelli che generano movimento per scheletri qualsiasi

| Candidato | Cosa fa | Codice / pesi / dati | Dove gira | Esito |
|---|---|---|---|---|
| **UniMate** (SIGGRAPH Asia 2026, 1,8k stelle, attivo) | Da testo a movimento per qualsiasi rig: quadrupedi, uccelli, insetti, serpenti, piante | MIT / **CC-BY-NC-4.0** (`Linzhan/UniMate`, verificato) / Mixamo, Objaverse-XL, Truebones | Linux, torch 2.5 cu124: da ricompilare per la 5070 Ti | **Escluso** per i pesi. Il più promettente: da riguardare se esce un checkpoint commerciale |
| **AnyTop** (SIGGRAPH 2025) | Diffusione su topologie qualsiasi, modelli per bipedi, quadrupedi, serpenti, volanti | MIT / **solo ricerca** (`Inbar2344/AnyTop`, scaricato in automatico da `utils/hf_handler.py`, verificato) / Truebones | Python 3.8, torch 2.4: WSL2 e ricompilazione | **Escluso** così com'è |
| **Puppeteer**, animazione | Adatta il movimento delle ossa a un video dato | Apache / Apache, ma scarica **CoTracker3** e **Video-Depth-Anything-Large**, entrambi CC-BY-NC | torch 2.1 cu118, WSL2 | **Escluso** (già nel progetto) |
| **SATA**, **OmniZoo** | Da testo a movimento per animali | Dati estratti dal videogioco *Planet Zoo* | – | **Esclusi** |
| **GenMM** (Apache-2.0) | Variazioni e raccordi a partire da una sola clip, su qualsiasi scheletro, senza addestramento | Apache / nessun peso / dati nostri | CPU, Windows, ha un componente Blender | **Utile** per variare le clip della libreria |
| **SinMDM** (MIT) | Diffusione addestrata su una sola clip | MIT / i suoi modelli su Google Drive non hanno licenza | da ricompilare | Forse, addestrato sulle nostre clip CC0 |
| SkelMo, SAMoR, TopoCap, AnimaX | Articoli 2026 | codice non pubblicato | – | Da riguardare |
| MoMask, MDM, T2M | Solo umanoidi, addestrati su HumanML3D/AMASS (non commerciali) | – | – | Non pertinenti |

### Clip e dataset con licenza chiara

- **Mesh2Motion** (3,5k stelle, attivo a ottobre 2026): il README dichiara **codice MIT e animazioni CC0**. GitHub non
  riconosce in automatico il file di licenza: da rileggere prima dell'uso. Ha modelli animati per quasi tutte le nostre
  famiglie:
  - volpe: morso, morte, attesa, corsa, camminata, salto;
  - cavallo: calcio, morte, corsa, trotto;
  - kaiju, un bipede con coda: attacco, colpo subito, morte, ruggito, colpo di coda, camminata;
  - ragno: attacco, morso, colpo subito, morte, camminata;
  - serpente: morso, colpo subito, morte, attesa;
  - drago: volo, planata, attesa, camminata;
  - uccello: battito d'ali, planata, attesa, camminata;
  - umano: serie ampia.

  Mancano solo le piante.
- **Quaternius** (Animated Monsters, Easy Enemies): CC0, con clip di movimento, attacco e morte.
- **Truebones Zoo**: motion capture di animali, "royalty free" anche per uso commerciale, ma vieta di ridistribuire i
  file grezzi; nulla trovato sull'addestramento di modelli. Utilizzabile come sorgente di clip (a pagamento), non
  verificato per addestrare.
- **Non utilizzabili**: DeformingThings4D e AnimalML3D (non commerciali), la motion capture di cani di AI4Animation
  (CC-BY-NC), Mixamo per addestrare (termini Adobe non verificati).

### Retarget fra scheletri diversi

- **Blender, script `bpy`** (vincoli di copia della rotazione per catena, poi cottura): maturo, nessun effetto sulla
  licenza di ciò che esce. È il modo per portare le clip CC0 sullo scheletro di SkinTokens dentro AssetFurnace.
- **IK Retargeter di Unreal**: maturo, funziona con qualsiasi topologia una volta definite le catene. Utile se il
  retarget si fa nel gioco.
- **Motion2Motion** (SIGGRAPH Asia 2025): trasferimento fra topologie diverse da poche corrispondenze fra ossa, su CPU.
  **Il repository non ha licenza**: si può solo reimplementare dall'articolo, sopra GenMM (Apache-2.0).
- **CAST** (2026): dal video a qualsiasi scheletro, codice e pesi MIT, torch 2.7 cu128 (va sulla 5070 Ti). Nuovissimo,
  una stella, addestrato anche su Objaverse con licenze per oggetto: **da tenere d'occhio**.
- SAN, R2ET, WalkTheDog: solo umanoidi o fermi. SAME: CC-BY-NC.

### Cosa è maturo e cosa no

- **Maturo**: clip CC0 fatte a mano (Mesh2Motion, Quaternius); retarget a catene in Blender o in Unreal; variazioni su
  esempio (GenMM).
- **Non maturo per un gioco commerciale**: generazione da testo per scheletri qualsiasi. Esiste ed è vicina (UniMate),
  ma i pesi sono non commerciali, e anche gli autori la definiscono "un primo passo".

## 2. Rig automatico

Nessun rigger **rilasciato e utilizzabile commercialmente** fa meglio di SkinTokens.

| Candidato | Creature non umane? | Codice / pesi | Esito |
|---|---|---|---|
| **SkinTokens** (attuale) | sì | MIT / MIT, Windows | Resta il rigger |
| UniRig | sì | MIT / MIT, WSL2 | Riserva (secondo gli autori di SkinTokens è peggiore) |
| **RigAnything** (Adobe) | sì | **Adobe Research License, non commerciale** | Escluso |
| **Anymate** | sì | codice senza licenza / pesi Apache, ma carica **PointLLM, CC-BY-NC** | Escluso |
| MagicArticulate | solo scheletro | Apache / Apache + Michelangelo LGPL, torch 2.1 | Solo scheletro, peggiore secondo gli autori di SkinTokens |
| **AniGen** (VAST, 2026) | da immagine a modello già riggato | MIT / MIT, ma carica **DSINE, non commerciale**; Linux, almeno 18 GB | Escluso |
| RigMo | – | CC-BY-NC, solo addestramento | Escluso |
| Make-It-Animatable, HumanRig, TARig | solo umanoidi | – | Non pertinenti |
| **PlantRig** (agosto 2026) | piante | MIT, ma niente pesi propri | Idea riusabile, vedi sotto |
| Rigel3D, ARMO, Animator-Centric (CVPR 2026), Branch-Centric | sì | codice non pubblicato | Da riguardare nel primo trimestre 2027 |

Tre idee da articoli recenti si applicano a SkinTokens senza cambiare modello:
- **Radice imposta** (PlantRig): si inizia la generazione dal punto più basso della parte spessa. Risolve la pianta
  radicata nel fiore. Da provare: SkinTokens genera token dopo token come il modello di PlantRig.
- **Più tentativi e copie ruotate** (Branch-Centric): SkinTokens ha già i parametri di campionamento. Si genera più
  volte, anche su copie ruotate del modello, e si tiene lo scheletro che arriva a tutti i piedi senza catene doppie. I
  controlli per scegliere sono quelli geometrici che la #13 ha già.
- **Serpente avvolto**: una colonna procedurale lungo l'asse del corpo, quando SkinTokens copre una sola spira.

## 3. Riconoscimento della famiglia con un giudice visione-linguaggio

- **Motore**: il ramo `claude/feat-style-judge` (bibbia dello stile) usa già Qwen3-VL-8B con **llama.cpp
  `llama-server`** (MIT, versione Vulkan per Windows, 32 MB, scaricata dopo conferma) e i file GGUF che AssetFurnace ha
  già (modello e file della visione). Pone domande sì/no e legge la probabilità del "sì". Il riconoscimento della
  famiglia può **riusare quello stesso motore**, senza aggiungerne un secondo e senza WSL2. Memoria stimata, non
  misurata: circa 8 GB, quindi in un passo a sé.
- **Debolezza nota e misurata**: i modelli visione-linguaggio **contano male le zampe**. Su animali modificati con una
  zampa in più la precisione è del 2% ("Vision Language Models are Biased", ICLR 2026): rispondono con il numero tipico
  della specie. Sono buoni a dire "che tipo di creatura", deboli proprio dove servono le regole (un cane a sei zampe, un
  drago senza ali).
- **Licenze dei modelli alternativi** (API di Hugging Face): Qwen3-VL in tutte le taglie, InternVL3.5, MiniCPM-V 4.5,
  Gemma 4 e SigLIP 2 sono Apache-2.0. **Gemma 3 no** (termini Gemma, accesso su approvazione). InternVL3-8B rimanda anche
  a una licenza Qwen da leggere.
- **Uso proposto, insieme alle regole e mai al loro posto**:
  - si fanno 4 render puliti del modello (davanti, lato, sopra, tre quarti) e si pongono domande sì/no per ogni famiglia
    e per la presenza di ali, zampe e camminata sulle dita;
  - se regole e giudice concordano, la famiglia è proposta così;
  - se le regole sono sicure e il giudice no, vince la regola e la persona vede il disaccordo;
  - se le regole sono incerte, il giudice sceglie quale famiglia proporre;
  - **il numero di zampe resta quello dello scheletro**;
  - la scelta finale resta della persona.
- **Prima di usarlo**: va misurato su un insieme etichettato di 60-100 rig con circa il 20% di casi strani, confrontando
  solo regole, solo giudice ed entrambi. Si accende solo se, quando concordano, hanno quasi sempre ragione. Oggi abbiamo
  11 rig veri: non bastano per decidere.

## Cosa cambia per la funzione 6 (da approvare)

1. **La #13 resta valida.** Il riconoscimento e i ruoli delle catene (zampe con lato e posizione, ali, coda, collo,
   corpo del serpente) sono proprio ciò che serve per adattare clip CC0 di un'altra creatura. Le clip procedurali
   restano come prova del rig e come riserva (piante, creature strane).
2. **Prossima PR proposta** (non aperta, aspetta l'utente): una libreria di clip CC0 per famiglia (Mesh2Motion, poi
   Quaternius) con un retarget a catene in Blender (`bpy`). Le corrispondenze vengono dai ruoli della #13: zampa
   anteriore sinistra con zampa anteriore sinistra, ala con ala, coda con coda. Le clip sono attesa, camminata, corsa,
   attacco, colpo subito e morte, controllate da `rig.deformation` e dalla coda di revisione. Download: solo i GLB CC0
   di Mesh2Motion, previa conferma, con nome e dimensione.
3. **Rig**: più tentativi di SkinTokens con scelta geometrica; radice imposta per le piante; colonna procedurale per i
   serpenti avvolti. Nessun nuovo modello.
4. **Riconoscimento**: voto del giudice Qwen3-VL riusando il motore della bibbia dello stile, dopo il suo merge e dopo
   la misura su un insieme etichettato.
5. **Scartati**:
   - UniMate e AnyTop: pesi non commerciali o solo ricerca;
   - Puppeteer per l'animazione: pesi CC-BY-NC scaricati in automatico;
   - SATA e OmniZoo: dati estratti da un videogioco;
   - RigAnything, Anymate, AniGen e RigMo: licenze non commerciali, dirette o nascoste;
   - Gemma 3: termini restrittivi.

   UniMate e CAST vanno riguardati quando uscirà un checkpoint commerciale.

## Fonti principali

- UniMate: <https://github.com/Friedrich-M/UniMate>, pesi <https://huggingface.co/Linzhan/UniMate> (cc-by-nc-4.0)
- AnyTop: <https://github.com/Anytop2025/Anytop>, pesi <https://huggingface.co/Inbar2344/AnyTop> (research-only)
- Mesh2Motion: <https://github.com/Mesh2Motion/mesh2motion-app> (MIT, animazioni CC0 dal README)
- GenMM: <https://github.com/liyuuuu98/GenMM> (Apache-2.0)
- Motion2Motion: <https://github.com/LinghaoChan/Motion2Motion_codes> (nessuna licenza)
- CAST: <https://github.com/fankewen/CAST>, pesi <https://huggingface.co/CAST-mocap/CAST> (MIT)
- RigAnything: <https://github.com/Isabella98Liu/RigAnything>; Anymate: <https://github.com/yfde/Anymate>;
  AniGen: <https://github.com/VAST-AI-Research/AniGen>; PlantRig: <https://arxiv.org/abs/2608.01072>
- Conteggio delle zampe nei modelli visione-linguaggio: <https://arxiv.org/abs/2505.23941>
- Motore del giudice: llama.cpp b11669, <https://github.com/ggml-org/llama.cpp/releases/tag/b11669>
