# Funzione 3: bibbia dello stile (progetto)

Data: 15 ottobre 2026. Stato: **proposta, in attesa di approvazione**. Niente è implementato.
Piano di riferimento: [`2026-10-15-auroconts-asset-pipeline.md`](2026-10-15-auroconts-asset-pipeline.md), funzione 3.
Il LoRA dello stile è **fuori** da questo lavoro (rimandato dall'utente il 15/10/2026).

In breve: una cartella che contiene le immagini approvate di un progetto con le loro note, i modelli di prompt per tipo di
asset, e un punteggio che dice quanto un concept nuovo somiglia a quelle immagini. Il punteggio esce nel formato dei
controlli della funzione 7, così lo studio lo mostra e i lotti lo usano come soglia.

Perché serve, con un esempio già sul disco: il lupo mannaro di `output/compare-20261015/` è stato chiesto con
"stylized fantasy … hand-painted fantasy MMORPG monster" ed è uscito quasi realistico, grigio e spento. **Le parole del
prompt non bastano a garantire lo stile**: serve una misura che lo dica prima di spendere 15 minuti nel 3D.

## 1. Il formato della bibbia

Una bibbia è **una cartella** autosufficiente. Si copia, si mette sotto git nella repo del gioco, si apre da un altro PC.

```
auroconts-style/
  bible.json
  images/
    3f9a1c2e-knight.png
    3f9a1c2e-knight.provenance.json     # copiata dall'originale: la licenza viaggia con l'immagine
    b71d0e44-chest.png
  .cache/                               # rigenerabile, da ignorare in git
    embeddings-siglip2-base-patch16-224.npz
    scores.json
```

`bible.json`:

```json
{
  "schema_version": 1,
  "name": "auroconts",
  "description": "Faithful and stylised: saturated colours, slightly exaggerated proportions, showy armour.",
  "style": {
    "wording": "stylised hand-painted fantasy game art, bold saturated colours, slightly exaggerated proportions, chunky ornate armour, clean readable shapes",
    "avoid": "photorealistic, photo, dark, gloomy, muted colours, desaturated"
  },
  "kinds": {
    "creature": {"wording": "oversized claws and head, vivid fur colours"},
    "equipment_part": {"wording": "large decorative trims, gold and coloured enamel"}
  },
  "images": [
    {
      "file": "images/3f9a1c2e-knight.png",
      "sha256": "3f9a1c2e…",
      "kind": "character",
      "note": "Colours and proportions are exactly right; the shoulder pads are the target size.",
      "added": "2026-10-16T09:12:00+00:00",
      "source": "images/stylized-fantasy-knight-hand-painted-gam__20261015-192236/….png"
    }
  ],
  "score": {"model": "siglip2-base-patch16-224", "k": 3, "warn": null, "fail": null}
}
```

- **Aggiungere** un'immagine: la si copia in `images/` con le prime 8 cifre del suo hash nel nome (niente doppioni,
  niente collisioni di nomi), si copia accanto la sua `.provenance.json` se c'è, si scrive la voce con la nota.
- **Togliere**: si cancella la voce e il file (con conferma nello studio). Il file sparisce, la bibbia resta coerente.
- La nota è obbligatoria: senza un "perché va bene", fra sei mesi nessuno sa cosa difendere.
- `kind` è facoltativo: un'immagine senza tipo vale per tutti i tipi.
- `score.warn` e `score.fail` vuoti significano "calcolate dalla bibbia stessa" (sezione 3); un numero li fissa.
- **Più bibbie**: AssetFurnace tiene un elenco in `output/styles/bibles.json` (`[{"name", "path"}]`). Una bibbia nuova
  nasce in `output/styles/<nome>/`; una cartella esterna (per esempio nella repo di Auroconts) si **registra** con il
  suo percorso, senza copiarla. Una è quella attiva nello studio; la CLI e i lotti la nominano con `--style-bible`.

Il codice: `image_to_3dlab/style/bible.py` (leggere, validare, aggiungere, togliere; scrittura atomica del JSON),
senza dipendenze nuove. Controlla che ogni file elencato esista e abbia il suo hash.

## 2. I modelli di prompt per tipo di asset

Due livelli, ciascuno nel suo posto:

1. **`image_to_3dlab/prompt_recipes.json`** (generico, vale per ogni gioco): cosa serve al **3D**, non allo stile. Oggi
   ha `character` e `prop`; aggiungo `creature`, `equipment_part` e `map_object` con `wording`, `tips` e le parole a
   rischio (`trouble`) adatte. Esempio per `equipment_part`: "single piece of armour shown on its own, not worn,
   front view, whole piece in view, plain light grey background, soft even studio lighting".
   I 5 tipi: personaggio (`character`), creatura (`creature`), oggetto (`prop`), pezzo di equipaggiamento
   (`equipment_part`), oggetto di mappa (`map_object`). Corrispondenza con i profili dei controlli della funzione 7:
   character e creature → `creature`, prop e map_object → `prop`, equipment_part → `equipment_part`.
2. **La bibbia** (specifica del progetto): lo stile, `style.wording` più `kinds.<tipo>.wording`.

Il prompt finale si compone sempre nello stesso ordine, in una funzione pura testata:

```
<soggetto scritto dall'utente>, <wording del tipo dalle ricette>, <wording dello stile>, <wording del tipo nella bibbia>
```

`image_to_3dlab/style/prompts.py: compose(subject, kind, bible=None, recipes=None) -> {"prompt", "negative", "parts"}`.
La usano la CLI (`pipeline.py style prompt …`), i lotti (una riga con `kind` e `subject` al posto dell'immagine) e lo
studio (anteprima nella pagina Stile; nella pagina Crea una riga di aggancio che aggiunge lo stile della bibbia attiva).

**Attenzione al negativo**: con Qwen-Image 2.1 a `cfg 1.0` (il default misurato di AssetFurnace) il prompt negativo
**non ha effetto**. Quindi `style.avoid` va nel prompt negativo solo se si alza il cfg (il doppio del tempo), e lo stile
va detto in positivo. Lo scrivo nella pagina, non lo nascondo.

**Coordinamento con la funzione 2**: non tocco `viewer/image_api.py` né la generazione guidata. Se la 2 vuole lo stile
della bibbia, chiama `compose()`.

## 3. Il punteggio di coerenza

Due controlli nel contratto della funzione 7, perché misurano cose diverse e un messaggio chiaro vale più di un numero
unico:

### `style.coherence`: quanto il concept somiglia alle immagini approvate

- Ogni immagine diventa un vettore (embedding) con un modello d'immagine. Il punteggio è la **media della somiglianza
  coseno con le k immagini approvate più vicine** (k = 3; con un `kind`, prima quelle dello stesso tipo).
  Il vicino, non il centro: la bibbia contiene cavalieri e forzieri, e un forziere nuovo deve somigliare ai forzieri
  approvati, non a una media fra cavaliere e forziere.
- **Soglie calcolate dalla bibbia**: per ogni immagine approvata calcolo lo stesso punteggio contro le altre ("lascia
  fuori uno"). Questo dice quanto si somigliano fra loro le immagini buone. Sotto il 10° percentile → `warn`; sotto il
  10° percentile meno la distanza fra mediana e 10° percentile → `fail`. Così le soglie si adattano a ogni progetto e
  crescono con la bibbia. Dopo la prova di accettazione si possono fissare a numeri in `bible.json`.
- Con meno di 8 immagini nella bibbia il controllo dà al massimo `warn` ("bibbia troppo piccola per giudicare").
- Gli embedding della bibbia sono in cache (`.cache/`), indicizzati per hash dell'immagine e modello: aggiungere
  un'immagine ne calcola una sola.

### `style.palette`: colori e luce

L'embedding sa "dipinto o foto", ma è debole su "troppo scuro" o "colori spenti". Questo controllo è aritmetica pura,
niente modello: sul soggetto (lo sfondo uniforme si toglie col colore dei bordi), mediana di saturazione e di
luminosità, confrontate con la fascia delle immagini della bibbia. Messaggio leggibile: "saturation 0.18, approved
images 0.42–0.71". Questo prende il lupo mannaro anche se l'embedding lo trovasse "dipinto".

### Come si registra e dove si usa

- `image_to_3dlab/style/score.py: score_concept(image, bible, kind=None) -> list[dict]`: i due risultati nel contratto
  (`check`, `status`, `value`, `threshold`, `message`, `details` con i 3 vicini e il modello). **Non serve un GLB**:
  è il cancello prima del 3D.
- Registrazione nella funzione 7: `style.coherence` e `style.palette` con `@register(..., needs=("reference_shape",))`
  leggono il concept da `Asset.reference` quando è un'immagine. Nel profilo `default.json` sono **spenti**
  (`"enabled": false, "bible": null`): un lotto li accende passando la bibbia negli `overrides`. Acceso senza bibbia →
  `fail` con il motivo (chi lo accende lo vuole).
- **Lotti (funzione 1)**: il punto giusto è **prima** di `generate`, per non spendere il 3D su un concept fuori stile.
  Propongo alla funzione 1 un passo `style` (file adattatore `image_to_3dlab/batch_steps/style.py`, con `preflight`
  che verifica bibbia e pesi senza mai scaricare) da mettere prima di `generate` nell'elenco dei passi. `fail` ferma la
  riga con il messaggio; `warn` la lascia andare e la segna. Da concordare con quella sessione.
- **CLI**: `pipeline.py style score --bible <cartella> concept.png [--kind creature]` stampa il JSON del contratto;
  uscita 0 per pass/warn, 2 per fail (come `scripts/game_materials.py`).
- **Studio**: sezione 5.

## 4. Il modello di embedding

Licenze verificate il 15/10/2026 su Hugging Face e GitHub (codice **e** pesi, la lezione di OmniPart):

| Modello | Codice | Pesi | Download | Per noi |
|---|---|---|---|---|
| **SigLIP 2 base/16 224** (`google/siglip2-base-patch16-224`) | Apache-2.0 (big_vision: "including models") | **Apache-2.0** (scheda del modello) | **1,54 GB** (file unico con testo e immagine; si usa solo la parte immagine, ~93 M parametri) | **Scelto**. Addestrato su immagini e testo: distingue bene "foto", "dipinto", "cartone". Gira sul processore |
| DINOv2 base (`facebook/dinov2-base`) | Apache-2.0 | **Apache-2.0** ("code and model weights", README) | 346 MB | Alternativa e confronto. Ottimo sulla forma, più debole sullo stile perché ignora il significato |
| DINOv3 | — | Licenza DINOv3 (commerciale con condizioni) | — | Escluso: condizioni, e Pixal3D lo porta già |
| Cell-DINO, XRay-DINO | — | **Non commerciale** | — | Esclusi (sono nella stessa repo di DINOv2: attenzione a non confonderli) |
| CSD ViT-L (stile, da "Measuring Style Similarity…") | MIT | CC-BY-4.0 sulla scheda, ma derivato da CLIP e addestrato su LAION-Styles; il README dice che i pesi caricati non riproducono i numeri dell'articolo | non verificato | Escluso: catena di licenze dei pesi poco chiara |
| OpenAI CLIP | MIT | MIT, ma la scheda dice che ogni uso distribuito, commerciale o no, è "out of scope" | non verificato | Escluso per prudenza |
| OpenCLIP su LAION-2B | MIT | MIT | non verificato | Alternativa possibile, ma LAION-5B fu ritirato a fine 2023 per contenuti illegali (poi ripubblicato ripulito): preferisco evitarlo |

**Dipendenze nuove**: `transformers` (Apache-2.0) per caricare SigLIP 2. Torch c'è già (2.14 con CUDA nel `.venv`).
Niente `open_clip`, niente `timm`.

**Download**: per la regola della repo, nessun peso si scarica senza una scelta esplicita. Il comando
`pipeline.py style setup` dice nome, licenza e dimensione ("SigLIP 2 base, Apache-2.0, 1,54 GB") e chiede sì; `--yes`
per i lotti. Lo studio ha lo stesso pulsante con la stessa frase. Il punteggio, se i pesi mancano, dice come averli e
non scarica nulla.

**Dove gira**: sul processore (CPU) per default, circa mezzo secondo per immagine; la GPU resta libera per la
generazione. Opzione `--device cuda` (con il lucchetto della GPU) per i lotti grandi.

**Provenienza**: il punteggio non cambia l'asset, quindi non entra nella sua licenza; il modello e la sua licenza sono
scritti nei `details` di ogni risultato.

## 5. Nello studio: la pagina "Stile"

Un pulsante "Style" nella barra, come "Review", che apre un foglio sopra il visore (UI in file miei:
`viewer/studio/style.js`, `viewer/studio/style-model.js` per la logica pura e testata, `viewer/style_api.py`).

- In alto: scelta della bibbia attiva, nuova bibbia, registra una cartella esterna.
- **La bibbia**: griglia delle immagini con nota e tipo; togli (con conferma); modifica nota.
- **Aggiungi**: da una scheda della Libreria (le immagini di `output/images/`) o da un caricamento; nota obbligatoria.
- **Concept recenti**: le ultime immagini generate con il loro punteggio (verde/giallo/rosso, i due controlli e i 3
  vicini mostrati accanto). Calcolo lento la prima volta, poi in cache per hash.
- **Prompt**: anteprima del prompt composto per tipo (sezione 2).

Righe di aggancio nei file condivisi, minime: `generate_api.py` (import e 2 righe di instradamento, come
`review_api`), `studio.html` (1 pulsante), `app.js` (import e 1 ascoltatore), forse `create.js` (lo stile della bibbia
attiva aggiunto al prompt; da coordinare con la funzione 2 se tocca lo stesso file).

## 6. Come provo l'accettazione

"Su un campione etichettato il punteggio separa i concept che l'utente approverebbe da quelli che rimanderebbe."

1. **Bibbia di prova**: 16 concept nello stile, generati con i modelli di prompt (3-4 per ciascuno dei 5 tipi, semi
   diversi). In mancanza di concept già approvati dall'utente faccio io da approvatore e lo dichiaro; il foglio di
   contatto va all'utente, che può togliere quelli che non approverebbe.
2. **Campione di prova**, soggetti **diversi** da quelli della bibbia:
   - 15 nello stile (positivi);
   - 15 fuori stile (negativi): 5 fotorealistici, 4 troppo scuri, 4 con colori spenti (fra cui il lupo mannaro di
     `compare-20261015`), 2 di un altro stile (pixel art, cartone piatto).
   - In più qualche **negativo difficile**: stesso soggetto e stesse parole di stile, ma "at night, gloomy" o "muted
     earth tones". È il caso del lupo: il prompt dice stilizzato e l'immagine no.
3. Le etichette nascono dalla costruzione; le confermo guardando ogni immagine, e il foglio di contatto con etichette e
   punteggi va all'utente.
4. **Numeri dichiarati**: area sotto la curva ROC (1 = separazione perfetta, 0,5 = a caso) per SigLIP 2, DINOv2,
   palette e la combinazione; e alla soglia di default quanti negativi vengono fermati e quanti positivi fermati per
   sbaglio. **Obiettivo proposto**: area ≥ 0,90; alla soglia di default almeno l'80% dei fuori stile in `warn` o
   `fail`, al massimo il 10% dei buoni.
5. Il campione resta in `output/style-acceptance-20261016/` (non in git, sono nostri contenuti) con il rapporto; nel
   documento metto i numeri e le immagini peggiori di ogni gruppo.

Tempo GPU: circa 50 immagini con Qwen-Image a 768 px e 8 passi; con il lucchetto della GPU condivisa.

## 7. Le PR

1. **Bibbia e prompt** (nessun modello): `style/bible.py`, `style/prompts.py`, i tre tipi nuovi nelle ricette, CLI
   `style new|add|remove|list|prompt`, test.
2. **Punteggio**: `style/score.py`, `style.palette`, `style.coherence`, `style setup` con conferma del download,
   registrazione nei controlli, prova di accettazione con i numeri nel documento.
3. **Pagina Stile** nello studio.
4. (Dopo il merge della funzione 1) l'adattatore `batch_steps/style.py`.

## 8. Alternative considerate

- **Solo prompt fissi**: è quello che ha prodotto il lupo grigio. Necessari, non sufficienti.
- **Un giudice con un modello visione-linguaggio** (Qwen3-VL, già in cache per Qwen-Image): "questo concept è nello
  stile di questi esempi?". Spiega il perché a parole, ma è lento, più difficile da tarare, e la risposta cambia con
  la formulazione. Tenuto come seguito se l'embedding non basta.
- **Testo come ancora** (SigLIP 2 confronta l'immagine con "a photograph" o "a dark image"): gratis con lo stesso
  modello. Lo provo nella prova di accettazione; entra solo se migliora i numeri.
- **Embedding di stile dedicato (CSD)**: il più adatto sulla carta, escluso per la licenza dei pesi.
- **Media della bibbia invece dei vicini**: più semplice, ma con tipi diversi nella stessa bibbia la media non
  somiglia a niente.

## 9. Rischi

- **Il soggetto pesa più dello stile**: un lupo realistico può somigliare a un lupo approvato più di quanto un
  forziere approvato somigli a un forziere nuovo. Mitigazioni: vicini dello stesso tipo, controllo della palette, e la
  prova di accettazione ha negativi con lo stesso soggetto proprio per misurarlo.
- **Bibbia piccola o poco varia**: soglie rumorose. Limite minimo di 8 immagini, e il punteggio "lascia fuori uno" si
  mostra nella pagina, così si vede quando una bibbia è troppo uniforme.
- **Campione costruito con gli stessi prompt della bibbia**: separare diventa troppo facile. Soggetti diversi e negativi
  difficili lo mitigano; il giudizio vero arriverà con i concept approvati dall'utente nel pilota.
- **Download da 1,54 GB** per usarne un quarto: accettabile, dichiarato. Se pesa, DINOv2 (346 MB) è il ripiego se la
  prova dice che basta.
- **Sfondo**: tutti i concept hanno sfondo grigio chiaro; un fondo diverso cambia l'embedding. Il punteggio usa
  l'immagine scontornata (`__matted.png`) quando c'è.

## Seguiti

(vuoto: ci finiscono le osservazioni di revisione rimandate)

## Domande per l'approvazione

1. Va bene **SigLIP 2** (Apache-2.0, 1,54 GB) come modello, con DINOv2 (346 MB) scaricato solo per il confronto nella
   prova? Il download avviene solo dopo il tuo sì.
2. Va bene che nella prova di accettazione faccia io da approvatore della bibbia di prova, con il foglio di contatto
   mandato a te per conferma?
3. Il passo `style` dei lotti prima di `generate`: lo concordo io con la sessione della funzione 1?
