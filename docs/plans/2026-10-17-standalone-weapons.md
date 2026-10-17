# Armi e scudi come asset autonomi

Data: 17/10/2026. Sostituisce la decisione C (arma e scudo appesi a un socket della mano del corpo).

## La decisione dell'utente

"Non dobbiamo collegare il personaggio alla sua arma, non è lo scopo di AssetFurnace; dobbiamo però generarle
bene." Armi e scudi sono asset a sé. In un set si fanno accanto ai pezzi, ma non si adattano al corpo, non si
skinnano alla mano e non si agganciano al corpo nell'export.

## Cosa fa un'arma fatta bene

1. **Immagine in posa canonica.** Il prompt chiede l'arma dritta, impugnatura in basso, punta o testa in alto,
   fondo piatto (`pieces.held_prompt`); lo scudo dritto, con il fronte verso chi guarda.
2. **Controllo dell'orientamento.** Il giudice delle immagini (Qwen3-VL) dice se l'impugnatura è in alto, e in
   quel caso l'immagine si gira (`slot_parts/orient.py`). Sul set vero del 17/10 una delle due spade era stata
   disegnata con la punta in giù nonostante il prompt, e il giudice l'ha girata.
3. **Candidati con punteggio.** Il giudice dà un sì a ogni immagine ("un'arma completa, in posa canonica, su
   fondo semplice"). La migliore è la scelta automatica; lo studio si ferma per la scelta, come per i pezzi.
4. **Impugnatura.** Il grounding di Qwen3-VL dà il riquadro dell'impugnatura sull'immagine, nella sua griglia
   0-1000 (QwenLM/Qwen3-VL, `cookbooks/2d_grounding.ipynb`, letto il 17/10/2026). Provato con llama.cpp e il
   GGUF del giudice, sulle spade vere: riquadri giusti, anche sull'immagine girata, in 0,5 s.
   - La camera di Pixal3D porta il riquadro sul 3D. L'impugnatura è il centro della superficie del modello
     vista dentro il riquadro; l'asse è l'asse lungo del modello, verso la punta.
   - Su un'arma con un manico lungo il giudice dà tutto il manico, comunque gli si chieda (tre formulazioni
     provate su un'ascia vera). Dove si chiude la mano lungo il riquadro lo dice la classe: verso il fondo
     per ascia, martello e mazza, a metà per un bastone.
   - Lo scudo si tiene per la cinghia sul retro, che una foto di fronte non mostra: il centro del retro.
   - Tutto è scritto in `pieces/<slot>/grip.json`: classe, lunghezza, riquadro, punto, asse, chi l'ha messo.
5. **Correzione sulle tre strade, con lo stesso codice** (`grip.correct_in`):
   - nello studio, il marcatore sull'immagine (un clic lo sposta), il tipo e la lunghezza;
   - da riga di comando, `slot_parts.py grip`;
   - dagli agenti, MCP `pieces_grip`.
6. **Export Unreal.** Static mesh `SM_` con il pivot sull'impugnatura, la lama o la testa lungo +Z, un socket
   `grip` (il nodo `SOCKET_grip` del GLB, o fatto dal controllo in Unreal), la lunghezza reale della classe o
   quella corretta, collisione convessa. Il controllo in Unreal verifica il socket e che il pivot stia sul
   modello; non il pavimento.
7. **Controlli e coda di revisione come asset a sé.** Ogni arma e scudo di un set ha il suo record di
   provenienza accanto al GLB finito (`<slot>_finished.provenance.json`, legato dallo SHA-256), i controlli con
   il profilo `weapon_or_shield` (l'immagine come riferimento della sagoma) e una voce nella coda di revisione
   (`<set>-<corpo>-<slot>`), dove si controlla la licenza. Il `checks` di `<set>.parts.json` li nomina.

## Tolto

I socket `SOCKET_Weapon`/`SOCKET_Shield` sul corpo, "Held" e "Held by" nello studio, l'opzione `--socket`,
il controllo della presa (`GRIP_REACH_CM`), l'arma e lo scudo messi in mano dal fit (`held_placement`), l'osso
della mano per arma e scudo.
