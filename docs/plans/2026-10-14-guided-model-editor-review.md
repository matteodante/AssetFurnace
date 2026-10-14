# Revisione indipendente del piano dell'editor guidato

Data: 14 ottobre 2026. Revisore: sottoagente dedicato, su richiesta esplicita dell'utente. Oggetto: [piano completo](2026-10-14-guided-model-editor.md), confrontato con la repository alla base `6042e60`. Questa è una revisione del piano, non una certificazione del codice futuro o della qualità dei tagli.

## Revisione corrente — estensione a scheletro e pesi

Richiesta aggiuntiva dell'utente: studiare e includere anche la correzione dello scheletro. Ho riletto l'intero piano esteso, §§1–14, e verificato le integrazioni con il codice reale. Prima stesura estesa esaminata il 14 ottobre 2026: SHA-256 `9adb8fd00473b59a5d261c4e25aad3b0aa7040885efe94952ab007ae99f2437e`; l'hash dopo le correzioni è riportato nel recheck seguente.

**Giudizio finale corrente: piano esteso pronto per S0/RIG0. S1–S3 chiusi nel documento; R1–R4 restano chiusi. Nessun rilievo P1/P2/P3 residuo di questa revisione.** Le prove restano obbligatorie: questo è un giudizio sul contratto di implementazione, non la certificazione di una feature già funzionante. Nessun codice prodotto è stato modificato; non ho eseguito una prova di rigging o retarget della futura feature.

### Recheck finale dell'estensione

Versione finale esaminata il 14 ottobre 2026: SHA-256 `2c484446d4930879e8ebb903d30944e8b7a1821eb008bf8f9d112247b70ff8c3`. Ho riletto i contratti numerici aggiornati e controllato i rimandi nei requisiti, nelle firme del documento/API, nell'invalidazione, nell'export UE, nei test e nelle milestone, compreso l'ultimo allineamento delle sezioni 6, 8, 9 e 12 alla calibrazione motion. I findings storici S1–S3 sono conservati più avanti.

| Rilievo | Disposizione | Riscontro nel documento aggiornato |
| --- | --- | --- |
| S1 — bind comune | **Chiuso nel piano** | §14.4 distingue bind autorevole e TRS iniziali, richiede `B[j] * I[j] = M_bind` per ogni joint ammesso e disabilita pittura/struttura quando il requisito non è certificabile. §14.5 subordina la preview pesi a questa verifica; §14.6 definisce il bind dei joint nuovi. §14.9 include articolazione → pesi → reload, TRS fuori bind recuperabili e contributi incoerenti da rifiutare. |
| S2 — frame e movimento | **Chiuso nel piano** | Operazioni distinte `joint_frame_adjust` e `motion_axis_calibrate`; la seconda ha formula nel frame rest-local, cinematica diretta dei figli, ordine dei passaggi e policy persistente. Il testo chiarisce torsione inclusa, calibrazione applicata una volta sola, clearance e necessità di conversione/ricalibrazione quando cambia il frame. Firma motion separata, aggiornamento clip e test sulla superficie compaiono in documento/API/UE e RIG3. |
| S3 — template dita | **Chiuso nel piano** | §14.6 fissa tre joint deformanti, punta guida esclusa dalla skin, mapping SOMA concreto per pollice e altre dita e metacarpale da verificare. Nuove catene e completamenti distali compatibili sono distinti da inserimenti interni esclusi. Altre cardinalità richiedono adapter provato o parent-follow esplicito; RIG4 verifica il caso reale. |

Il contratto `R_local = R_rest_local * D_local`, con `D_corrected = Q_cal * D_local * inverse(Q_cal)`, è coerente come rotazione dell'asse del delta in quello spazio. `Q_cal` identità e delta nullo conservano il percorso di base; il caso X45°/Z90° distingue una calibrazione reale da un semplice cambio del frame. Il piano richiede correttamente una verifica della superficie e della ricostruzione gerarchica, non soltanto del gizmo. L'equivalenza con retarget/clearance legacy resta un test da eseguire in RIG0, dichiarato come tale.

Le stime aggiornate sono internamente coerenti: RIG0–RIG4 **26–44 giornate**, RIG0–RIG3 **20–34**; totale completo **60–103**, primo rilascio esteso **54–93**, inclusa la nuova calibrazione motion. Restano stime preliminari. Non ho trovato una nuova contraddizione concreta fra §§1–13 e §14: calibrazione e modifica strutturale hanno firme e derivati diversi, l'editing dei pesi conserva il riposo solo dopo il gate, e il rilascio parziale RIG3 non viene presentato come completamento di RIG4.

### Storico — copertura dell'estensione alla prima lettura

| Aspetto | Esito della verifica |
| --- | --- |
| Articolazioni e figli | §§1, 3 e 14.4 distinguono modifica a riposo, singolo giunto e catena. La ricostruzione delle trasformazioni locali dei figli e la chiusura delle dipendenze sono previste; lock, simmetria e scale negative sono coperti. |
| Bind e mesh a riposo | La formula proposta conserva esattamente il contributo di ciascun joint a pesi fissi. Manca una precondizione per garantire l'invarianza anche cambiando successivamente i pesi: S1. |
| Pesi locali | Il piano include ROI, quattro influenze, somma unitaria, lock, destinatari espliciti della massa, pruning misurato, undo dell'intero vettore e simmetria verificata. Non impone il rebind globale legacy. |
| Asse di flessione | Il piano promette una correzione guidata dell'asse, ma il retarget delle rotazioni globali può annullare l'effetto di un puro cambio di frame: S2. |
| Catene mancanti | RIG4 è esplicita e compresa nella consegna completa. Nuovi ID, tabella joint, bind, pesi, mapping e fallback parent-follow sono trattati. Il numero di segmenti supportato va ristretto mediante template e mappe provate: S3. |
| Kimodo e clip importate | Distinzione corretta: sorgenti Kimodo con hash e parametri completi, conversione importata limitata, vecchie clip stale e dichiarazione di export statico quando la conversione manca. Root motion, timing, STEP e baking con controllo tra campioni sono considerati. |
| Sessioni e lineage | §§5, 6, 8, 9 e 14 estendono documenti, revisioni rig/pesi, firme, invalidazione, undo e salvataggio congiunto. Non resta la precedente assunzione che bind e clip siano sempre invariati. |
| UE | §§9 e 14.8 prevedono Skeleton di prova separato, nuove verifiche e dipendenze di LOD, physics, socket e Blueprint. Nessuna sostituzione automatica dello Skeleton condiviso. |
| Scope e stima | Distinte release RIG3 e completamento RIG4. I conti 34–59 + 24–40 = 58–99 giornate e 34–59 + 18–30 = 52–89 sono corretti. Sono stime da rivedere dopo le prove, non una misura verificata. Un eventuale comando aggiuntivo di calibrazione motion in risposta a S2 deve entrare anche nelle attività e nella stima. |

### S1 — P2, chiuso: certificare la bind pose prima di garantire l'invarianza durante la pittura dei pesi

**Riferimenti della stesura estesa esaminata:** §14.4, righe 479–488; §14.5, righe 496–504; §14.9, prove Invarianza al riposo/Pesi. §1 promette mesh a riposo preservata nel rilascio esteso.

La formula `I1[j] = inverse(B1[j]) * B0[j] * I0[j]` è corretta nello spazio comune dichiarato: mantiene `C[j] = B[j] * I[j]`. La somma pesata resta quindi identica **con gli stessi pesi**. Non segue però che la somma resti identica dopo la pittura se i `C[j]` differiscono tra i joint ammessi. La riga 486 lascia aperto l'editing dei pesi anche quando rest e bind non sono coerenti, senza precisare questo requisito.

Controesempio algebrico della revisione, non difetto già osservato su uno specifico asset: due contributi a riposo `C0 = identità` e `C1 = traslazione di 0,2 m`; pesi `(0,5, 0,5)` producono uno spostamento di 0,1 m, pesi `(1, 0)` lo eliminano. I vertici del buffer non sono cambiati, ma la superficie mostrata come riposo sì. Il no-op e il solo cambio del bind non intercettano questo caso.

**Cambio richiesto:** RIG0/preflight deve identificare una bind pose autorevole per ogni skin, distinta dai TRS dei nodi appena caricati. In tale posa i contributi dei joint tra cui è permesso ridistribuire i pesi devono coincidere con la stessa trasformazione mesh→riferimento canonico entro tolleranza, incluse trasformazioni di nodi intermedi e istanze. Se la posa non è ricostruibile o il requisito fallisce, non dichiarare disponibili le capability che promettono mesh a riposo invariata; una normalizzazione/rebind esplicita è un'operazione distinta, mai una correzione silenziosa. Testare separatamente TRS iniziali fuori bind recuperabili e inverse bind incoerenti da rifiutare; includere la sequenza spostamento articolazione → pittura pesi → reload.

Il [tutorial Khronos sullo skinning](https://github.com/KhronosGroup/glTF-Tutorials/blob/main/gltfTutorial/gltfTutorial_020_Skins.md) conferma che la matrice di joint combina trasformazione globale e inverse bind e che il risultato è pesato per vertice. La precondizione e il controesempio sopra sono deduzioni algebriche della revisione.

### S2 — P2, chiuso: il cambio del frame di riposo non corregge da solo l'asse di flessione delle clip

**Riferimenti:** §14.3, `joint_axis_adjust`; §14.4, riga 477 e contratto del bind; §14.7, righe 531–537; RIG2/RIG3. Codice: [kimodo_retarget.py](../../scripts/kimodo_retarget.py), righe 435–444.

Il retargeter esistente costruisce la rotazione globale come `D(t) * alignment * R_rest`. In un caso con alignment identità e puro cambio del frame, ricreare l'inverse bind fa sì che `R_rest` si cancelli nel prodotto posa×inverse-bind. Esempio: `R0 = identità`, `R1 = rotazione Z di 90°`, `D = rotazione X di 45°`; la nuova deformazione è `D * R1 * inverse(R1) = D`. Il frame visualizzato ruota, ma il movimento rimane attorno all'asse precedente. Il problema si presenta anche nel percorso importato se si conserva la rotazione globale rispetto al vecchio rest e la si moltiplica per il nuovo frame.

Non è un errore nella formula del bind: è la distinzione tra cambiare coordinate di riferimento e modificare il movimento che agisce sulla mesh. Il piano attuale le presenta insieme come controllo dell'asse di flessione.

**Cambio richiesto:** definire due semantiche: correzione del frame rest/bind, utile per export e movimenti futuri e senza promessa di cambiare il bending delle clip esistenti; eventuale calibrazione dell'asse motion, persistente e applicata coerentemente alle clip interessate, che modifichi davvero la rotazione applicata. Per la seconda fissare spazio di riferimento, formula, ordine rispetto ad alignment/spread/clearance e propagazione ai figli; salvare parametri e firma della policy, invalidare le clip e testare l'effetto nel GLB ricaricato. Non deve diventare un keyframe della sola posa osservata. In alternativa limitare esplicitamente la UI iniziale al frame di riferimento. Aggiornare i requisiti, operazioni journal, test e attività/stima per la scelta adottata.

**Test discriminante:** solo cambio di frame → deformazione precedente conservata nel caso di controllo; calibrazione motion → asse di flessione misurabilmente diverso, riposo invariato, effetto coerente nelle altre clip e nessun cambio involontario del root motion. Un test che controlla soltanto orientamento del gizmo o matrici del joint non basta.

### S3 — P3, chiuso: fissare i template e le mappe delle dita invece di promettere distribuzione generica fra catene

**Riferimenti:** §14.6, righe 512–520; §14.7, manifest `joint-ID → ruolo SOMA`; RIG4. Codice: [kimodo_retarget.py](../../scripts/kimodo_retarget.py), `_spread` alle righe 82–89 e mapping dita alle righe 228–234.

Il codice attuale seleziona ruoli SOMA mediante indici arrotondati lungo la catena; non interpola una distribuzione articolare generale. Una catena più lunga può assegnare lo stesso ruolo a più joint e quindi non produrre flessione indipendente in tutti i segmenti. Un mapping esplicito uno-a-uno per joint può essere sufficiente per template scelti, ma la frase secondo cui il numero di segmenti non è vincolato alla sorgente perché il mapping «può distribuire» lascia indeterminata la capacità promessa.

**Cambio suggerito:** fissare in RIG0/RIG4 i template supportati, numero/ruolo di joint deformanti ed eventuale punta non deformante, con mapping SOMA concreto e versionato per pollice e altre dita. Provare le cardinalità ammesse; le altre sono unsupported oppure parent-follow dichiarato. Non occorre introdurre un retargeter generico per soddisfare Completa dita. Se si sceglie invece una vera distribuzione/interpolazione, occorre specificarne schema, algoritmo, vincoli e test e includerla nella stima.

### Verifiche sul codice e rischi già coperti

- Il percorso [rig_api.py](../../viewer/rig_api.py) / [blender_rebind.py](../../scripts/blender_rebind.py) richiede realmente una scena preparata e rigenera Rigify. Il profilo quadrupede non è un adapter già pronto per i GLB SkinTokens: la scelta di un adapter GLB canonico distinto è corretta.
- [blender_rebind_weights.py](../../scripts/blender_rebind_weights.py), righe 128–158, elimina i vecchi gruppi, trasferisce per superficie vicina e fallisce quando tutti i vertici risultano senza pesi; non blocca ogni caso di alcuni vertici senza pesi. Il piano descrive correttamente il limite e non riusa questo processo come pittura locale.
- Il fallback Middle per mani incomplete, l'ereditarietà del mapping dell'antenato per ossa extra e `stiffen_fingers` sono presenti nel retargeter. La precedenza del mapping confermato dall'utente va aggiunta e il piano lo richiede già.
- La gestione del gomito che mantiene fermi polso/spalla richiede di ricalcolare trasformazioni locali anche di figli non traslati globalmente. Il piano registra questa chiusura; non è una lacuna.
- La normalizzazione con lock deve preservare i singoli pesi protetti, non solo la loro somma. Le prove «lock intatti», la precedenza delle influenze bloccate e l'undo del vettore completo rendono già esplicito l'invariante nel piano; non ho aperto un finding duplicato.
- L'adattamento delle clip importate è limitato a rig rigidi, gerarchia invariata e mapping verificato; traslazioni non-root e scale animate non sono tacitamente ignorate. Il fallback statico resta escluso dai gate obbligatori della pipeline.
- Le dipendenze nuove sono integrate nei paragrafi precedenti: la frase su tracce/pesi invariati in §6 è delimitata ai pennelli di forma dalla precisazione subito successiva; §§8–9 rinviano alla matrice di invalidazione rig. Non ho rilevato una contraddizione residua in questi passaggi.
- La [documentazione Epic sull'editing dello scheletro](https://dev.epicgames.com/documentation/unreal-engine/skeleton-editing-in-unreal-engine), ricontrollata, distingue aggiornamento dei figli, orientamento e pesi. Nel piano è una fonte di comportamento, senza diventare una dipendenza della web app.

Questa prima lettura non aveva riaperto R1–R4. Il successivo recheck di S1–S3 è concluso nella sezione «Recheck finale dell'estensione»; le prove RIG0 non sono ancora state eseguite o certificate.

## Storico — giudizio finale del piano della sola forma

**Esito precedente, riferito al piano prima dell'estensione scheletro: pronto per iniziare S0, R1–R4 chiusi.** Questo approvava il contratto di lavoro della forma, non certificava ancora fattibilità, prestazioni o qualità della feature. Il giudizio corrente sull'estensione è riportato sopra.

Browser per l'interazione, documento canonico e Blender per le operazioni definitive sono una separazione ragionevole per questa repository. Il piano evita correttamente di promettere riparazioni universali e include il costo dell'assembler glTF. La prima lettura aveva individuato tre contratti da precisare e un chiarimento visivo; il testo storico dei rilievi è mantenuto sotto per tracciabilità.

## Storico — recheck e disposizione R1–R4

Ho riletto l'intero piano aggiornato il 14 ottobre 2026, comprese le interazioni fra le aggiunte e API, Library, derivati, test e milestone. Versione esaminata: SHA-256 `7bd53fe9bd7c4f5f6562baa9ddd2cbdcd53903f626b18bcc27c5c6f7582093d8` del file principale. Non ho modificato il piano principale né eseguito le prove future.

| Rilievo | Disposizione | Riscontro nel piano aggiornato |
| --- | --- | --- |
| R1 — corpus reale | **Chiuso nel piano** | §11 rende obbligatori un caso reale di fessura e uno di distacco, un secondo asset e un rapporto con stage/hash/difetto e verifiche. La mancanza del corpus lascia S0 incompleta; §12 riporta il gate nella checklist. Le fixture sintetiche restano indipendenti dai file locali. |
| R2 — commit delle patch | **Chiuso nel piano** | §8 definisce blob e record immutabili, `draft/HEAD.json` come unico puntatore di commit, ACK successivo, proiezioni ricostruibili, recovery dalla catena, retry idempotente e Apply sull'esatta sequenza confermata. §12 include fault injection, ACK perso e sequenza non confermata. |
| R3 — accesso e server statico | **Chiuso nel piano** | §10 esplicita workspace a utente unico, token fuori dalla web root, bootstrap senza endpoint anonimo, controllo delle letture/scritture e deployment remoto verificato. GET/HEAD statici di `output/edits` sono negati; artefatti e download passano dalle API protette, con fetch autenticato e blob URL per il viewer. §12 include test delle route e del percorso statico. Il testo non promette isolamento multiutente né riservatezza delle API storiche. |
| R4 — preview della separazione | **Chiuso nel piano** | §7 distingue l'evidenziazione immediata della superficie da rimuovere, il clipping provvisorio da misurare e la geometria definitiva del candidato. Per Stacca, lo scostamento di ispezione è temporaneo e non esportato. Il risultato non viene presentato come Boolean definitivo eseguito nel browser. |

Le aggiunte non introducono nuove contraddizioni concrete nei requisiti. L'indicazione UE 5.8 è presentata come riscontro locale da ricontrollare, senza codificare un percorso personale nel prodotto. Restano aperte soltanto le incertezze sperimentali già assegnate a S0: qualità sui casi reali, ricostruzione delle cuciture, attributi/pesi dopo i tagli, compatibilità delle versioni e prestazioni. Non sono risultati già acquisiti e non sono rilievi di piano lasciati senza risposta.

## Copertura dei requisiti dell'utente

| Requisito | Valutazione |
| --- | --- |
| Sposta, gonfia/sgonfia, leviga | Coperto in §§3 e 6: selezione locale, maschere, falloff, undo, controlli della densità e delle cuciture. |
| Fessura e distacco | Entrambi inclusi come requisiti di rilascio e distinti in §7. Il percorso per superfici aperte è riconosciuto come lavoro separato. Vedere R1 sul gate dei casi reali. |
| Anteprima immediata, poi Applica | Chiaro per i pennelli; per i tagli l'anteprima è un volume di intenzione. Vedere R4 per evitare aspettative sbagliate senza introdurre Boolean completi nel browser. |
| Modifica a riposo e in posa | Coperta. La correzione viene riportata alla geometria di riposo; non è un morph legato alla posa. La separazione avviata in posa ha un passaggio esplicito a riposo. |
| Verifica sulle altre animazioni | Coperta da §9: tutte le clip compatibili, tempi registrati e copertura parziale dichiarata. Il piano non scambia un campionamento per una garanzia su ogni frame. |
| Semilavorati e metadati | Snapshot, bozza, operazioni, candidato, revisione, provenienza e derivati sono previsti. Vedere R2 sul recupero dei tratti già confermati. |
| Consegna coerente a UE | Coperta: stessa revisione per mesh e clip, derivati obsoleti identificati e import reale nel progetto di prova. Il successo del GLB non viene confuso con quello dell'import FBX. |
| Interfaccia semplice nella web app | Coperta con un modo Modifica nel viewer esistente; non viene introdotto un editor generalista o un pass AI obbligatorio. |

## Rilievi della prima lettura — tutti chiusi nel recheck

I paragrafi seguenti descrivono la stesura iniziale. La disposizione corrente è quella della tabella precedente; i cambi suggeriti sono stati recepiti e non rappresentano richieste ancora pendenti.

### R1 — P2: il gate dei tagli deve includere asset reali della pipeline

**Riferimenti:** §11, tabella S0 e «S0: esperimenti da eseguire per primi»; §12, corpus minimo. Nella prima stesura, l'inventario reale è obbligatorio per il profilo GLB e il no-op, mentre gli esperimenti di separazione sono descritti principalmente su mano/tessuto sintetici e l'aggiunta del guerriero resta facoltativa.

**Perché conta:** una sottrazione riuscita su un ponte creato appositamente non dimostra che si possano correggere i difetti delle mesh effettivamente generate. Il rischio specifico — superfici aperte, sovrapposte o nonmanifold — è già spiegato bene nel piano, ma può restare fuori dal criterio di uscita del primo investimento. Non è una richiesta di rivedere ora tutti gli esperimenti del guerriero.

**Cambio suggerito:** prima di chiudere S0 scegliere un piccolo corpus reale, identificato da stage, hash e difetto selezionato, oltre alle fixture sintetiche. Richiedere almeno un caso reale di fessura e un caso reale di distacco rappresentativi degli output che si vogliono supportare, con UV/materiali e, dove disponibile, rig. I file possono restare fuori dalla CI pubblica se la provenienza non consente redistribuzione; rapporto e risultati devono comunque essere obbligatori. Se non sono disponibili, S0 resta incompleta su questa prova invece di riportare una fattibilità generale. Un caso obbligatorio non elaborabile impone una decisione esplicita su tecnica o scope prima della UI completa.

### R2 — P2: il commit atomico va definito anche per le patch, non solo per accept

**Riferimenti:** §8, layout `draft/operations.jsonl` + delta/checkpoint + `session.json`; route `POST .../patches`; paragrafi Concorrenza e Autosave. Il piano prevede rename atomico per i singoli file e journal per accept, ma non definisce quale scrittura rende durevole una pennellata.

**Perché conta:** un crash fra scrittura del delta, append al journal e aggiornamento dell'ultima sequenza può lasciare un gesto confermato ma non recuperabile, oppure farlo riapplicare quando il browser ritenta la stessa chiave. L'atomicità del singolo rename non rende atomica questa sequenza di più file. È un caso diverso dal commit finale della revisione e riguarda direttamente la promessa «bozza recuperabile».

**Cambio suggerito:** scegliere una fonte di verità per i tratti committed. Per esempio: blob immutabile del delta verificato → record immutabile di patch con hash base/risultato e chiave idempotente → avanzamento atomico del puntatore dell'ultima sequenza; solo allora rispondere con ACK. Il JSONL può essere una proiezione ricostruibile. Al restart ignorare record non committed e ricostruire lo stato confermato; un retry deve restituire l'esito precedente, non sommare nuovamente il delta. Specificare anche che Apply aspetta/indica l'ultima sequenza confermata. Aggiungere fault injection dopo ciascuna scrittura e test di ACK perso/retry. Si può scegliere un altro meccanismo, purché garantisca lo stesso invariante.

### R3 — P2: chiarire accesso remoto e protezione del namespace statico delle sessioni

**Riferimenti:** §10, paragrafo su origine, token anti-CSRF e ID autorizzati; §8, storage sotto `output/edits`. Codice: [serve.py](../../viewer/serve.py), righe 8, 68 e 114; [generate_api.py](../../viewer/generate_api.py), classe `Handler` alla riga 2033 e fallback `super().do_GET()` alla riga 2526.

**Evidenza:** il server attuale serve l'intera repository con `SimpleHTTPRequestHandler`; `--auto` può ascoltare su `0.0.0.0` per il proxy Runpod. Non ho trovato un sistema generale di login/identità degli utenti a cui agganciare «ID autorizzati per sessione». Inoltre i nuovi manifest/buffer sotto `output/edits` sarebbero leggibili attraverso il fallback statico, indipendentemente dalle verifiche aggiunte alle route API.

**Perché conta:** anti-CSRF e controllo Origin non sono autenticazione del client remoto. Lasciare indefinito il confine rischia di far implementare autorizzazioni soltanto sulle route nuove mentre i dati o eventuali token persistiti restano raggiungibili tramite un altro percorso. Non serve introdurre un sistema multiutente completo, ma il deployment supportato deve essere concreto.

**Cambio suggerito:** specificare che l'uso remoto di Edit richiede un tunnel privato oppure un proxy autenticato verificato; un bind remoto o un URL di proxy da soli non attestano questa condizione. Definire emissione/conservazione del token e non collocare segreti in file serviti dalla repository. Negare l'accesso statico diretto ai file privati di sessione e servire manifest, buffer, checkpoint e download attraverso route coerenti con la stessa politica di accesso. In alternativa dichiarare esplicitamente un modello locale a utente unico con limiti remoti, senza promettere isolamento fra utenti. Testare sia route API sia URL statici diretti, senza ampliare il lavoro a una riscrittura generale del server.

### R4 — P3: descrivere meglio la preview immediata della separazione

**Riferimenti:** §3, passi Correggere/Applica; §7, Apri fessura, punti 2–3. Il testo è onesto: mostra subito il cutter e rimanda la superficie definitiva ad Applica. La preferenza dell'utente per la preview immediata può però essere interpretata come visibilità immediata del risultato.

**Cambio suggerito:** fissare un contratto visivo minimo: durante il gesto evidenziare la regione che verrà tolta e, se fattibile nel benchmark, nasconderla provvisoriamente tramite clipping/maschera di rendering; per Stacca mostrare selezione e separazione temporanea/ispezionabile. Etichettare questa vista come approssimata: caps, nuovi UV e deformazione definitiva arrivano con il candidato. Il semplice contorno del cutter non deve essere presentato come una mesh già riparata. Questo chiarimento non richiede un secondo motore CSG nel browser e non cambia il motore Blender proposto.

## Aspetti verificati che non considero omissioni

- **Identità glTF/Blender:** il documento canonico, gli ID per revisione, i dati per angolo di faccia e il test no-op evitano l'assunzione errata che gli indici sopravvivano al round-trip. La ricostruzione conservativa delle cuciture e la sidecar per gli asset nuovi sono già trattate.
- **Inverse skinning:** la formula proposta è corretta per pesi e posa congelati quando la parte lineare è invertibile. Nel [Three r170 locale](../../viewer/vendor/three.module.js), shader `skinning_vertex` e `applyBoneTransform` applicano la combinazione pesata tra i cambi di bind. Il piano include matrici non identità, scale, casi singolari e confronto al renderer; non occorre inventare un correttivo per posa.
- **Picking e BVH:** è già esplicitato che il BVH della posa di riposo non basta e che i cambi di topologia richiedono ricostruzione. La [documentazione SkinnedMesh](https://threejs.org/docs/pages/SkinnedMesh.html) conferma la necessità di aggiornare i bounds per mesh animate; il piano richiede correttamente il controllo sulla versione vendorizzata.
- **Dipendenze:** il [manifest upstream di three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh/blob/master/package.json), ricontrollato durante la revisione, riporta 0.9.15 e peer Three >=0.159.0. Questo non dimostra da solo compatibilità di esecuzione con r170: il benchmark/pin richiesto dal piano copre già il rischio.
- **Attributi dopo il taglio:** UV per corner, normali/tangenti, patch materiale, quattro influenze, pesi ambigui e assenza di anatomia nascosta sono esplicitati. Un trasferimento automatico sempre corretto non viene promesso.
- **Clip e derivati:** il caricamento corrente di GLB separati è reale in [viewer.js](../../viewer/studio/viewer.js), righe 292–305 e 327–333. Il piano lo affronta con geometria condivisa, verifica di bind/rig e derivati legati alla revisione. Non manca una soluzione per il cambio clip.
- **Library:** [assets_api.py](../../viewer/assets_api.py), righe 255–290, sceglie oggi stage e clip in base ai run più recenti. L'integrazione mediante manifest espliciti è quindi necessaria e già prevista, compresa la conservazione degli ID storici.
- **Stima e gate:** 34–59 giornate non è presentato come una misura già verificata. L'assembler, la separazione su superfici aperte e la preservazione degli attributi giustificano una nuova stima dopo S0. Non ho una base sperimentale per certificare quella durata, ma non vedo una promessa ingannevole di feature completa in pochi giorni.
- **Verifica UE:** è correttamente un gate di rilascio reale, separato dalla validazione glTF. Non serve aggiungere una promessa di test Unreal totalmente automatici alla prima versione.

## Metodo e limiti della revisione

Ho letto tutte le sezioni del piano e confrontato i punti di integrazione con `viewer/studio/viewer.js`, `app.js`, `library.js`, `viewer/components/model-viewport.js`, `viewer/animation/player.js`, `viewer/serve.py`, `generate_api.py`, `assets_api.py`, `activity_api.py`, riferimenti a Finish/Autorig, `image_to_3dlab/blender.py`, il Three vendorizzato e l'harness dei test JS. Ho ricontrollato fonti primarie mirate per Three/BVH; il fetch della pagina Blender Boolean 4.5 ha restituito HTTP 402 anche in questa revisione, quindi non lo presento come una verifica completa degli operatori di Blender.

Nessun test della feature è stato eseguito perché la feature non esiste ancora. Nessun codice prodotto modificato. I rischi già dichiarati dal piano non sono stati duplicati come difetti della revisione. L'esito dopo le correzioni è registrato nella sezione «Recheck e disposizione finale».
