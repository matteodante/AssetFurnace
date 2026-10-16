# Piani di sviluppo

I documenti in questa cartella descrivono funzionalità proposte. Lo stato di un piano non implica che il codice sia già implementato.
Stato aggiornato al 16/10/2026. Le regole trasversali decise dall'utente (niente controlli sui pixel, controlli imparati
validati sui suoi giudizi, motori in `vendor/` senza ComfyUI, turni sulla GPU, review di Codex facoltativa) sono nel
[piano della pipeline](2026-10-15-auroconts-asset-pipeline.md#regole-trasversali-decise-dallutente-15-16102026).

| Piano | Stato | Revisione |
| --- | --- | --- |
| [Pipeline di produzione per un gioco intero (primo utente: Auroconts)](2026-10-15-auroconts-asset-pipeline.md) | In sviluppo: 1, 5 e 7 su `main`; 2, 3, 6 e 8 in parte; 4 con il progetto. Tabella dello stato e regole trasversali nel documento | — |
| [Funzione 1: lotti e accesso per agenti (progetto)](2026-10-15-batch-mcp-design.md) | Implementato (PR #2): lotti JSON/CSV con ripresa, server MCP, pagina Batches; [uso](../batches.md) | Codex, poi stop |
| [Funzione 2: dal riferimento al concept (progetto)](2026-10-15-reference-concept-design.md) | Strada scelta: Qwen-Image 2.1 in modifica dentro `sd-cli`. File della visione su `main` (PR #9); prova dei 4 casi fatta, in attesa del giudizio dell'utente | Da fare |
| [Funzione 3: bibbia dello stile (progetto)](2026-10-15-style-bible-design.md) | PR 1 implementata (PR #10: bibbia, prompt, riferimenti di stile). PR 2 in corso: gara fra SigLIP 2, CSD e giudice Qwen3-VL sui giudizi dell'utente | Codex, 2 giri |
| [Funzione 4: personaggi in pezzi e equipaggiamento (progetto)](2026-10-16-slot-parts-design.md) | Progetto unito (PR #20); revisione del 16/10 sera: tre modi nello studio, pezzi dalla strada C (un'immagine per pezzo con Qwen Turbo), B0 di riserva. [Rapporto della prova](2026-10-15-slot-parts-spike.md), [progetto della prova](2026-10-15-slot-parts-spike-design.md) | Codex, 1 giro facoltativo |
| [Funzione 5: profilo d'uscita per Unreal (progetto)](2026-10-15-unreal-profile-design.md) | Implementato (PR #6, #7), accettazione superata su UE 5.8.3 | Codex, 3 giri |
| [Funzione 6: famiglie di creature (progetto)](2026-10-15-creature-families-design.md) | PR 1 unita (PR #8: riconoscimento, `rig.deformation`, digitigradi e quadrupedi, studio). In revisione: PR #13 (molte zampe, serpenti), PR #16 (alati, piante), PR #14 (il lupo in Unreal: fronte dalle anche, misura del passo) | Codex, 3 giri |
| [Funzione 7: controlli e coda di revisione (progetto)](2026-10-15-checks-review-design.md) | Implementato (PR #4): contratto JSON, controlli generici, coda nello studio. Compenetrazioni in posa al secondo passo | Codex, poi stop |
| [Funzione 8: materiali pronti per il gioco, v2 (progetto)](2026-10-15-game-materials-v2-design.md) | Approvato il 16/10/2026. PR 1 e 2 implementate (PR #11 ORM e pezzo comune, PR #12 etichette). Prossime: zone per materiale (SAM 3), luce tolta, controllo imparato, PBR stimato. La [v1](2026-10-15-game-materials-design.md) è superata e resta per la storia e i Seguiti | Codex, 1 giro |
| [Modifica manuale in Blender e nuove versioni](2026-10-15-blender-roundtrip.md) | Implementato: apertura, importazione, versioni e preview; [uso e limiti](../blender-editing.md) | [Revisione tecnica finale e test](2026-10-15-blender-roundtrip-review.md); Blender reale e revisione UI |
| [Editor guidato dei modelli nella web app](2026-10-14-guided-model-editor.md) | Rimandato: la priorità corrente è il flusso Blender manuale. [Prove precedenti conservate](2026-10-14-guided-model-editor-progress.md) | [Revisione storica del piano: R1–R4 e S1–S3 risolti](2026-10-14-guided-model-editor-review.md) |
