# Regole di progetto (lette a ogni turno)

## Economia dei token (priorità assoluta)

La sessione principale gira su Fable 5.1, il modello più costoso. Ogni suo turno rilegge tutto il contesto. Le regole qui sotto valgono sempre; il dettaglio e il perché sono in `docs/TOKEN_ECONOMY.md`.

1. **Ogni chiamata Agent porta `model: "opus"`** (haiku per ricerche banali). Mai un agente senza `model`: il default eredita Fable.
2. **Un risveglio per lavoro, non per evento.** Niente `Monitor`, niente `CronCreate` di polling. Un driver in background (nohup) esegue tutta la coda e termina; un solo `Bash run_in_background` che esce a fine lavoro, oppure un agente Opus che aspetta con cicli `until` in primo piano (≤10 min ciascuno) e riporta una volta.
3. **Mai leggere artifact o file grandi nella sessione principale.** Dashboard: ripubblicare dal file locale senza `read`; se serve una lettura, la fa un agente Opus.
4. **La contabilità è delegata.** NIGHT_LOG, WAVE_PLAN, memoria, ricostruzione del sito, riassunti di sweep: un solo agente Opus per passaggio, la sessione principale rilancia il rapporto.
5. **Risposte brevi.** Nessun messaggio di conferma per eventi di routine; dopo un output grande suggerire `/compact`.
6. **Non controllare un training più spesso di ogni 500 iterazioni** e restare inattivi nel frattempo.

## Vincoli operativi (invariati)

- Mai uccidere processi dell'utente (`evaluate_viser.py`, `grasp_lab_viser.py`, tutto sotto `/home/simone/.venv`); mai `sudo`; solo docker rootless; mai toccare `~/SimToolReal_AnimRL` e `~/simtoolreal_animrl_ee` sui server; mai toccare job GPU di altri utenti.
- NIGHT_LOG.md in italiano, con orario CEST preso da `date`, solo per eventi reali.
- Interprete: `env -u PYTHONPATH deps/IsaacLab/.venv/bin/python`; test con `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
