# Economia dei token con Claude Code

Scritto per: Simone, che paga i token e vuole che la sessione principale (Fable 5.1) resti il più possibile inattiva.

## Il problema in una frase

Ogni turno della sessione principale rilegge l'intero contesto (100–150k token quando la sessione è lunga). Il costo quindi non dipende da quanto Claude scrive, ma da **quante volte si sveglia** e da **quanto è grosso il contesto** quando lo fa. Il 21/09 il 15% del budget settimanale è sparito in un giorno per queste cause, in ordine di peso:

| Causa | Costo | Rimedio |
|---|---|---|
| Un turno per ogni evento dei `Monitor` (oltre 40 in un giorno) | un contesto intero a evento | un solo risveglio a fine lavoro |
| Cron di polling ogni 12 minuti tutta la notte | un contesto a tick, anche a vuoto | niente cron di polling |
| Lettura di due artifact per ripubblicarli | ~50k token di HTML entrati nel contesto | ripubblicare senza leggere; letture agli agenti |
| Contesto ricresciuto dopo la compattazione | ogni turno più caro | `/compact` dopo output grandi |

Gli agenti invece erano già tutti su Opus (verificato nei transcript): il problema non erano loro.

## Quale meccanismo usare (e perché)

Claude Code offre quattro posti dove mettere una regola. Differiscono per **quando** vengono letti, ed è questo che decide se la regola viene dimenticata.

| Meccanismo | Quando è in contesto | Adatto a |
|---|---|---|
| `CLAUDE.md` nella radice del progetto | **sempre, a ogni turno** | regole che non devono mai essere dimenticate |
| Memoria (`~/.claude/projects/.../memory/`) | solo l'indice a inizio sessione; il contenuto quando Claude lo giudica rilevante | contesto di progetto, preferenze |
| Skill (`/nome`) | solo quando viene invocata | procedure da eseguire su richiesta (un audit, una checklist) |
| Hook e permessi in `.claude/settings.json` | **applicati meccanicamente**, senza passare dal modello | divieti che devono valere anche se il modello se ne dimentica |

Conclusione: un documento da solo non basta, perché la memoria è richiamata a discrezione e una skill va invocata. La combinazione giusta è:

1. **`CLAUDE.md`** con le regole in forma breve e imperativa (fatto: sezione "Economia dei token"). È l'unico testo garantito in contesto a ogni turno.
2. **Vincoli meccanici** in `.claude/settings.json` (fatto): `Monitor` e `CronCreate` negati dai permessi, e un hook `PreToolUse` che rifiuta ogni chiamata `Agent` senza `model` opus/sonnet/haiku. Questi funzionano anche quando Claude non ricorda.
3. **Questo documento** come spiegazione e riferimento; non serve invocarlo.
4. Una skill invocabile (`/token-audit`) è utile solo per un controllo periodico: contare i risvegli, verificare i modelli degli agenti, misurare il contesto. Non è il posto per la regola.

## Le regole operative

1. **Agenti sempre su Opus.** Ogni `Agent` con `model: "opus"` (haiku per lookup banali). Impostare anche il default in `/config` ("default subagent model" → Opus) come seconda rete.
2. **Un risveglio per lavoro.** I lavori lunghi (code di training, ladder, sim2sim) girano in un driver `nohup` sul desktop che fa tutto e termina. La sessione principale si sveglia una volta: o con un singolo `Bash run_in_background` che esce a fine coda, o con il rapporto di un agente Opus che ha aspettato con cicli `until … sleep 60` in primo piano (ogni chiamata ≤10 minuti, quindi l'agente resta residente senza rimbalzare al chiamante).
3. **Niente letture pesanti nella sessione principale.** Artifact, JSON di sweep, transcript: li legge un agente Opus e riporta le cifre. Le due dashboard sono già pubblicate da questa conversazione, quindi una ripubblicazione non richiede `read`.
4. **Contabilità delegata.** Un solo agente Opus per passaggio scrive NIGHT_LOG, WAVE_PLAN, memoria e ricostruisce il sito; la sessione principale copia il rapporto.
5. **Contesto corto.** `/compact` dopo ogni tool output grande; risposte brevi; nessun messaggio di conferma per eventi di routine.
6. **Controlli radi sui training**: al più ogni 500 iterazioni.

## Cosa fare quando la sessione è di nuovo cara

- Lanciare `/compact`.
- Controllare che nessun `Monitor` o cron sia attivo (`CronList`; i monitor compaiono come task in background).
- Verificare i modelli degli agenti: nei transcript in `/tmp/claude-*/…/tasks/*.output` il campo `"model"` deve essere `claude-opus-5` o `claude-haiku-4-5`.
- Se una regola viene ignorata due volte, spostarla da documento a vincolo meccanico (hook o permesso), non riscriverla più in grande.
