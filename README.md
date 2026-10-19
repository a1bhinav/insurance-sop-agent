# Insurance Claims SOP Agent

A conversational claims-support agent that **follows a fixed business workflow without sounding like a form**.

```
VERIFY_ID  ──►  RESOLVE_INTENT  ──►  PROCESS_CASE  ──►  POST_PROCESS
 strict           guided              flexible           guided
```

The LLM interprets messy language, resolves ambiguity and phrases every reply; a deterministic
SOP controller owns the phase order, the safety gates (identity, consent, scope, attempt limits)
and the only facts the model is allowed to speak. Memory is global: anything useful the caller
says is stored the moment they say it, even if it belongs to a later phase.

* Test UI with a live **SOP inspector** (phase rail, identity gate, cross-phase memory, guard log, email outbox)
* Works with **Anthropic** (default) or any **OpenAI-compatible** API; falls back to a deterministic **mock mode** with no key
* 47 automated tests run the whole workflow without a model, including the demo test case and the bonus scenarios

---

## 1. Run it

### Docker (recommended)

```bash
git clone <this repo> && cd goaly
cp .env.example .env            # put your key in LLM_API_KEY (or paste it into the UI later)
docker compose up --build
# → open http://localhost:8000
```

Or without compose:

```bash
docker build -t insurance-sop-agent .
docker run --rm -p 8000:8000 -e LLM_API_KEY=sk-ant-... insurance-sop-agent
```

### Local Python (3.10+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEY=sk-ant-...           # optional; omit to run in mock mode
uvicorn sop_agent.api.app:app --reload  # → http://localhost:8000
```

### Giving the agent an API token

Any one of these works (first match wins):

| How | Details |
|---|---|
| Environment | `LLM_API_KEY` (also accepts `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), `LLM_PROVIDER=anthropic\|openai\|mock`, `LLM_MODEL` |
| `.env` file | Same variables; loaded automatically from the project root |
| Test UI | "Model API key" field in the top bar — sent per request in the `X-LLM-API-Key` header, kept only in that browser tab |
| API | Headers `X-LLM-API-Key`, `X-LLM-Provider`, `X-LLM-Model` on any request |

Defaults: `claude-sonnet-5-5` for Anthropic, `gpt-4o` for OpenAI. `LLM_EXTRACTOR_MODEL` lets you use a
cheaper model (e.g. `claude-haiku-5-5`) for the per-turn extraction step. `LLM_BASE_URL` points the
OpenAI-compatible provider at Azure/OpenRouter/Ollama/vLLM. With no key at all the server starts in
**mock mode** (rule-based extractor + templated replies) so the SOP can still be exercised end to end.

---

## 2. Try the workflow

Open the UI, pick a **sample caller** (or just type), and watch the inspector on the right.

**The demo test case** — one utterance:

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied
> healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

What happens in that single turn (visible in the inspector's gate log):

```
EXTRACTED: pii=[dob, full_name, id_last4, policy_number] intent=denial_question
MEMORY case_hint: case_type=healthcare · status=denied · month=1
GATE identity: matched 3/3  → PASSED
TRANSITION VERIFY_ID -> RESOLVE_INTENT
CLAIM selected CL-2048 (resolved from remembered hints)   ← no "which claim?" question
TRANSITION RESOLVE_INTENT -> PROCESS_CASE
```

Then ask *"what documents do I need and how do I submit them?"*, *"how long once I submit?"*, say
*"that's all"*, and choose whether to receive the **email summary** (rendered in the inspector and
written to `data/outbox/`).

Other scripted callers: partial answers with a question asked *before* verification (held and answered
after), a frustrated caller (bonus), wrong details hitting the attempt limit, a caller who refuses
verification, out-of-scope questions (`what is RL?`), and an **authorized representative** who needs
the policyholder's consent (switch the consent scenario to `timeout` to see the escalation).

Command line equivalents:

```bash
python scripts/run_scenarios.py                 # annotated transcripts for every scenario
python scripts/run_scenarios.py demo frustrated # a subset
pytest -q                                       # 47 tests, no model needed
```

---

## 3. How the harness works

```
caller text
   │
   ▼
┌───────────────────────────┐   every turn, every phase
│ Extractor (LLM → JSON)    │   pii · role · intent · case hints · questions · scope ·
│ fallback: rule-based      │   emotion · yes/no answer · refusal · human request
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐   Memory.absorb() — hints are kept even when the
│ Cross-phase memory        │   current phase can't act on them yet
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐   deterministic: owns phase order, gates, allowed actions,
│ SOP controller            │   tool calls (lookup, verify, consent, email). Produces a
│  VERIFY_ID → … → POST     │   Plan = directive + whitelisted FACTS + must-nots + tone
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐   phase-specific system prompt; may only use FACTS;
│ Responder (LLM)           │   fallback: templates
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐   LeakGuard: pre-verification replies may not contain any
│ Output guards             │   claim literal/ID/amount → replaced by a safe reply
│                           │   PIIGuard: never echo record values the caller didn't give
└────────────┬──────────────┘
             ▼
          reply + state + trace  (the UI renders all three)
```

### The freedom dial

| Phase | Freedom | What the model sees | What the controller enforces |
|---|---|---|---|
| `VERIFY_ID` | **strict** | No claim data at all — only verification progress and the caller's own words | ≥ 3 matching PII factors (name, DOB, phone, email, SSN/national-ID last 4); policy number only locates the record; mismatches say *which* field failed, never the stored value; 3 failed attempts → locked → human; refusals → explain why → alternatives → stop persuading → human |
| `RESOLVE_INTENT` | guided | Claim *briefs* (type, date, status, number) | Candidate set is computed from hints (`case_type`, `status`, month/year, claim id); one match → auto-select, several → disambiguate, none → list what's on file |
| `PROCESS_CASE` | **flexible** | The selected claim record, document requirements, alternatives, follow-up guidance, today's date | Model answers freely but only from FACTS; switching claims, "anything else?", unsupported intents (contact changes, coverage) → human offer |
| `POST_PROCESS` | guided | Masked email on file | Explicit yes/no consent to send; summary contains what was discussed, status/outcome, next steps; representatives' summaries go to the policyholder's email |
| `HUMAN_HANDOFF` / `CLOSED` | strict | Reference number | Terminal; a verified caller may re-open with a new claim question |

### Memory across phases

`TurnSignals` are extracted on **every** turn regardless of phase and merged into `Memory`
(intent hints, case hints, questions asked too early, volunteered notes, email preference). In
`VERIFY_ID` the agent acknowledges the hint in the caller's own words ("I've noted you're calling
about your denied healthcare claim from January — I'll pull that up as soon as we've verified…")
without adding any fact. After verification `RESOLVE_INTENT` resolves the claim from those hints
instead of asking, and `PROCESS_CASE` answers the question that was held.

### Scope control

The extractor labels each message `in_scope`, `small_talk` or `out_of_scope`. Off-topic requests
are declined without being answered (not even partially) and the agent steers back to the current
step. Consecutive off-topic strikes reach `SOP_OFF_TOPIC_LIMIT` (3) → the agent offers a human
representative; continuing past that → handoff. A mixed message ("…my DOB is 1985-03-15, also
what's the weather?") is handled normally with the aside declined in one clause.

### Emotional support and SOP recovery (bonus)

Emotion (`frustrated`, `angry`, `anxious`, `confused`, `refusing`, `grateful`) and intensity are
extracted each turn. The controller turns them into a **tone directive** for the responder
(acknowledge specifically first, no defensiveness, explain *why* the gate exists once, offer the
concrete alternatives, keep moving) and tracks a negative streak: after several upset turns or
`SOP_REFUSAL_LIMIT` refusals it stops persuading and offers a human — without ever disclosing
claim details or skipping verification. The example *"I already told you who I am. This is
ridiculous…"* is the `frustrated` sample caller.

### Representatives and consent

If the caller speaks for someone else, the controller checks `representatives.json`, requires the
policyholder's identifying details, then sends a (simulated) consent request whose status follows
`consent_scenarios.json`. The model cannot talk its way past a pending consent; a `timeout`
scenario escalates to a human after `SOP_CONSENT_MAX_CHECKS`.

### Defence in depth

1. The controller only puts whitelisted facts in the prompt — pre-verification there is nothing to leak.
2. Prompts state the phase policy and must-nots explicitly.
3. `LeakGuard` scans every pre-verification reply for claim identifiers, claim literals from the
   fixtures, money amounts and denial language; a hit replaces the reply with a safe templated one
   (`LEAK_BLOCKED` in the log).
4. `PIIGuard` redacts record values the caller never supplied (phone, email, DOB, ID digits).
5. The browser never receives raw PII — `verification.provided` is masked in the API response.
6. Any model failure degrades to the rule-based extractor / templates, so the SOP keeps working.

---

## 4. Project layout

```
apps/insurance_claims/fixtures/   starter data (policyholders, claims, representatives, consent, guidelines)
sop_agent/
  config.py                       env-driven settings (provider, key, models, SOP thresholds)
  data/store.py, tools.py         fixture store, claim resolution, grounded fact bundles, consent, email outbox
  sop/signals.py                  TurnSignals — the structured reading of one caller message
  sop/state.py                    Phase, Memory, VerificationState, Plan, SessionState
  sop/verification.py             identity gate (normalisation, aliases, factor counting, attempts)
  sop/controller.py               the state machine (phase handlers, interrupts, tone, handoff)
  sop/guards.py                   LeakGuard / PIIGuard
  llm/provider.py                 Anthropic · OpenAI-compatible · mock (httpx only)
  llm/extractor.py                LLM extractor + rule-based extractor
  llm/responder.py                LLM responder + template responder + email composer
  llm/prompts.py                  extractor schema, per-phase responder policy, email prompt
  api/app.py                      FastAPI routes + static UI
  harness.py, scenarios.py        wiring, session registry, sample callers
ui/                               test console (vanilla HTML/CSS/JS)
tests/                            pytest suite (mock mode; FakeProvider for the LLM plumbing)
scripts/run_scenarios.py          print annotated transcripts
```

## 5. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | provider/model in use, thresholds, phases |
| `GET` | `/api/scenarios` | sample caller scripts |
| `POST` | `/api/sessions` `{consent_scenario}` | new conversation → greeting + state |
| `POST` | `/api/sessions/{id}/messages` `{message}` | one caller turn → reply, state, trace, emails |
| `GET` | `/api/sessions/{id}` | state, transcript, per-turn traces, emails |
| `GET` | `/api/sessions/{id}/emails` | summary emails sent in this session |
| `DELETE` | `/api/sessions/{id}` | drop the session |

Interactive docs at `/docs`.

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai` (any OpenAI-compatible endpoint), `mock` |
| `LLM_API_KEY` | — | API auth token; falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` |
| `LLM_MODEL` / `LLM_EXTRACTOR_MODEL` | provider default | reply model / extraction model |
| `LLM_BASE_URL` | provider default | custom endpoint |
| `ALLOW_CLIENT_API_KEY` | `true` | accept a key from the UI per request |
| `SOP_MIN_PII_FACTORS` | `3` | matching identity factors required |
| `SOP_REP_MIN_PII_FACTORS` | `2` | policyholder factors a representative must give before consent |
| `SOP_MAX_VERIFY_ATTEMPTS` | `3` | failed attempts before verification locks |
| `SOP_OFF_TOPIC_LIMIT` | `3` | consecutive off-topic strikes before a human is offered |
| `SOP_REFUSAL_LIMIT` | `3` | refusals before the agent stops persuading |
| `SOP_CONSENT_MAX_CHECKS` | `4` | pending consent checks before escalation |
| `SOP_DEMO_DATE` | real date | fixed "today" (the sample appeal deadline is 2026-03-18) |
| `COMPANY_NAME`, `AGENT_NAME` | Harbor Mutual Insurance, Sam | persona |

## 7. Notes and limits

* Verified live against `claude-sonnet-5-5` (all eight sample callers). Two things learned the hard
  way and already handled in `llm/provider.py`: current Claude models reject the `temperature`
  parameter, and their hidden reasoning tokens count toward `max_tokens`, so the budgets are generous.

* Sessions live in memory (per process); the email outbox is a JSON file per email in `data/outbox/`.
  Swap `EmailOutbox.deliver()` for SMTP/SES and `SessionRegistry` for Redis to productionise.
* Fixture data is the only knowledge source by design; questions it cannot answer get an honest
  "I don't have that detail" plus a human offer.
* Mock mode is intentionally plain — it exists to make the SOP testable and to keep the agent
  working if the model is down, not to replace it.
