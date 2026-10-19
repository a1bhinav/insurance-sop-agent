/* Claims SOP Agent test console — vanilla JS, talks to the FastAPI backend. */
(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    provider: $("provider"), apiKey: $("apiKey"), model: $("model"), consentScenario: $("consentScenario"),
    newSession: $("newSession"), banner: $("banner"), llmPill: $("llmPill"), scenario: $("scenario"),
    nextLine: $("nextLine"), runAll: $("runAll"), scenarioDesc: $("scenarioDesc"), transcript: $("transcript"),
    composer: $("composer"), input: $("input"), send: $("send"), turnCount: $("turnCount"), phaseRail: $("phaseRail"),
    terminalNote: $("terminalNote"), factors: $("factors"), fieldList: $("fieldList"), gateMeta: $("gateMeta"),
    memory: $("memory"), casePanel: $("casePanel"), signals: $("signals"), eventLog: $("eventLog"),
    emailPanel: $("emailPanel"), emails: $("emails"), rawSignals: $("rawSignals"),
  };

  const PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"];
  const FIELD_LABELS = { full_name: "name", dob: "date of birth", phone: "phone", email: "email", id_last4: "ID last 4", policy_number: "policy no." };
  const PENDING_LABELS = {
    consent_wait: "waiting for policyholder consent", confirm_claim: "confirm which claim", choose_claim: "which claim?",
    anything_else: "anything else?", email_offer: "send email summary?", human_offer: "transfer to a human?",
  };

  let health = null;
  let session = null;        // { session_id }
  let scenarios = [];
  let queue = [];            // remaining scripted lines
  let busy = false;
  let eventHistory = [];     // [{turn, events}]

  // ------------------------------------------------------------ helpers
  function headers() {
    const h = { "Content-Type": "application/json" };
    const key = el.apiKey.value.trim();
    if (key) h["X-LLM-API-Key"] = key;
    if (el.provider.value) h["X-LLM-Provider"] = el.provider.value;
    if (el.model.value.trim()) h["X-LLM-Model"] = el.model.value.trim();
    return h;
  }
  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    return res.json();
  }
  function showBanner(text, kind = "") {
    el.banner.textContent = text;
    el.banner.className = "banner" + (kind ? " " + kind : "");
    el.banner.hidden = !text;
  }
  function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function phaseLabel(p) { return ({ VERIFY_ID: "Verify identity", RESOLVE_INTENT: "Resolve intent", PROCESS_CASE: "Process case", POST_PROCESS: "Follow-up", CLOSED: "Closed", HUMAN_HANDOFF: "Human handoff" })[p] || p; }

  // persist the key for this tab only
  try { el.apiKey.value = sessionStorage.getItem("llm_api_key") || ""; el.provider.value = sessionStorage.getItem("llm_provider") || ""; el.model.value = sessionStorage.getItem("llm_model") || ""; } catch (_) { /* private mode */ }
  for (const [node, key] of [[el.apiKey, "llm_api_key"], [el.provider, "llm_provider"], [el.model, "llm_model"]]) {
    node.addEventListener("change", () => { try { sessionStorage.setItem(key, node.value); } catch (_) { /* ignore */ } refreshHealth(); });
  }

  // ------------------------------------------------------------ health / setup
  async function refreshHealth() {
    try {
      health = await api("/api/health");
      const llm = health.llm;
      el.llmPill.textContent = llm.provider === "mock" ? "Mock mode · rule-based, no model" : `${llm.provider} · ${llm.model}`;
      el.llmPill.className = "llm-pill" + (llm.provider === "mock" ? " mock" : "");
      if (llm.provider === "mock") {
        showBanner("Running in mock mode: a rule-based extractor and templated replies drive the SOP. Add a model API key above (or set LLM_API_KEY on the server) to let the model interpret and phrase.");
      } else {
        showBanner("");
      }
      if (!el.consentScenario.options.length) {
        for (const s of health.consent_scenarios) {
          const o = document.createElement("option"); o.value = s; o.textContent = s === "default" ? "default (approved on 2nd check)" : s; el.consentScenario.appendChild(o);
        }
      }
    } catch (e) {
      el.llmPill.textContent = "server unreachable";
      showBanner(`Cannot reach the API: ${e.message}`, "error");
    }
  }

  async function loadScenarios() {
    const data = await api("/api/scenarios");
    scenarios = data.scenarios;
    for (const s of scenarios) {
      const o = document.createElement("option"); o.value = s.key; o.textContent = s.title; el.scenario.appendChild(o);
    }
  }

  // ------------------------------------------------------------ session
  async function newSession() {
    const consent = el.consentScenario.value || "default";
    const data = await api("/api/sessions", { method: "POST", body: JSON.stringify({ consent_scenario: consent }) });
    session = { id: data.session_id };
    eventHistory = [];
    el.transcript.innerHTML = "";
    el.emails.innerHTML = ""; el.emailPanel.hidden = true;
    el.rawSignals.textContent = "";
    addMessage("agent", data.greeting, data.state.phase);
    renderState(data.state, null);
    el.input.focus();
  }

  function addMessage(role, text, phase, extraClass = "") {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role} ${extraClass}`.trim();
    const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text;
    const stamp = document.createElement("div"); stamp.className = "stamp";
    const who = document.createElement("span"); who.textContent = role === "user" ? "Caller" : (health?.agent?.name || "Agent");
    stamp.appendChild(who);
    if (phase) {
      const ph = document.createElement("span");
      ph.className = "phase" + (phase === "HUMAN_HANDOFF" ? " alert" : (phase === "PROCESS_CASE" || phase === "POST_PROCESS" || phase === "CLOSED") ? " verified" : "");
      ph.textContent = phaseLabel(phase);
      stamp.appendChild(ph);
    }
    wrap.appendChild(bubble); wrap.appendChild(stamp);
    el.transcript.appendChild(wrap);
    el.transcript.scrollTop = el.transcript.scrollHeight;
    return wrap;
  }

  async function sendMessage(text) {
    if (!text.trim() || busy) return;
    if (!session) await newSession();
    busy = true; el.send.disabled = true; el.nextLine.disabled = true; el.runAll.disabled = true;
    addMessage("user", text, null);
    const pending = addMessage("agent", "…", null, "pending");
    try {
      const data = await api(`/api/sessions/${session.id}/messages`, { method: "POST", body: JSON.stringify({ message: text }) });
      pending.remove();
      addMessage("agent", data.reply, data.state.phase);
      eventHistory.push({ turn: data.trace.turn, events: data.trace.events });
      renderState(data.state, data.trace);
      renderEmails(data.emails);
      if (data.llm && health && data.llm.provider !== health.llm.provider) refreshHealth();
    } catch (e) {
      pending.remove();
      showBanner(`The agent could not answer: ${e.message}`, "error");
    } finally {
      busy = false; el.send.disabled = false;
      el.nextLine.disabled = queue.length === 0; el.runAll.disabled = queue.length === 0;
    }
  }

  // ------------------------------------------------------------ rendering
  function renderState(s, trace) {
    el.turnCount.textContent = s.turn_count ? `${s.turn_count} caller turn${s.turn_count === 1 ? "" : "s"}` : "new conversation";

    // phase rail
    const idx = PHASES.indexOf(s.phase);
    for (const li of el.phaseRail.querySelectorAll("li")) {
      const i = PHASES.indexOf(li.dataset.phase);
      li.classList.toggle("active", li.dataset.phase === s.phase);
      li.classList.toggle("done", (idx === -1 ? (s.phase === "CLOSED" ? i <= 3 : s.verification.verified && i === 0) : i < idx));
    }
    if (s.phase === "HUMAN_HANDOFF") {
      el.terminalNote.hidden = false; el.terminalNote.className = "terminal-note handoff";
      el.terminalNote.textContent = `Handed to a human representative (${s.handoff_reference}) — ${s.handoff_reason || ""}`;
    } else if (s.phase === "CLOSED") {
      el.terminalNote.hidden = false; el.terminalNote.className = "terminal-note closed";
      el.terminalNote.textContent = s.email_sent_id ? "Conversation closed · summary email sent" : "Conversation closed · caller skipped the email";
    } else { el.terminalNote.hidden = true; }

    // identity gate
    const v = s.verification;
    const required = v.caller_role === "representative" ? (health?.thresholds?.rep_min_pii_factors ?? 2) : (health?.thresholds?.min_pii_factors ?? 3);
    let f = "";
    for (let i = 0; i < required; i++) f += `<span class="factor ${i < v.matched_count ? "on" : ""}">${i < v.matched_count ? "✓" : ""}</span>`;
    f += `<span class="factor-label">${v.matched_count} of ${required} identifying details matched${v.caller_role === "representative" ? " (representative call)" : ""}</span>`;
    if (v.verified) f += `<span class="verified-tag">verified${v.caller_role === "representative" ? " · consent approved" : ""}</span>`;
    else if (v.locked) f += `<span class="locked-tag">locked</span>`;
    el.factors.innerHTML = f;
    const fields = ["full_name", "dob", "phone", "email", "id_last4", "policy_number"];
    el.fieldList.innerHTML = fields.map((k) => {
      const given = k in v.provided;
      const cls = v.matched.includes(k) ? "matched" : v.mismatched.includes(k) ? "mismatched" : given ? "given" : "";
      const mark = v.matched.includes(k) ? "✓ " : v.mismatched.includes(k) ? "✕ " : given ? "· " : "";
      return `<li class="${cls}" title="${k === "policy_number" ? "locates the record; does not count as a factor" : ""}">${mark}${FIELD_LABELS[k]}${k === "policy_number" && given ? " (locator)" : ""}</li>`;
    }).join("");
    const bits = [];
    if (v.attempts) bits.push(`${v.attempts} of ${health?.thresholds?.max_verify_attempts ?? 3} failed attempts`);
    if (v.refusals) bits.push(`${v.refusals} refusal${v.refusals === 1 ? "" : "s"}`);
    if (v.caller_role === "representative") bits.push(`representative ${v.rep_name || ""}${v.rep_relationship ? " (" + v.rep_relationship + ")" : ""}${v.rep_on_file === true ? " · on file" : v.rep_on_file === false ? " · not on file" : ""}`);
    if (v.consent) bits.push(`consent ${v.consent.consent_id}: ${v.consent.status} after ${v.consent.checks} check${v.consent.checks === 1 ? "" : "s"}`);
    el.gateMeta.textContent = bits.join(" · ");

    // memory
    const m = s.memory;
    const rows = [];
    rows.push(row("intent", m.intent_hints.length ? chips(m.intent_hints.map((x) => x.replace(/_/g, " "))) : empty("nothing yet")));
    const ch = Object.entries(m.case_hints).map(([k, val]) => `${k.replace(/_/g, " ")}: ${val}`);
    rows.push(row("claim hints", ch.length ? chips(ch) : empty("nothing yet")));
    rows.push(row("held questions", m.pending_questions.length ? chips(m.pending_questions, "quiet") : empty("none")));
    if (m.email_preference) rows.push(row("email", chips([m.email_preference === "yes" ? "wants summary" : "no email"], "quiet")));
    el.memory.innerHTML = rows.join("");

    // case
    if (!v.verified) {
      el.casePanel.innerHTML = `<div class="locked">Claim data stays out of the model's context until identity is verified.</div>`;
    } else if (s.selected_case_id) {
      el.casePanel.innerHTML = `<div class="claim"><span class="id">${esc(s.selected_case_id)}</span>${s.claims_discussed.length > 1 ? `<div class="meta">also discussed: ${esc(s.claims_discussed.filter((c) => c !== s.selected_case_id).join(", "))}</div>` : ""}` +
        `<div class="meta">topics: ${esc(s.topics_covered.map((t) => t.replace(/_/g, " ")).join(", ") || "—")}</div></div>`;
    } else {
      el.casePanel.innerHTML = `<div class="locked">Verified — no claim selected yet.</div>`;
    }

    // signals
    const sig = [];
    sig.push(["last emotion", s.last_emotion + (s.negative_streak ? ` <span class="warn">(upset for ${s.negative_streak} turn${s.negative_streak === 1 ? "" : "s"})</span>` : "")]);
    sig.push(["off-topic", s.off_topic_strikes ? `<span class="warn">${s.off_topic_strikes} of ${health?.thresholds?.off_topic_limit ?? 3} strikes</span>` : `<span class="ok">none</span>${s.off_topic_total ? ` (${s.off_topic_total} total)` : ""}`]);
    sig.push(["waiting on", s.pending_question ? esc(PENDING_LABELS[s.pending_question] || s.pending_question) : "—"]);
    if (s.human_offered) sig.push(["human offered", "yes"]);
    if (trace?.signals) sig.push(["extraction", trace.signals.source === "llm" ? "model" : trace.signals.source === "fallback" ? '<span class="warn">rules (model call failed)</span>' : "rules (mock)"]);
    el.signals.innerHTML = sig.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("");

    // event log
    el.eventLog.innerHTML = eventHistory.slice(-6).map(({ turn, events }) =>
      `<li class="turn">Turn ${turn}</li>` + events.map((e) => `<li class="${evClass(e)}">${esc(e)}</li>`).join("")
    ).join("") || `<li class="empty">Gate and guard events appear here after each caller turn.</li>`;
    el.eventLog.scrollTop = el.eventLog.scrollHeight;

    if (trace) el.rawSignals.textContent = JSON.stringify({ signals: trace.signals, directive: trace.directive, facts_shared_with_model: trace.facts_keys, tone: trace.tone }, null, 2);
  }
  function evClass(e) {
    if (e.startsWith("LEAK_BLOCKED") || e.startsWith("PII_REDACTED")) return "block";
    if (e.includes("PASSED")) return "pass";
    if (e.startsWith("TRANSITION")) return "transition";
    if (e.startsWith("GATE") || e.startsWith("CONSENT") || e.startsWith("REP") || e.startsWith("OFF_TOPIC") || e.startsWith("REFUSAL")) return "gate";
    if (e.startsWith("MEMORY")) return "memory";
    return "";
  }
  function row(k, v) { return `<div class="row"><span class="k">${k}</span><span>${v}</span></div>`; }
  function chips(items, extra = "") { return `<span class="chips">${items.map((i) => `<span class="chip ${extra}">${esc(i)}</span>`).join("")}</span>`; }
  function empty(t) { return `<span class="empty">${t}</span>`; }

  function renderEmails(emails) {
    if (!emails || !emails.length) return;
    el.emailPanel.hidden = false;
    el.emails.innerHTML = emails.map((e) =>
      `<div class="email"><div class="hdr"><span>To</span><b>${esc(e.to)}</b><span>Subject</span><b>${esc(e.subject)}</b><span>Sent</span><b>${esc(e.sent_at)}</b></div><pre>${esc(e.body)}</pre></div>`
    ).join("");
  }

  // ------------------------------------------------------------ sample scripts
  el.scenario.addEventListener("change", () => {
    const s = scenarios.find((x) => x.key === el.scenario.value);
    queue = s ? [...s.turns] : [];
    el.scenarioDesc.textContent = s ? s.description : "";
    el.nextLine.disabled = !queue.length; el.runAll.disabled = !queue.length;
    if (s?.consent_scenario) el.consentScenario.value = s.consent_scenario;
    if (s) { newSession().catch((e) => showBanner(e.message, "error")); }
  });
  el.nextLine.addEventListener("click", async () => {
    if (!queue.length) return;
    const line = queue.shift();
    await sendMessage(line);
  });
  el.runAll.addEventListener("click", async () => {
    while (queue.length && !busy) {
      const line = queue.shift();
      await sendMessage(line);
      await new Promise((r) => setTimeout(r, 400));
    }
  });

  // ------------------------------------------------------------ composer
  el.composer.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = el.input.value;
    el.input.value = "";
    await sendMessage(text);
  });
  el.input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); el.composer.requestSubmit(); }
  });
  el.newSession.addEventListener("click", () => { queue = []; el.scenario.value = ""; el.scenarioDesc.textContent = ""; el.nextLine.disabled = true; el.runAll.disabled = true; newSession().catch((e) => showBanner(e.message, "error")); });

  // ------------------------------------------------------------ boot
  (async () => {
    await refreshHealth();
    try { await loadScenarios(); } catch (e) { showBanner(e.message, "error"); }
    if (health) await newSession().catch((e) => showBanner(e.message, "error"));
  })();
})();
