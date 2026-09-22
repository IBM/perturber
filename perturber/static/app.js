const $ = (id) => document.getElementById(id);

// --- Theme: dark (default) / light, persisted; first visit follows the OS preference. ---
const THEME_KEY = "perturber.theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "light" ? "☀ Light" : "🌙 Dark";
}
function initTheme() {
  let theme;
  try { theme = localStorage.getItem(THEME_KEY); } catch (e) { /* storage disabled */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }
  applyTheme(theme);
}
function toggleTheme() {
  const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage disabled */ }
}
initTheme();
// A perturbation name is unique only within its category, so the playground keys everything by
// "category/name" (this distribution ships only the native category).
const perturbationKey = (category, name) => `${category}/${name}`;
// Model-backed perturbations require an API key. They are exactly the ones exposing a `model`
// param (the model id), across the paraphrasing, reasoning, and context families; this keys on
// that param so it covers every current and future LLM perturbation without enumerating families.
const isLlmPerturbation = (p) => !!(p && p.params && p.params.model);
// Initial selection when no shared URL or persisted state applies.
const DEFAULT_KEY = perturbationKey("native", "realistic_typos");
// Family display order in the dropdown, by conceptual weight (fundamental first, model-backed and
// structural last). Families not listed here fall back to alphabetical after these.
const FAMILY_ORDER = [
  "control", "word_level", "character", "formatting", "schema", "normalization",
  "padding", "context", "paraphrasing", "adversarial",
];
let CATALOG = {};        // "category/name" mapped to perturbation metadata
let flatList = [];       // [{category, name, ...}]
let HISTORY = [];        // snapshots of successful runs, newest first
let activeRun = null;    // id of the currently displayed run

// The model dropdown is populated from the perturbation's static `model` param choices (rendered
// by the generic param builder), so there is no live model fetch. Any model id the configured
// endpoint serves can be selected from that list, or set via the PERTURBER_LLM_MODELS env var.

async function loadVersion() {
  // Best-effort: show the deployed version in the header. A failure here must not block the app.
  try {
    const res = await fetch("/version");
    const data = await res.json();
    if (data && data.version) $("version").textContent = "v" + data.version;
  } catch (e) {
    /* leave the version blank if unavailable */
  }
}

async function loadCatalog() {
  const res = await fetch("/perturbations");
  const grouped = await res.json();
  const sel = $("perturbation");
  sel.innerHTML = "";
  flatList = [];
  CATALOG = {};
  // Group options by (category, family) so the long list is scannable. HTML optgroups can't
  // nest, so each subsection is one optgroup labelled "category · family".
  for (const [category, perts] of Object.entries(grouped)) {
    const byFamily = {};
    for (const p of perts) {
      p.category = category;
      const key = perturbationKey(category, p.name);
      CATALOG[key] = p;
      flatList.push(p);
      // Out-of-scope perturbations stay in CATALOG (so a shared URL can still resolve and run
      // them) but are not offered as choices in the dropdown.
      if (p.in_scope === false) continue;
      (byFamily[p.family] ||= []).push(p);
    }
    const familyRank = (f) => {
      const i = FAMILY_ORDER.indexOf(f);
      return i === -1 ? FAMILY_ORDER.length : i;
    };
    const families = Object.keys(byFamily).sort(
      (a, b) => familyRank(a) - familyRank(b) || a.localeCompare(b)
    );
    for (const family of families) {
      const group = document.createElement("optgroup");
      group.label = `${category} · ${family}`;
      for (const p of byFamily[family]) {
        const opt = document.createElement("option");
        opt.value = perturbationKey(category, p.name);
        // Mark model-backed perturbations so they are distinguishable in the list.
        opt.textContent = isLlmPerturbation(p) ? `${p.name} (LLM)` : p.name;
        group.appendChild(opt);
      }
      sel.appendChild(group);
    }
  }
  // Prefer native/realistic_typos as the initial selection when present; otherwise fall back to
  // the first catalog entry. A shared URL or persisted state (applied on load) overrides this.
  if (CATALOG[DEFAULT_KEY]) {
    $("perturbation").value = DEFAULT_KEY;
  }
  if (flatList.length) onSelect();
}

function onSelect() {
  const p = CATALOG[$("perturbation").value];
  $("desc").innerHTML =
    `<span class="badge">${p.category}</span><span class="badge">${p.family}</span>` +
    (p.seed_sensitive ? `<span class="badge">seed-sensitive</span>` : "") +
    `<div class="desc-text">${p.description || ""}</div>`;

  // Model-backed perturbations expose the exact system prompt; show it just above the text box.
  $("prompt-block").innerHTML = p.prompt
    ? `<details class="prompt-details"><summary>View system prompt</summary>` +
      `<div class="prompt-note">Sent as the <code>system</code> message; your text is the ` +
      `<code>user</code> message.</div>` +
      `<div class="prompt-text">${escapeHtml(p.prompt)}</div></details>`
    : "";

  const sensitive = !!p.seed_sensitive;
  const llm = isLlmPerturbation(p);
  // LLM perturbations hide the seed section entirely (the seed is meaningless for a model call).
  // Other non-seed-sensitive perturbations keep the greyed-out controls with an explanatory note.
  $("seed-block").style.display = llm ? "none" : "";
  if (!llm) {
    $("seed").disabled = !sensitive;
    $("randomize").disabled = !sensitive;
    $("autorandom").disabled = !sensitive;
    $("seed-note").textContent = sensitive
      ? ""
      : "Seed has no effect, this perturbation is deterministic.";
  }

  // Model-backed perturbations (paraphrasing family + emotion_prompt) need an API key.
  $("llm-row").style.display = llm ? "" : "none";

  // Offer the sample that best shows off this family, but only when the box still holds a built-in
  // sample (never overwrite the user's own text).
  const isSchema = p.family === "schema";
  if (textIsSample()) {
    $("texts").value = sampleFor(p);
  }
  // Schema input and output are multi-line JSON, so give both boxes more room while active.
  $("texts").style.minHeight = isSchema ? "340px" : "";
  $("output").style.minHeight = isSchema ? "340px" : "";

  const box = $("params");
  box.innerHTML = "";
  for (const [name, spec] of Object.entries(p.params || {})) {
    const label = document.createElement("label");
    label.textContent = name;
    label.htmlFor = "param-" + name;
    box.appendChild(label);

    let field;
    if (Array.isArray(spec.choices) && spec.choices.length) {
      // A closed set of allowed values renders as a dropdown.
      field = document.createElement("select");
      for (const choice of spec.choices) {
        const opt = document.createElement("option");
        opt.value = choice;
        opt.textContent = choice;
        field.appendChild(opt);
      }
      field.value = spec.default ?? spec.choices[0];
    } else if (spec.type === "float") {
      // Float params render as a 0 to 2 slider paired with an editable number box; the two stay in
      // sync. Only the slider carries data-param (the single source of truth for collectParams).
      const dflt = spec.default ?? 0;
      field = document.createElement("input");
      field.type = "range";
      field.min = "0"; field.max = "2"; field.step = "0.01";
      field.value = dflt;
      const num = document.createElement("input");
      num.type = "number";
      num.min = "0"; num.max = "2"; num.step = "0.01";
      num.className = "slider-val";
      num.value = dflt;
      // Keep the two in sync; both fire input so persist() (bound on #params) still runs.
      field.addEventListener("input", () => { num.value = field.value; });
      num.addEventListener("input", () => { field.value = num.value; });
      const wrap = document.createElement("div");
      wrap.className = "slider-row";
      field.id = "param-" + name;
      field.dataset.param = name;
      field.dataset.type = spec.type;
      wrap.appendChild(field);
      wrap.appendChild(num);
      box.appendChild(wrap);
      continue;
    } else {
      field = document.createElement("input");
      field.type = (spec.type === "int") ? "number" : "text";
      field.value = spec.default ?? "";
    }
    field.id = "param-" + name;
    field.dataset.param = name;
    field.dataset.type = spec.type;
    // The model param renders as a dropdown from its static `model` choices (via the choices
    // branch above); no live model fetch.
    if (name === "model") field.dataset.modelSelect = "1";
    box.appendChild(field);
  }
}

function collectParams() {
  const params = {};
  document.querySelectorAll("#params input, #params select").forEach((el) => {
    const name = el.dataset.param;
    if (!name) return;  // e.g. the paired number box of a float slider carries no data-param
    const t = el.dataset.type;
    let v = el.value;
    if (t === "int") v = parseInt(v, 10);
    else if (t === "float") v = parseFloat(v);
    else if (t === "bool") v = ["1", "true", "yes", "on"].includes(String(v).toLowerCase());
    params[name] = v;
  });
  return params;
}

// A fresh random seed for each run; the same (text, name, params, seed) is always reproducible,
// and the history list lets any earlier seed be restored.
function randomSeed() {
  return Math.floor(Math.random() * 2147483648); // [0, 2^31)
}

let lastResultTexts = null;  // the texts currently shown, for the Copy button

function showResult(texts, category, name, params, seed, elapsed) {
  $("output").className = "out";
  // Number the results only when there is more than one input line; a single result reads
  // better without a "1." prefix.
  const multi = texts.length > 1;
  $("output").innerHTML = texts
    .map((t, i) => {
      const idx = multi ? `<span class="idx">${i + 1}. </span>` : "";
      return `<div class="result-item">${idx}${escapeHtml(t)}</div>`;
    })
    .join("");
  // LLM (model-backed) perturbations are seed-independent, so omit the seed from the meta line.
  const isLlm = params && params.temperature !== undefined;
  const timing = formatElapsed(elapsed);
  $("meta").textContent = `${category}/${name} · params ${JSON.stringify(params)}` +
    (isLlm ? "" : ` · seed ${seed}`) +
    (timing ? ` · ${timing}` : "");
  lastResultTexts = texts;
  $("copy").disabled = false;
}

// Summarise the per-text elapsed_ms array for display: the total across all inputs.
function formatElapsed(elapsed) {
  if (!Array.isArray(elapsed) || !elapsed.length) return "";
  const total = elapsed.reduce((a, b) => a + b, 0);
  return `${total.toFixed(total < 100 ? 1 : 0)} ms`;
}

// Copy the current result (one line per text) to the clipboard.
async function copyResult() {
  if (!lastResultTexts || !lastResultTexts.length) return;
  const text = lastResultTexts.join("\n");
  const btn = $("copy");
  const original = btn.textContent;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied!";
  } catch (e) {
    window.prompt("Copy the result:", text);
    return;
  }
  setTimeout(() => { btn.textContent = original; }, 1200);
}

function _histItem(run) {
  const item = document.createElement("button");
  item.className = "hist-item";
  item.dataset.id = run.id;
  item.title = "Restore this run";
  const preview = run.result.texts[0] ?? "";
  // LLM runs are seed-independent; show the temperature (which shapes the output) instead.
  const lead = (run.params && run.params.temperature !== undefined)
    ? `t=${run.params.temperature}`
    : `s=${run.seed}`;
  const timing = formatElapsed(run.result.elapsed_ms);
  item.innerHTML =
    `<span class="seed">${lead}</span>` +
    `<span class="name">${run.name}</span>` +
    `<span class="preview">${escapeHtml(preview)}</span>` +
    (timing ? `<span class="hist-time">${timing}</span>` : "");
  item.addEventListener("click", () => restoreRun(run.id));
  return item;
}

// Reconcile the history list against HISTORY instead of rebuilding it wholesale, so pressing
// Perturb only prepends the new entry (and re-points the active highlight) rather than
// repainting every row, which was visibly flickering.
function renderHistory() {
  const box = $("history");
  const existing = new Map([...box.children].map((el) => [el.dataset.id, el]));
  let ref = box.firstChild;
  for (const run of HISTORY) {
    let el = existing.get(run.id);
    if (el) {
      existing.delete(run.id);
      if (el !== ref) box.insertBefore(el, ref);  // move into order only if needed
      ref = el.nextSibling;
    } else {
      el = _histItem(run);
      box.insertBefore(el, ref);  // new entries land in their HISTORY position (newest first)
    }
    el.classList.toggle("active", run.id === activeRun);
  }
  // Remove any items no longer in HISTORY (e.g. after Clear).
  for (const el of existing.values()) el.remove();
  $("hist-hint").textContent = HISTORY.length ? `(${HISTORY.length})` : "";
}

function restoreRun(id) {
  const run = HISTORY.find((r) => r.id === id);
  if (!run) return;
  // Restore the full input snapshot, then re-render the stored result without re-calling.
  $("perturbation").value = perturbationKey(run.category, run.name);
  onSelect(); // rebuild param inputs for this perturbation
  for (const [k, v] of Object.entries(run.params)) {
    const el = document.querySelector(`#params [data-param="${k}"]`);
    if (el) {
      el.value = v;
      // Notify paired widgets (e.g. a float slider's number box syncs on input).
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
  $("seed").value = run.seed;
  $("texts").value = run.texts.join("\n");
  activeRun = id;
  showResult(run.result.texts, run.result.category, run.name, run.result.params, run.seed,
             run.result.elapsed_ms);
  renderHistory();
  persist();
}

// Perturb uses the seed currently on screen. When "auto" is checked, the seed is randomized for
// this run instead. An explicit seed (e.g. reproducing a shared URL) always wins and is used as-is.
async function run(explicitSeed) {
  const p = CATALOG[$("perturbation").value];
  // Most perturbations treat each line as a separate input. A schema perturbation's input is one
  // multi-line JSON document, so send the whole box as a single text (never split on newlines).
  const isSchema = p.family === "schema";
  const texts = isSchema ? [$("texts").value] : $("texts").value.split("\n");
  const params = collectParams();
  let seed;
  if (explicitSeed !== undefined && explicitSeed !== null) {
    seed = explicitSeed;
  } else if ($("autorandom").checked && p.seed_sensitive) {
    // Auto-randomize only affects seed-sensitive perturbations; a deterministic one ignores it.
    seed = randomSeed();
  } else {
    seed = parseInt($("seed").value, 10) || 0;
  }
  $("seed").value = seed; // surface the seed actually used
  const out = $("output");

  $("run").disabled = true;
  // Only show the "Perturbing…" placeholder if the request is actually slow. For the usual
  // near-instant local response, swap the result in place with no intermediate blank frame,
  // which was causing a visible flicker on every click.
  const pending = setTimeout(() => {
    out.className = "out";
    out.textContent = "Perturbing…";
    $("meta").textContent = "";
  }, 150);
  try {
    const headers = { "content-type": "application/json" };
    // Model-backed perturbations authenticate with a per-request API key sent as an
    // Authorization: Bearer header (never in the body, so it is not echoed or shared). An
    // optional base URL overrides the endpoint via X-LLM-Base-URL.
    if (isLlmPerturbation(p)) {
      const key = $("llm-key").value.trim();
      if (key) headers["Authorization"] = "Bearer " + key;
      const baseUrl = $("llm-base-url").value.trim();
      if (baseUrl) headers["X-LLM-Base-URL"] = baseUrl;
    }
    const res = await fetch(`/perturb/${p.category}/${p.name}`, {
      method: "POST",
      headers,
      body: JSON.stringify({ texts, seed, ...params }),
    });
    clearTimeout(pending);
    const data = await res.json();
    if (!res.ok) {
      out.className = "out err";
      // FastAPI errors carry a `detail`; show that plainly, else the raw payload.
      out.textContent = typeof data.detail === "string" ? data.detail : JSON.stringify(data, null, 2);
      return;
    }
    showResult(data.texts, data.category, data.name, data.params, data.seed, data.elapsed_ms);
    // Record a full snapshot so this exact run can be restored later, unless it is identical to
    // the most recent entry (same perturbation, params, seed, and text), which for a deterministic
    // perturbation would only produce the same result. Model-backed (LLM) perturbations are
    // non-deterministic, so identical inputs still yield new output: never dedup those.
    const prev = HISTORY[0];
    const isDuplicate =
      prev &&
      !isLlmPerturbation(p) &&
      prev.category === data.category &&
      prev.name === data.name &&
      prev.seed === data.seed &&
      JSON.stringify(prev.texts) === JSON.stringify(texts) &&
      JSON.stringify(prev.params) === JSON.stringify(data.params);
    if (isDuplicate) {
      activeRun = prev.id;
    } else {
      const id = Date.now() + "-" + Math.random().toString(36).slice(2, 7);
      HISTORY.unshift({
        id, seed: data.seed, category: data.category, name: data.name, texts, params: data.params, result: data,
      });
      activeRun = id;
    }
    renderHistory();
    // The seed box keeps the seed that was actually used (already set above), so it always matches
    // the recorded history entry. With "auto" on, the next click randomizes afresh at run time.
    persist();
  } catch (e) {
    clearTimeout(pending);
    out.className = "out err";
    out.textContent = "Request failed: " + e.message;
  } finally {
    $("run").disabled = false;
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// --- Current form state (the "perturb instance"): perturbation, params, seed, texts ---
// `autorandom` is a UI preference persisted locally; it is intentionally left out of the shareable
// URL (see shareUrl) so a shared link reproduces an exact seed rather than imposing auto-randomize.
function currentState() {
  const p = CATALOG[$("perturbation").value];
  return {
    category: p ? p.category : undefined,
    name: p ? p.name : undefined,
    params: collectParams(),
    seed: parseInt($("seed").value, 10) || 0,
    texts: $("texts").value,
    autorandom: $("autorandom").checked,
  };
}

// Apply a form state. Rebuilds the param inputs for the chosen perturbation before filling them.
// Resolves by (category, name); falls back to the first matching name if the category is missing
// (e.g. an older shared link that predates category-qualified URLs).
function applyState(s) {
  if (!s || !s.name) return false;
  let key = perturbationKey(s.category, s.name);
  if (!CATALOG[key]) {
    const match = flatList.find((p) => p.name === s.name);
    if (!match) return false;
    key = perturbationKey(match.category, match.name);
  }
  $("perturbation").value = key;
  onSelect();
  for (const [k, v] of Object.entries(s.params || {})) {
    const el = document.querySelector(`#params [data-param="${k}"]`);
    if (el) {
      el.value = v;
      // Notify paired widgets (e.g. a float slider's number box syncs on input).
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
  if (s.seed !== undefined) $("seed").value = s.seed;
  if (s.texts !== undefined) $("texts").value = s.texts;
  if (s.autorandom !== undefined) $("autorandom").checked = s.autorandom;
  return true;
}

// --- Persistence (localStorage; degrades to in-memory if unavailable) ---
const STORE_KEY = "perturber.v2";
function persist() {
  syncUrl();
  try {
    // The API key is stored separately from `form` so it never flows into shareUrl/currentState.
    localStorage.setItem(STORE_KEY, JSON.stringify({
      form: currentState(), history: HISTORY,
      llmKey: $("llm-key").value, llmBaseUrl: $("llm-base-url").value,
    }));
  } catch (e) { /* storage disabled or full; keep working in-memory */ }
}
function loadPersisted() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function clearHistory() {
  HISTORY = [];
  activeRun = null;
  renderHistory();
  persist();
}

// The general sample text the form starts with; also used by Reset and as the fallback for any
// family without its own example.
const DEFAULT_TEXT =
  "According to all known laws of aviation, there is no way that a bee should be able to fly. " +
  "Its wings are too small to get its fat little body off the ground.";

// Schema-family perturbations take a JSON Schema (the value for response_format.json_schema.schema).
const SCHEMA_DEFAULT_TEXT = JSON.stringify({
  type: "object",
  properties: {
    reasoning: { type: "string" },
    answer: { type: "string" },
    confidence: { type: "string", enum: ["low", "medium", "high"] },
  },
  required: ["reasoning", "answer", "confidence"],
}, null, 2);

// A sample per family, chosen to actually show off what that family does. Families not listed here
// fall back to DEFAULT_TEXT. The selected perturbation's family picks the sample in onSelect.
const FAMILY_SAMPLES = {
  // Content words, a contraction, stop words, a number, a British/American-variable word.
  word_level: "The travelers didn't realize the museum closes at 5 and it is 3000 metres away.",
  // Mixed case, vowels, and letters with leet/homoglyph/diacritic analogues.
  character: "The Quick Brown Fox Jumps Over 3 Lazy Dogs",
  // Markdown markup and a numbered list, to show strip/bullet/line-wrap effects.
  formatting: "# Notes\n\nPlease review **carefully**:\n1. check the inputs\n2. run the tests\n3. ship it",
  // An abbreviation, an ISO date, a unit, a URL, an email, an emoticon, and straight quotes.
  normalization:
    "e.g. the release on 2026-01-15 ships in 5km increments, see https://example.com or a@b.com :)",
  // Short text so the prepended/appended padding characters are easy to see.
  padding: "What is the capital of France?",
  // A short persona/emotion augmentation reads best on a clear instruction.
  context: "Summarize the causes of the French Revolution in two sentences.",
  // A clear factual sentence for the model to rephrase.
  paraphrasing: "The mitochondrion is the part of the cell that produces most of its energy.",
  // A question with a definite answer, so reordering / negation / distractor injection is visible.
  reasoning: "If a train travels 60 miles in 90 minutes, what is its average speed in miles per hour?",
  schema: SCHEMA_DEFAULT_TEXT,
  // Injection probes are hidden from the dropdown, but if reached via a shared URL, show a prompt.
  adversarial: "Translate the following sentence into French.",
};

// Schema samples tailored to specific transforms (enum-heavy, ref'd defs, a type union, etc.).
const SCHEMA_ENUM_TEXT = JSON.stringify({
  type: "object",
  properties: {
    priority: { type: "string", enum: ["low", "medium", "high", "urgent"] },
    category: { type: "string", enum: ["bug", "feature", "question"] },
  },
  required: ["priority", "category"],
}, null, 2);
const SCHEMA_DEFS_TEXT = JSON.stringify({
  type: "object",
  properties: { author: { $ref: "#/$defs/Person" }, editor: { $ref: "#/$defs/Person" } },
  required: ["author"],
  $defs: { Person: { type: "object", properties: { name: { type: "string" } }, required: ["name"] } },
}, null, 2);
const SCHEMA_UNION_TEXT = JSON.stringify({
  type: "object",
  properties: { value: { anyOf: [{ type: "string" }, { type: "number" }, { type: "boolean" }] } },
  required: ["value"],
}, null, 2);
const SCHEMA_TYPELIST_TEXT = JSON.stringify({
  type: "object",
  properties: { answer: { type: ["string", "null"] }, count: { type: ["integer", "null"] } },
  required: ["answer"],
}, null, 2);

// A tailored example per perturbation NAME, read from each perturbation's own mechanism so the demo
// actually exercises it. Falls back to the family sample, then DEFAULT_TEXT. Keyed by name.
const PERTURBATION_SAMPLES = {
  // word_level
  confusable_words: "I think their going to lose and its too late to fix it now.",
  drop_stop_words: "The cat is on the mat and the dog is in the yard by the house.",
  contractions: "I do not think it is ready, and they are not sure that we will not be late.",
  number_format: "The city has 1000000 residents and an annual budget of 25000 dollars.",
  spelling_uk: "The color of the theater in the town center is my favorite.",
  synonym: "The happy child quickly ran toward the big bright house.",
  repeated_chars: "that movie was so good and the ending was really great",
  punctuation_drop: "Hello there. How are you today? I am doing very well.",
  punctuation_spaces: "Wait, really? Yes! I can't believe it; that is amazing.",
  word_merge: "The report is due on Friday afternoon without fail.",
  word_split: "The report is due on Friday afternoon without fail.",
  typos: "The quick brown fox jumps over the lazy dog near the river.",
  realistic_typos: "The quick brown fox jumps over the lazy dog near the river.",
  sequence_spaces: "The quick brown fox jumps over the lazy dog.",
  // character
  diacritics_strip: "The café owner prepared a soufflé and a piña colada for the naïve tourist.",
  diacritics_add: "The quick brown fox jumps over the lazy dog.",
  disemvowel: "The quick brown fox jumps over the lazy dog.",
  // normalization: one clear target each
  abbreviations: "Bring water, snacks, etc. to the meeting, e.g. granola bars, i.e. something light.",
  date_unit: "The launch on 2026-01-15 covered 5km in under 30min at 12kg of thrust.",
  emoji: "Great work today :) the release went smoothly :D and nobody was sad :(",
  emoticon: "Great work today 🙂 the release went smoothly 😀 and nobody was sad 🙁",
  quote_style: 'She said "hello" and asked if it\'s really Tom\'s turn to present.',
  smart_to_ascii: "“Smart” quotes, an en–dash, an em—dash, and an ellipsis… all folded.",
  url_email_mask: "Email me at alice@example.com or see https://example.com/docs for details.",
  // formatting
  strip_markdown: "# Report\n\nPlease review **carefully** and note the `config` value:\n\n- first item\n- second item",
  bullet_reformat: "Steps to follow:\n1. gather the inputs\n2. run the pipeline\n3. review the output",
  markdown_wrap: "def greet(name):\n    return f\"hello {name}\"",
  line_wrap: "This is a fairly long single line of text that should visibly wrap once a fixed column width is applied to it.",
  tabs_for_spaces: "name    role    team\nalice   lead    infra",
  extra_whitespace: "The quick brown fox jumps over the lazy dog.",
  // schema
  reorder_enum: SCHEMA_ENUM_TEXT,
  reorder_union: SCHEMA_UNION_TEXT,
  expand_type_shorthand: SCHEMA_TYPELIST_TEXT,
  inline_defs: SCHEMA_DEFS_TEXT,
  additional_properties_false: SCHEMA_DEFS_TEXT,
  add_redundant_constraints: SCHEMA_DEFAULT_TEXT,
  reorder_properties: SCHEMA_DEFAULT_TEXT,
};

// The sample for a given family (or the general default).
function sampleForFamily(family) {
  return FAMILY_SAMPLES[family] || DEFAULT_TEXT;
}

// The best sample for a perturbation: its own tailored example, else its family's, else the default.
function sampleFor(p) {
  if (!p) return DEFAULT_TEXT;
  return PERTURBATION_SAMPLES[p.name] || sampleForFamily(p.family);
}

// All built-in sample strings, used to decide whether the box still holds a sample (vs user input).
function allSamples() {
  return [DEFAULT_TEXT, ...Object.values(FAMILY_SAMPLES), ...Object.values(PERTURBATION_SAMPLES)];
}

// True when the textarea still holds one of the built-in samples (safe to swap), not user input.
function textIsSample() {
  const v = $("texts").value.trim();
  if (v === "") return true;
  return allSamples().some((s) => s.trim() === v);
}

// Full reset: form back to defaults, history cleared, saved state and URL query wiped.
function resetAll() {
  HISTORY = [];
  activeRun = null;
  try { localStorage.removeItem(STORE_KEY); } catch (e) { /* storage disabled */ }
  // Reset to the default perturbation (native/realistic_typos) when present, else the first entry.
  if (CATALOG[DEFAULT_KEY]) {
    $("perturbation").value = DEFAULT_KEY;
  } else if (flatList.length) {
    $("perturbation").value = perturbationKey(flatList[0].category, flatList[0].name);
  }
  $("seed").value = 0;
  $("autorandom").checked = true;  // auto-randomize is the default
  // Set the sample tailored to the reset perturbation, then let onSelect apply it.
  const resetP = CATALOG[$("perturbation").value];
  $("texts").value = resetP ? sampleFor(resetP) : DEFAULT_TEXT;
  onSelect(); // rebuilds param inputs at their declared defaults; keeps the sample above
  $("output").className = "out";
  $("output").textContent = "Pick a perturbation and press Perturb.";
  $("meta").textContent = "";
  lastResultTexts = null;
  $("copy").disabled = true;
  renderHistory();
  // Save the cleared state to localStorage, but do NOT sync the URL from the form (that is what
  // persist() would do), Reset should leave the address bar at the base URL with no query string.
  // The API key is a credential, not part of the perturb instance, so Reset preserves it.
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      form: currentState(), history: HISTORY,
      llmKey: $("llm-key").value, llmBaseUrl: $("llm-base-url").value,
    }));
  } catch (e) { /* storage disabled */ }
  history.replaceState(null, "", location.pathname);
}

// --- Share: encode the current perturb instance in the URL query string ---
function shareUrl() {
  const s = currentState();
  const q = new URLSearchParams();
  if (s.category) q.set("c", s.category);
  q.set("p", s.name);
  q.set("seed", s.seed);
  q.set("texts", s.texts);
  for (const [k, v] of Object.entries(s.params)) q.set("param_" + k, v);
  return `${location.origin}${location.pathname}?${q.toString()}`;
}

// Keep the address bar in sync with the current perturb instance so it can be shared by copying
// the URL directly. replaceState (not pushState) avoids flooding the back-button history as the
// form changes.
function syncUrl() {
  history.replaceState(null, "", shareUrl());
}

function stateFromUrl() {
  const q = new URLSearchParams(location.search);
  const name = q.get("p");
  if (!name) return null;
  const params = {};
  for (const [k, v] of q.entries()) {
    if (k.startsWith("param_")) params[k.slice(6)] = v;
  }
  return {
    category: q.get("c") ?? undefined,
    name,
    params,
    seed: q.has("seed") ? parseInt(q.get("seed"), 10) || 0 : 0,
    texts: q.get("texts") ?? "",
  };
}

async function share() {
  const url = shareUrl();
  const btn = $("share");
  const original = btn.textContent;
  try {
    await navigator.clipboard.writeText(url);
    btn.textContent = "Copied!";
  } catch (e) {
    // Clipboard blocked (e.g. non-secure context): fall back to a prompt.
    window.prompt("Share this link:", url);
    btn.textContent = original;
    return;
  }
  setTimeout(() => { btn.textContent = original; }, 1200);
}

// Apply a perturb instance from the current URL, if any, and run it so the shared result shows
// immediately (using the URL's seed, not a fresh random one). Returns true if a URL state applied.
function applyUrlStateAndRun() {
  const urlState = stateFromUrl();
  if (!urlState || !applyState(urlState)) return false;
  run(urlState.seed);
  return true;
}

$("perturbation").addEventListener("change", () => { onSelect(); persist(); });
$("seed").addEventListener("change", persist);
$("texts").addEventListener("input", persist);
$("params").addEventListener("input", persist);
$("run").addEventListener("click", () => run());
$("randomize").addEventListener("click", () => { $("seed").value = randomSeed(); persist(); });
$("autorandom").addEventListener("change", persist);
$("llm-key").addEventListener("input", persist);
$("llm-base-url").addEventListener("input", persist);
$("share").addEventListener("click", share);
$("clear").addEventListener("click", clearHistory);
$("reset").addEventListener("click", resetAll);
$("copy").addEventListener("click", copyResult);
$("theme-toggle").addEventListener("click", toggleTheme);

// A same-document navigation (e.g. editing the query string and pressing Enter, or Back/Forward)
// fires popstate rather than reloading the page; re-apply and run the instance from the new URL.
window.addEventListener("popstate", () => { applyUrlStateAndRun(); });

// On load: catalog first, then a shared URL takes precedence (preload + run), else restore state.
loadVersion();
loadCatalog().then(() => {
  const persisted = loadPersisted();
  if (persisted && Array.isArray(persisted.history)) {
    HISTORY = persisted.history;
    renderHistory();
  }
  // Restore the saved API key and base URL regardless of which state path applies below.
  if (persisted && typeof persisted.llmKey === "string") {
    $("llm-key").value = persisted.llmKey;
  }
  if (persisted && typeof persisted.llmBaseUrl === "string") {
    $("llm-base-url").value = persisted.llmBaseUrl;
  }
  if (applyUrlStateAndRun()) {
    return; // URL instance applied and run
  }
  if (persisted && persisted.form) {
    applyState(persisted.form);
  }
  syncUrl(); // reflect the loaded instance in the address bar
}).catch((e) => {
  $("output").className = "out err";
  $("output").textContent = "Could not load catalog: " + e.message;
});
