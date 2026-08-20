# SecGPT ↔ DarkAI — Sibling Relationship, Overlap & Coordination

> **Purpose:** define what each project owns, where they overlap, and how to
> coordinate them without duplicating work. DarkAI is a sibling lab in the
> CADRE-Platform ecosystem, maintained by the same author.
>
> **Status:** DarkAI lab stack shipped + operator-validated; curriculum
> (229 exercises) not yet executed. SecGPT Phase 2 adversarial evaluation in
> progress.
>
> **Agent instruction:** if you are an AI agent working in this repo, read
> [AGENTS.md](../AGENTS.md) and the rules section at the bottom of this file.

---

## 1. The one-line difference

| | SecGPT | DarkAI |
|---|---|---|
| **Attacks the** | model **brain** (weights, training, inference internals) | model **body** (deployed AI applications and their ecosystem) |
| **Question answered** | *How do you build an LLM and what breaks inside it?* | *How do you attack and defend AI applications in production?* |
| **Targets** | Our own trained models (SecGPTv2.5 / v3 / Prod) | 12 containerized vulnerable AI apps (chatbot, RAG, MCP, agents, sinks, ML services) |
| **Method** | PyTorch tensors, log-probs, retraining | HTTP endpoints, prompts, audit logs, Docker |
| **Has a blue team?** | No (eval rigor only) | Yes — audit schema, 15 detection rules, Loki, forensics UI, Praxis fix loop |
| **Visibility** | Public (on GitHub) | Private (intended for eventual public release) |

**DarkAI, in one paragraph:** a Docker Compose AI red-team & forensics lab
(`172.20.0.0/24` bridge, 12 services) with a 14-phase curriculum (P0–P13,
229 exercises) mapped to MITRE ATLAS / OWASP LLM Top 10 2025 / Google SAIF /
NIST AI 600-1. Every app emits JSONL audit events (schema v2) consumed by a
detection-rule engine, Loki, and a forensic investigation UI (darkai-triage).
Praxis — a sibling SAST/fix/verify CLI with 28 agents — is baked into the ops
container for the defensive loop. Located at
`C:\STUDY\Github\CADRE-Platform\DarkAI` (from this repo: `..\CADRE-Platform\DarkAI`).

---

## 2. Overlap matrix — same techniques, different surfaces

SecGPT Phase 2 (see [LLM_SECURITY_DEMO.md](LLM_SECURITY_DEMO.md)) and DarkAI
exercise the **same public attack catalog** (OWASP LLM/ML Top 10, ATLAS).
The difference is always the attack surface. Do not treat these as duplicates.

| Technique | SecGPT Phase 2 | DarkAI exercise | Same? |
|---|---|---|---|
| Prompt injection | A1 `attacks/prompt_injection.py` — override rate across 3 models | P2 (EX-P2-001..038): live chatbot, payloads, garak probing | technique only |
| Membership inference | A4 — log-prob gap on **weights** | P5 — `membership.py` cosine distance over **RAG HTTP** | name only |
| Data extraction | A3 — verbatim training-data recall | P4/P5 — RAG exfil, `invert.py` embedding inversion | technique only |
| Data poisoning | B1 — backdoor pairs + retrain | P4 — `poison_demo.py` label-flip/Trojan (94% ASR) | concept only |
| Model theft | Tier 3 — distillation | P8 — `model_steal.py` black-box surrogate | technique only |
| DoS / sponge | Tier 3 — resource exhaustion | P8 — `sponge.py` latency inflation | technique only |
| Classification perturbation | A5 — SMS/KDD typo flips | P8 — `goodwords_evasion.py` | technique only |

**Canonical example** — the two `membership.py` implementations:

```
SecGPT  : read model log-probs of own weights on leaked vs held-out eval items (AUROC)
DarkAI  : HTTP POST /search to darkai-rag, measure cosine distance of top hit (threshold)
```

Same MITRE ATLAS technique (AML.T0022), zero shared code, zero shared
conclusions. That is the intended relationship.

---

## 3. Differentiation contract (what each repo owns)

1. **SecGPT owns the weights.** Anything that requires reading tensors,
   log-probs, gradients, or retraining belongs here (attacks A1–A5, B1–B2,
   distillation, benchmark harness, data-quality iterations).
2. **DarkAI owns the applications.** Anything that requires a served,
   networked AI app (prompts over HTTP, RAG ingestion, MCP tool calls, agent
   delegation, audit logs, detection rules, forensics) belongs there.
3. **No attack-script duplication.** If an attack needs an HTTP target, write
   it against DarkAI or reference DarkAI's tool — do not re-implement it
   inside SecGPT. If it needs model internals, keep it here.
4. **Boundary (DarkAI rule 6):** DarkAI exercise content, payloads, and lab
   specifics must NOT be published through SecGPT. Cross-references and
   names are fine; copying exercise material is not.
5. **No secrets across repos.** API keys, AD credentials, `.env` content, and
   DarkAI's internal docs never enter SecGPT.

---

## 4. Shared assets (use both, don't duplicate)

DarkAI ships standalone tools that SecGPT can legitimately reuse as *external
attack evidence* — most run without the Docker stack:

| Tool | DarkAI path | Reusable by SecGPT for |
|---|---|---|
| `poison_demo.py` | `tools/data-attacks/` | Tier 2 poisoning demonstration (standalone, synthetic data) |
| `tensor_stego.py` | `tools/data-attacks/` | Steganography demo in model artifacts |
| `model_steal.py` | `tools/infra-attacks/` | Tier 3 distillation demo (needs an HTTP target) |
| `sponge.py` | `tools/infra-attacks/` | Tier 3 resource-exhaustion demo |
| `goodwords_evasion.py` | `tools/evasion/goodwords/` | A5 classification perturbation (standalone) |
| garak config | `tools/garak/darkai-chatbot.yml` | External probe harness (see §5) |
| darkai-triage UI | `tools/darkai-triage/` | Forensic review of agent session logs (Claude/Codex JSONL) |

---

## 5. The one intended integration point: garak ↔ SecGPT-Prod

DarkAI's garak setup (EX-P2-037) is the bridge where the two projects should
actually meet:

```
SecGPT-Prod served via an OpenAI-compatible endpoint (vLLM / FastChat / TGI)
        ↓  OpenAI /v1/chat/completions
garak (DarkAI tooling) probes with DAN jailbreaks + prompt-injection families
        ↓
objective probe results on a REAL trained model, as an independent benchmark
```

This gives SecGPT an adversarial evaluation it does not currently have
(independent of the 291-prompt harness) and gives DarkAI a second, real-world
probe target beyond the stub backend. Implement only when both projects have
headroom — it is optional, not required.

---

## 6. Coordination recommendations (how to sequence)

1. **Close SecGPT Phase 2 first** (it is public and in flight — land the
   untracked `SecGPT-Prod/src/attacks/` + `Docs/LLM_SECURITY_DEMO.md`).
2. **Reuse DarkAI's standalone tools as Phase 2 evidence** where the overlap
   matrix allows (`poison_demo.py`, `goodwords_evasion.py`, `tensor_stego.py`
   run with zero lab stack). This exercises DarkAI tooling for free.
3. **Then execute DarkAI's core curriculum** (P0–P2: bootstrap, recon, prompt
   injection + audit log + triage) — this is the lab-specific proof of work
   DarkAI still lacks.
4. **Only then** consider the garak ↔ SecGPT-Prod integration (optional).
5. Keep both READMEs cross-linked once DarkAI goes public; until then, keep
   references to DarkAI out of the public README (private repo — broken links).

---

## 7. Rules for AI agents working in SecGPT (read this)

1. **DarkAI exists** at `..\CADRE-Platform\DarkAI` — a sibling AI red-team lab
   by the same author. Never assume SecGPT must cover "application-layer LLM
   attacks" alone; check DarkAI's exercises first.
2. **If a task is an HTTP/app-layer attack** (prompt injection against a
   served app, RAG poisoning, MCP, agents, XSS/SQLi sinks) — it belongs in
   DarkAI, not here. Reference it; do not re-implement.
3. **If a task needs model internals** (weights, log-probs, retraining,
   hallucination of *our own* models) — it belongs here, in SecGPT.
4. **Never copy DarkAI payloads, exercise text, or lab artifacts into this
   repo.** Names and cross-references are fine; material is not.
5. **Never write DarkAI secrets** (keys, credentials, internal docs content)
   into this repo.
6. **Shared tools** in DarkAI (`tools/data-attacks/`, `tools/infra-attacks/`,
   `tools/evasion/goodwords/`, garak config) may be *referenced and invoked*
   as external evidence, but prefer keeping SecGPT's Phase 2 attack scripts
   (`SecGPT-Prod/src/attacks/`) self-contained and model-internal.
7. If in doubt about ownership, default to: **SecGPT = model-internal,
   DarkAI = application.**
