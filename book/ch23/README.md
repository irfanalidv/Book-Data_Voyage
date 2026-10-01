# Chapter 23: Real-World Case Studies

> **Stepping outside TalentLens:** Three production AI systems I've built and shipped: Reflecta, Godam, and RAGNav. What worked, what broke, and what carries over, told so that "I built this once" becomes "you should do this every time."

## The problem we're solving

The previous twenty-two chapters built TalentLens: one system, deeply, end-to-end. That's how a textbook teaches. But real engineering happens across systems, and the patterns that transfer aren't the ones that show up in any single project's architecture diagram.

This chapter steps back. Three systems I've shipped to real users with real money attached: a voice AI wellness companion, an FMCG inventory PWA for the Nepal market, and an open-source Python library on PyPI. Different stacks, different problem domains, different failure modes. The lessons that surface when you compare them are the ones worth carrying into the next thing you build.

Run the chapter's measurement script to see the comparative metrics and architecture diagrams:

```bash
python book/ch23/ch23_case_studies.py
```

Outputs the system-architecture comparison figure, the lessons-matrix figure, and the summary markdown under `book/ch23/reports/`.

---

## Why case studies, and why now

TalentLens is one architecture. Production is a distribution of architectures. This chapter compares three shipped systems so patterns (observability-first, RLS before features, CI regression gates) transfer when the stack changes.

**Why not only TalentLens:** Readers need proof the book's shipping chapters generalise beyond job search. Voice latency, offline PWAs, and library CI gates are different failure surfaces.

---

## The code

```bash
python book/ch23/ch23_case_studies.py
```

---

## The methods

### Patterns that transfer across systems

Five patterns that appeared in all three systems, regardless of stack:

1. **The hard part is rarely the AI.** Across all three systems, the model was the easiest part. The hard parts were authentication, rate limiting, webhook signature verification, RLS policies, and observability: the kind of work that doesn't show up in tutorials.

2. **Observability before features.** Every system shipped a real production incident that would have been caught earlier with logs. Reflecta's "Call me now" endpoint silently failed for 3 days. Godam's RLS policies leaked data for 6 hours. RAGNav's first release had a regression no one caught because there was no benchmark gate. The pattern is the same: build the alarm, then build the feature.

3. **Production hardening is a phase, not a checklist.** Each system went through a 1-2 week "production hardening" sprint after the first usable version. Rate limits, webhook secrets, admin auth, error budgets. These aren't features you add; they're a stage of the project's life. Plan for it.

4. **Costs are easier to ignore than to fix later.** RAGNav's cost discipline was deliberate from day one. Reflecta's wasn't, and the resulting LLM bills shaped product decisions in ways that should have been engineering decisions instead.

5. **Build for the actual user, not the ideal one.** Godam was the clearest example: Nepal traders run on intermittent connectivity, low-end Android, mixed Hindi/Nepali/English, and a cash-first workflow. Building "for India" would have built the wrong product. The lesson generalises: every production system has assumed users who don't exist, and discovering this in week one is the difference between shipping and re-shipping.

## Case study 1: Reflecta (voice AI, B2C)

**What it is:** Voice-first AI wellness companion. Users call a phone number, talk to the AI, get back a transcribed dashboard with mood tracking. `app.getreflecta.com`.

**Stack:** Next.js 15, Neon Postgres + pgvector, Groq/Llama 3.3 70B, Bolna AI (telephony), Twilio Verify, Vercel.

**What worked:**
- Voice latency stayed under 800ms because Groq's inference speed (200-400ms) is what made this viable. A different LLM provider would have killed the product.
- Outsourcing telephony to Bolna instead of building on raw Twilio Voice. Bolna handles audio quality, retries, and voice activity detection, three problems we didn't have to solve.
- pgvector for conversation history retrieval scaled to ~100k records before needing thought about migration.

**What broke:**
- The "Call me now" backend endpoint silently failed for 3 days. Vercel logs existed but nobody was watching them.
- LLM hallucinations in a wellness context. The model occasionally "remembered" things the user never said. Groundedness checks became mandatory.
- Webhook signature verification was a 1-day implementation that should have been there from the first commit.

**Transferable lesson:** When latency is part of the product (voice, real-time, anything user-facing in seconds), the LLM provider choice is a product decision, not an engineering one. Pick on latency first, capability second.

## Case study 2: Godam (FMCG PWA, B2B SaaS for Nepal)

**What it is:** Inventory management for Nepal-based FMCG distributors. Three roles: admin, manager, field agent. Godown management, cheque tracking with photo uploads, daily sales reports as PDFs, costing sheet generation. Offline-first PWA. `getgodam.com`.

**Stack:** Next.js 15, Supabase (PostgreSQL + JWT + Storage), jsPDF, Vercel, Namecheap DNS.

**What worked:**
- PWA offline-first via IndexedDB + Supabase sync. Field agents in areas with intermittent 3G can record sales offline; sync happens on reconnect.
- jsPDF for business reports. Don't reach for a PDF microservice when the requirement is "print a daily report."
- Single monthly subscription (NPR 16,000/month) instead of per-feature pricing. Friction kills SaaS at this segment.
- Nepali fiscal year (Bikram Sambat) and number formatting were built in from day one. The product feels native, not localised.

**What broke:**
- Supabase RLS policies were misconfigured at launch. Field agents could see each other's data for 6 hours before a manager noticed. Always test RLS with a non-admin user before go-live.
- Photo uploads hit file size limits we hadn't tested on real mobile networks. Client-side compression became table stakes.

**Transferable lesson:** "Building for [market]" is a planning fiction. You're building for *the user in front of you*: their device, their network, their workflow, their fiscal calendar. Generic SaaS assumes a generic user who doesn't exist in regional B2B markets.

## Case study 3: RAGNav (open-source library, PyPI)

**What it is:** Production-grade hybrid retrieval (BM25 + dense) library with built-in monitoring. SQuAD R@3 = 0.956. Ships with ConfidenceDriftMonitor (PSI/KS tests), MLflowLogger, and a GitHub Actions CI gate that fails the build if R@3 drops below threshold. `pypi.org/project/ragnav`.

**Stack:** Python, sentence-transformers, BM25Okapi, MLflow, GitHub Actions.

**What worked:**
- Regression gate in CI. Every PR runs the full SQuAD benchmark and fails if R@3 drops by more than 1 point. Catches model regressions before merge.
- Confidence drift monitoring built into the library, not bolted on. PSI and KS tests run alongside retrieval, no separate tooling needed.
- Tight scope. RAGNav does hybrid retrieval; it doesn't do LLM generation, document chunking, or vector store management. Smaller surface, easier to maintain.

**What broke:**
- The first release had a regression in dense retrieval that went undetected because the CI gate was added in v0.2, not v0.1. Two-day fix; lesson was about ordering.
- Documentation was an afterthought. Adoption stalled until the README got the same care as the code.

**Transferable lesson:** Library code lives or dies by its docs and its CI. A library with perfect code and bad docs is a library no one uses. A library with great docs and a flaky CI is a library people stop trusting after the third broken release.

## Comparison: where the three systems differ

| Dimension | Reflecta | Godam | RAGNav |
|---|---|---|---|
| Domain | Consumer voice AI | B2B SaaS (regional) | Developer infrastructure |
| Stack complexity (1-10) | 7 | 5 | 4 |
| Latency budget | <800ms | seconds OK | offline |
| Failure cost | User trust | Customer data leak | Build break |
| Primary risk | Hallucination | RLS / data isolation | Regression |
| Monetisation | Pre-revenue (beta) | NPR 16k/mo subscription | Open source |
| Real users | Beta | Production | PyPI downloads |

The comparison figure (`book/ch23/reports/figures/ch23_system_architectures.png`) visualises the architectural differences. The lessons matrix (`book/ch23/reports/figures/ch23_lessons_matrix.png`) maps each lesson to which systems it applies to.

---

## Interpreting the output: what do these figures show?

![System architecture comparison: Reflecta, Godam, RAGNav](reports/figures/ch23_system_architectures.png)

**`ch23_system_architectures.png`**: Side-by-side stacks. Use it to explain *where* complexity lives (telephony vs RLS vs retrieval CI), not to memorise logos.

![Lessons matrix: which patterns apply to which system](reports/figures/ch23_lessons_matrix.png)

**`ch23_lessons_matrix.png`**: Which lessons apply to which product. If a lesson is empty for all three, it is probably TalentLens-specific, not universal.

---

## Common mistakes I made (more than once)

Four patterns that repeated across the three systems. Each one cost real time; each one was avoidable in hindsight.

**Mistake 1: shipping the feature before the alarm.** Every system had a production incident that observability would have caught earlier. Reflecta: silent endpoint failure. Godam: RLS leak. RAGNav: undetected regression. The fix is uniform: Sentry or an equivalent in the first PR, not the tenth.

**Mistake 2: assuming "MVP" excuses production hygiene.** Webhook signature verification, rate limiting, admin auth, audit logs: none of these are MVP features in the product sense, but all of them are MVP features in the security sense. Skipping them in v0.1 means rewriting them in v0.3 under customer pressure.

**Mistake 3: assuming the AI is the bottleneck.** It almost never is. The bottleneck is database design, RLS policies, auth flow, deployment pipeline. Time spent prompt-engineering while RLS is broken is time spent in the wrong place.

**Mistake 4: building for a hypothetical user.** Three different systems, three different failure modes of the same shape, assuming users have something they don't have. Bandwidth, a recent phone, fluent English, a specific workflow. Building for the actual user is a separate engineering activity from building the feature, and skipping it is not a shortcut.

## Interview questions

**Q1: You've shipped several production AI systems. What was the same across them?**

Template answer: "The model was never the hard part. Across a voice product, a B2B inventory app, and an open-source library, the problems that cost real time were observability, access control, rate limiting, and release discipline. So I now put logging and alerts in the first pull request, treat a short hardening phase as part of every plan, and watch cost from day one instead of after the first surprising bill."

**Q2: Tell me about a production incident you caused.**

Template answer: "Pick one you understand end to end and tell it in four beats: what happened, how it was found, what you changed, and what you do differently now. For example: a 'call me now' endpoint on a voice product failed silently for three days. Logs existed; nobody watched them. We found out from a user. The fix was trivial; the lasting change was alerting on error rates for every user-facing endpoint before launch, not after."

**Q3: Why outsource telephony to a vendor instead of building on raw Twilio Voice?**

Template answer: "Buy what isn't your product. Our product was the conversation, not audio transport; the vendor already handled audio quality, retries, and voice-activity detection, three problems we'd otherwise have spent months on. The trade-offs are cost per minute and dependency on their roadmap, so I kept our logic behind our own interface to make switching possible."

**Q4: What's the difference between an MVP and a v0.1 product?**

Template answer: "An MVP limits *features*; it doesn't limit *hygiene*. A v0.1 can have one workflow, but it still needs authentication, access control, rate limits, and basic observability. Those aren't features; they're what makes it safe to have users at all. Skipping them is how teams end up rewriting the system in month four under customer pressure."

**Q5: Why release a library as open source rather than build a paid service around it?**

Template answer: "It depended on where the value was. A retrieval library is infrastructure other engineers embed; open source lowers the barrier to adoption, and the credibility and consulting work it generates is worth more than a subscription at that stage. A hosted service would have meant running infrastructure before there was demand to pay for it. Open source plus paid support is a model you can move to later."

---

## What's next

The three case studies inform every chapter that came before. Specifically:

- **Chapter 18 (Agents)** uses the same observability pattern Reflecta should have had from day one: every tool call is traced, every error is captured in a trace file.
- **Chapter 19-20 (FastAPI + Docker)** show the production hardening sequence Reflecta and Godam both went through, systematised.
- **Chapter 22 (PyPI Package)** shows the library-shipping discipline that RAGNav demanded.

Each case study has a public URL or repository; the chapter's code (`book/ch23/ch23_case_studies.py`) generates comparison figures and the summary markdown from the structured case study data, so future updates flow through one source of truth.

---

## TalentLens checkpoint

- [ ] `python book/ch23/ch23_case_studies.py` regenerates both figures
- [ ] You can name one incident from the chapter that observability would have caught on day one
- [ ] You linked one lesson to a TalentLens chapter (e.g. RAGNav CI gate → Chapter 21)

**Concepts you own:**

- Observability before features: every case study shipped an incident that logs would have caught earlier
- Production hardening as a phase: rate limits, RLS, and webhook verification are not optional "later"
- Pattern transfer across stacks: the lesson generalises even when the framework does not

---

## Files in this chapter

| Path | What it is |
|---|---|
| `book/ch23/README.md` | This file |
| `book/ch23/ch23_case_studies.py` | Chapter executable; generates figures and summary |
| `book/ch23/reports/case_studies_summary.md` | Long-form prose summary of all three case studies |
| `book/ch23/reports/figures/ch23_system_architectures.png` | Architecture comparison figure |
| `book/ch23/reports/figures/ch23_lessons_matrix.png` | Lessons-by-system matrix figure |
