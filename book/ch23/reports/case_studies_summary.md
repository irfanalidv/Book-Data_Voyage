# Real-World Case Studies — Three Production AI Systems

Three systems I built and shipped. What worked, what broke, and what you can borrow.

## Reflecta — Voice-first AI wellness companion
*app.getreflecta.com*

**What it does:**
Users call a phone number and talk to an AI wellness companion. The call is transcribed, analysed by an LLM, and summarised into a personal dashboard. The LLM tracks mood patterns over time using pgvector similarity search on past conversation embeddings.

**Stack:** Next.js 15, Neon Postgres + pgvector, Groq/Llama 3.3 70B, Bolna AI (telephony), Twilio Verify, Vercel

**Key lessons:**
- Voice latency is unforgiving — anything over 800ms feels broken. Groq's inference speed (200-400ms) is what made this viable.
- Telephony adds a layer of failure modes that don't exist in web apps. Bolna handles retries and audio quality; don't build this yourself.
- The hardest part wasn't the AI — it was the security hardening: webhook signature verification, rate limiting per phone number, admin auth. Ship these before the first real user.
- pgvector for conversation history retrieval works well under 100k records. At larger scale, move the vector operations to a dedicated service.

**What broke (and why):**
- The 'Call me now' backend endpoint silently failed for 3 days because Vercel logs weren't being monitored. Add Sentry or equivalent on day 1.
- LLM hallucinations in a wellness context are a real risk — the model occasionally 'remembered' things the user never said. Groundedness checks on every response are mandatory.

---

## Godam — FMCG trade and inventory PWA for Nepal
*getgodam.com*

**What it does:**
Inventory management for a Nepal-based FMCG distribution business. Three user roles: admin, manager, field agent. Features include godown management, cheque tracking with photo uploads, daily sales reports as PDFs, and costing sheet generation. Runs as a PWA — works offline, installable on Android.

**Stack:** Next.js 15, Supabase (PostgreSQL + JWT), jsPDF, Vercel, Namecheap DNS

**Key lessons:**
- Nepali business requirements differ from Indian or Western defaults. Fiscal year follows Bikram Sambat calendar. Reports use Nepali number formatting. Build for the actual user, not an imaginary one.
- PWA offline-first is genuinely hard. IndexedDB sync with Supabase on reconnect took 3x longer to implement than estimated.
- Monthly subscription (NPR 16,000/month) sustains the product. Per-feature pricing would have created friction. One price, everything in.
- PDF generation with jsPDF is sufficient for business reports. Don't use a PDF service for something this simple.

**What broke (and why):**
- Supabase row-level security policies were misconfigured at launch. Field agents could see each other's data for 6 hours. Always test RLS with a non-admin user before go-live.
- Photo uploads to Supabase storage had file size limits we hadn't tested on real mobile networks. Added client-side compression.

---

## RAGNav — Open-source hybrid RAG retrieval library
*pypi.org/project/ragnav*

**What it does:**
A PyPI library for hybrid BM25 + dense retrieval with production monitoring. SQuAD benchmark R@3 = 0.956. Includes ConfidenceDriftMonitor (PSI/KS tests), MLflowLogger, and a GitHub Actions CI with a regression gate that fails the build if R@3 drops below threshold.

**Stack:** Python, BM25 (rank-bm25), FAISS, SentenceTransformers, MLflow, GitHub Actions

**Key lessons:**
- Open-source libraries need a README that shows results in the first three lines. Benchmarks (R@3=0.956) get stars. Descriptions of architecture do not.
- The CI regression gate (fail build if accuracy drops) was the single most valuable engineering decision. It caught 4 regressions introduced by dependency updates over 6 months.
- Hybrid retrieval (BM25 + dense) outperforms either alone by 8-12% on R@3. The exact keyword match from BM25 fills gaps the embedding model misses on technical terms.
- Publishing to PyPI is the easy part. Maintaining documentation, answering issues, and keeping dependencies updated is the actual work.

**What broke (and why):**
- FAISS on Apple Silicon required a different install path than on Linux. Lost a weekend to this. Test on multiple platforms before announcing.
- MLflowLogger added a hard dependency on mlflow that bloated the install for users who didn't need logging. Moved to optional extras.

---

## Cross-cutting lessons

These patterns showed up across all three projects:

**1. Ship monitoring before users.** Every incident above was caught late because I didn't have logging and alerting in place before the first real user. Add Sentry, structured logs, and a health endpoint before you tell anyone about the product.

**2. The AI part is rarely the hard part.** The hard parts are: auth, rate limiting, mobile edge cases, payment flows, and keeping dependencies working 6 months later. The LLM call is 10 lines.

**3. Charge from day one.** Godam's NPR 16,000/month subscription made it sustainable and gave the client skin in the game. Free tiers attract users who don't need what you built.

**4. Constraints are clarifying.** Building for Nepal (different calendar, different number formatting, unreliable network) forced better engineering decisions than building for a generic 'global' user.