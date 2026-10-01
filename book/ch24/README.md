# Chapter 24: The India Playbook — Getting Hired, Paid Well, and Building a Global Career from India

> **TalentLens milestone:** We built the platform. Now we use its data to understand the actual market we're entering: what skills are hiring, what remote roles pay, how to position yourself, and how to get paid in USD while sitting in Bangalore or Siliguri.

---

## The problem we're solving

Nobody tells you this part.

The textbooks stop at "deploy your model." The tutorials end at "now you know machine learning." And then you're sitting there with a GitHub repo, a certificate from some online course, and no idea how to get hired, let alone how to get hired at ₹30L+ or how to land a remote contract with a US company that pays in dollars.

This chapter is the one I wish existed when I was starting. It is operational rather than motivational. By the end you'll know exactly what a competitive profile looks like, how to structure your job search, what Indian companies vs product companies vs remote contracts pay and require, how to negotiate, how to get paid internationally without getting crushed by tax confusion, and how to think about your career in 5-year arcs rather than the next offer.

We use TalentLens where the chapter script actually measures the bundled corpus, and we separate that from the market taxonomy sections, which are my read of early-2026 India informed by seven years of offers, contracts, and hiring conversations, cross-checked against public compensation data.

---

## Why the India playbook, and why now

You have the technical stack (Chapters 5–22). This chapter converts it into **offers**: how Indian markets differ, what profiles get replies, and how remote USD contracts work.

**Why not a generic career blog:** Third-party salary surveys average away the five distinct Indian markets below. TalentLens data is the same corpus you built; the claims are checkable.

---

## The code

```bash
python book/ch24/ch24_career_market_analysis.py
```

Writes `book/ch24/reports/career_intelligence_report.md` and figures under `book/ch24/reports/figures/`.

---

## The methods

### What the chapter script measures: and what it does not

`ch24_career_market_analysis.py` loads postings from `jobs_clean_path()` (the same cleaned CSV Chapters 5–6 produce) and runs four analysis paths. Three write figures under `book/ch24/reports/figures/`; all paths feed `book/ch24/reports/career_intelligence_report.md`.

**Measured from your corpus (when salary and `is_remote` are populated):** `analyse_remote_premium()` drops rows with missing `salary_annual_inr`, splits on `is_remote`, and computes remote vs on-site medians, counts, and `premium_pct`. That is the only headline salary comparison the script derives from the dataframe itself. It lands in the log, in `career_intelligence_report.md`, and in `ch24_remote_premium.png`.

**Author-configured reference tables (not dataframe groupbys):** `MARKET_SEGMENTS`, `SKILL_SALARY_PREMIUM`, and `CAREER_TRAJECTORY` are dictionaries in the script: my market read for teaching, not medians computed from the bundled 576-row synthetic corpus. They drive `analyse_market_segments()`, `analyse_skill_salary_premium()`, `plot_career_trajectory()`, and the five-market figure `ch24_salary_by_market_segment.png`. Treat those charts as labelled reference bands, not as output of `df.groupby(...)`.

**Personalised report:** `write_career_intelligence_report()` mixes measured remote premium with `Config` fields (`target_role`, `target_city`, `years_experience`) to produce a 90-day action plan. Re-run after you publish a larger dataset and compare your medians to mine.

---

## What the data says about the Indian AI/DS job market

Before strategy, let's name the five markets we actually compete in. The median bands below are **my read of the early-2026 Indian market** (seven years of offers, contracts, and hiring conversations, cross-checked against public compensation data), **not** output of the bundled corpus (576 synthetic rows, with no company-type column to group by).

> **On these numbers:** once you have collected live postings (Chapter 5's `--live` mode), re-run `python book/ch24/ch24_career_market_analysis.py` against your `jobs_clean.csv` and compare your measured remote premium and salary-disclosed rows to the reference bands here. The script's dataframe path only computes the remote-vs-on-site block; everything in the five-market list is author judgment until you have enough rows to group by `company_type` yourself.

### The five distinct markets

The "data science job market" in India is not one market. It's at least five, with very different dynamics:

**Market 1: Service companies (TCS, Infosys, Wipro, HCL)**
- Median salary: ₹8–12L
- What they hire for: data analyst work, dashboard maintenance, basic ML model running
- What they call it: "Data Scientist," "AI Engineer," "ML Engineer"
- Reality: 80% of the work is SQL queries and Excel. The AI title is for billing rate purposes.
- Who should take these roles: people with 0–1 year experience who need a brand name and are okay with slow growth. Exit within 2 years.

**Market 2: Indian product companies (Swiggy, Zepto, Razorpay, CRED, Meesho)**
- Median salary: ₹18–35L depending on level
- What they hire for: real ML work: recommendation systems, fraud detection, ranking, pricing models
- Reality: good engineering, fast growth, expect to ship
- Who should target these: anyone with 2+ years of real ML experience and a portfolio that demonstrates shipping

**Market 3: MNC India offices (Google, Microsoft, Amazon, Uber India, LinkedIn India)**
- Median salary: ₹30–80L depending on level
- What they hire for: same as global roles: you're doing real ML research or engineering
- Reality: the interviews are the same as the US interviews. FAANG-level bar.
- Who should target these: strong fundamentals, good at system design, can do LeetCode at medium-hard level

**Market 4: Global startups / scaleups with India presence**
- Median salary: ₹25–50L (often with equity that might actually be worth something)
- What they hire for: full-stack AI engineering: you own the problem end to end
- Reality: most ambiguous, highest upside, highest variance
- Who should target these: people who can work autonomously, don't need hand-holding, want to see their work matter

**Market 5: Remote contracts for foreign companies**
- Pay: $3,000–$12,000/month (₹25L–₹100L annualised), via Deel/Toptal/direct
- What they hire for: same bar as the company's own country, with no discount for "India pricing" at good companies
- Reality: the best financial outcome available to most Indian AI engineers right now, but requires strong independent track record
- Who should target these: 3+ years of real experience, strong GitHub, ability to communicate in async English, no need for office

**TalentLens insight (measured from your CSV only):** On the bundled corpus, `analyse_remote_premium()` prints remote **₹21.8L** (n=218) vs on-site **₹18.9L** (n=358) at the median, **+15%**. *(The five market bands above are author reference, not computed from this dataframe.)*

Before you put +15% in a negotiation, remember Chapter 8. Tested within each seniority level read from the job title, the gap shrank to about ₹1L or less at mid, senior, and lead levels, none significant; the one "significant" junior result rested on 13 remote postings and a role-mix difference. Remote postings in this corpus pay more because they are more often senior, not because remote work carries a premium of its own. The script now says exactly that in `career_intelligence_report.md`. On your own live data, run Chapter 8's stratified test first; quote a remote premium only if it survives.

---

## The profile that gets hired

Let me be specific. Here's what a hiring manager or recruiter at a product company or international startup is actually looking at when they review your profile.

### GitHub: the non-negotiable

Your GitHub is read before your resume at most startups. Here's what they're looking for:

**What makes a GitHub profile work:**
- 3–5 pinned repositories that are complete, documented, and runnable
- Each repo has a proper README: what it does, how to install it, an example output
- Commit history that shows you work (not 50 commits in one day then nothing for 6 months)
- Code that looks like someone who cares: type hints, docstrings, tests, not `main.py` with 500 lines and no comments

**What kills a GitHub profile:**
- Repos named `untitled`, `test1`, `ml-project-final-v3`
- READMEs that say "TODO"
- Tutorial code copied from YouTube with minor changes
- No commits for 3+ months

**The TalentLens project as your centrepiece:** By the time you finish this book, you have a real project. A deployable FastAPI service (Chapters 19–20). A packaged library ready for PyPI (Chapter 22). RAG-powered semantic search (Chapter 16). EDA that produces real insights on real data (Chapter 7). This is portfolio-worthy. Put it front and centre. Write the README like you're pitching it to a user, not explaining it to a professor.

### LinkedIn: the distribution channel

LinkedIn is how most non-FAANG AI jobs find you, not the other way around. Your profile is a distribution problem, not a credentials problem.

**The headline formula that works:**
`[What you build] | [Key tech] | [What you're open to]`

Examples of bad headlines:
- "Data Science Enthusiast": everyone is an enthusiast
- "M.Sc. Data Science, IISER Tirupati": your degree is not your value
- "Looking for opportunities": signals desperation

Examples of good headlines:
- "AI Engineer: RAG systems, FastAPI, Python | Open to remote roles"
- "Building production ML systems | LLMs, vector DBs, MLOps"
- "Shipping AI products | Ex-[Company] | Generative AI & agentic systems"

**The About section:** 3 short paragraphs. What you build, what you've shipped (be specific: "a voice-first AI wellness app used by X users" not "built various AI projects"), what you're looking for. Write it in first person. Do not use "passionate" or "enthusiastic" or "dynamic."

**Activity matters:** Post once a week minimum. Not motivational quotes. Share something you learned, something you built, a mistake you made and how you fixed it. This compounds over 6 months into inbound recruiters and connection requests from people worth knowing.

### The resume: shorter than you think

Two pages maximum. One page preferred for under 5 years experience.

**The structure that works:**
1. Name + contact (email, GitHub URL, LinkedIn URL, city)
2. 3-line summary: who you are, what you build, what you're looking for
3. Experience: reverse chronological, bullet points that say `[Action] [What] [Result]`
4. Projects (separate section if projects are stronger than experience)
5. Skills (one line: Python, PyTorch, FastAPI, PostgreSQL, pgvector, Docker, GCP)
6. Education

**Bullets that work:** "Built a RAG pipeline over the TalentLens job-postings corpus (sample dataset shipped with the book; scale numbers depend on your own collection run) using pgvector and FastAPI, reducing search latency from 2.1s to 340ms". It is specific, has a result, uses real tech names. This is exactly the kind of bullet TalentLens earns you: Chapter 16 gives you the RAG layer, Chapter 19 gives you the API, and you have the latency numbers from your own profiling runs.

**Bullets that don't work:** "Worked on machine learning projects using various tools and technologies to deliver business value."

---

## The interview map: what each type actually asks

### Service company interviews (TCS, Infosys, Wipro)

Mostly HR + aptitude + basic Python. If you've completed Chapter 9 of this book, you're overqualified. The challenge is passing the HR round, not the technical round; they filter for culture fit and stability.

### Indian product company interviews (Swiggy/Zepto/CRED tier)
- Round 1: Take-home assignment (2–4 hours): clean a messy dataset, build a model, present findings
- Round 2: Technical screen: your take-home + system design basics
- Round 3: Bar raiser / culture fit

What they actually test: can you frame a problem correctly, can you make defensible choices, can you communicate clearly. They're not testing whether you can solve LeetCode hard. They're testing whether you think like an engineer.

### MNC (Google/Microsoft/Amazon India) interviews
- 4–6 rounds: coding (LeetCode medium-hard), ML fundamentals, ML system design, behavioural
- Coding: trees, graphs, dynamic programming. This is real; you need to practice LeetCode
- ML fundamentals: explain gradient descent, bias-variance tradeoff, how would you approach [problem]
- ML system design: "design a recommendation system for YouTube": 45 minutes, whiteboard

### Remote / international startup interviews
- Usually 3 rounds: async take-home → 60 min technical call → founder/team call
- Take-home is the most important: this is where people self-select out
- Technical call: walk through your take-home, live coding (usually simpler than FAANG), architecture discussion
- The deal-maker: can you communicate clearly in English over video? Can you run a meeting? Can you ask good questions?

**The key difference between Indian company and international startup interviews:** Indian companies test what you know. International startups test how you think. One rewards preparation, the other rewards genuine curiosity and problem-framing ability.

---

## Negotiation: what most people get wrong

The single most financially impactful skill in your career is negotiation, and it is almost never taught. Here's the compressed version.

### The rules

**Rule 1: Never give a number first.**
When asked "what's your expected salary?" the correct answer is "I'm more focused on finding the right role. Can you share the band for this position?" Most companies will share it. If they don't, say "I'm looking for a compensation that's competitive for a [role] at this level in [city/remote]. What does that look like here?" You still haven't given a number.

**Rule 2: The first offer is never the final offer.**
Every company builds room to negotiate into their first offer. If you accept immediately, they're relieved, not impressed. The response to a first offer: "Thank you, I'm excited about this role. Can I have 48 hours to review?" Then come back with a counter.

**Rule 3: Counter with a specific number and a reason.**
"I was hoping for ₹28L" is weak. "Based on my research on comparable roles in Bangalore and the scope of this position, I was targeting ₹28L. Is there flexibility there?" is stronger. The reason doesn't need to be complex. It just needs to exist.

**Rule 4: Negotiate everything, not just base.**
Variable pay, joining bonus, remote flexibility, stock/ESOPs, learning budget, hardware allowance, WFH days. Each of these has different budget pools. Base might be fixed; joining bonus might not be.

**Rule 5: If you have another offer, use it.**
"I have an offer at ₹X. Is there any way to get closer to that?" is the strongest negotiating position. You don't need to lie. You do need to actually be talking to multiple companies at once.

### Realistic salary benchmarks by experience (Bangalore, 2026)

> **📑 Reference: Salary bands by role and experience (author's market read, early 2026)**

Illustrative bands from my hiring and negotiation experience, not medians computed from the bundled TalentLens sample. Use them for framing counters; verify against your own `career_intelligence_report.md` once you have enough disclosed salaries in your corpus.

| Role | 0–2 years | 2–5 years | 5–8 years | 8+ years |
|------|-----------|-----------|-----------|----------|
| Data Analyst | ₹5–10L | ₹10–18L | ₹18–28L | ₹25–40L |
| Data Scientist | ₹8–14L | ₹14–25L | ₹25–40L | ₹35–60L |
| ML Engineer | ₹10–18L | ₹18–32L | ₹30–50L | ₹45–80L |
| AI Engineer | ₹12–22L | ₹22–38L | ₹35–60L | ₹50–90L |
| Remote (USD contracts) | $2–4k/mo | $4–7k/mo | $6–10k/mo | $10–18k/mo |

Actual offers vary widely by company type: service companies sit at the bottom of each band, MNC India offices at the top. When you have live TalentLens data, the script's remote-vs-on-site block is the only table row this chapter derives from your CSV; these role bands stay reference material until you have thousands of disclosed salaries per role.

---

## Interpreting the output: what do the career figures show?

After running `ch24_career_market_analysis.py`, open `book/ch24/reports/figures/`:

- **`ch24_salary_by_market_segment.png`**: Which of the five India markets pays at the median and how wide the P25–P75 band is within each segment. Answers: "where should I aim my applications?" Uses the script's `MARKET_SEGMENTS` reference table, not a dataframe groupby.
- **`ch24_remote_premium.png`**: Do remote-disclosed salaries beat on-site on *your* corpus right now? Answers: "is remote worth prioritising in my search?" This one is measured, so read the medians the script printed, not a hard-coded multiplier.
- **`ch24_skill_salary_correlation.png`**: Which skills carry the largest stated premium in 2026 (RAG, LLMs, MLOps vs table-stakes Python/SQL). Answers: "what do I learn next for ROI?" Reference premiums from the script config; association, not causation, the same caution as Chapter 7 EDA.
- **`ch24_career_trajectory.png`**: How three career paths (service exit, product from day one, remote contract at year three) compound over five years. Answers: "should I optimise for the next offer or the next arc?" Author projections, not fitted from posting history.

---

## Getting a remote contract: the actual steps

This is the path most worth pursuing if you have 3+ years of experience, because the financial outcome is 2–3x better than on-site Indian roles.

### Step 1: Build the signal that remote hiring managers look for

Remote companies can't see you work. They hire based on evidence that you can work autonomously, communicate clearly, and ship without hand-holding. That evidence is:
- A public GitHub with completed projects: the TalentLens repo from this book is one
- Writing: LinkedIn posts, a technical blog, library documentation. Chapter 22's README for `talentlens-core` is a working sample of the kind of documentation remote teams expect
- Async communication samples: how you write issue comments, PR descriptions, README text

### Step 2: Find the opportunities

**Platforms:**
- Toptal (hard to get in, but once in, consistently good rates)
- Deel (direct contract infrastructure: you find the company, Deel handles compliance)
- Contra (freelance for startups, lower barrier than Toptal)
- LinkedIn filtered for "Remote" + "India" or just "Remote"
- AngelList / Wellfound for funded startups
- X (Twitter): DM founders directly. "I built X, I can help you with Y". Direct, specific, not a mass application

**The cold outreach formula that works:**

Subject: `[Specific skill] → [Specific problem you can solve for them]`

Body (3 sentences maximum):
1. What you built that's relevant to them (link)
2. The specific problem you think they have that you can solve
3. Ask for 20 minutes

Do not send your resume unsolicited. Do not explain your entire background. One thing, one link, one ask.

### Step 3: Structure the contract correctly

When you land a remote contract, you need two things set up correctly. (What follows is how I and people I know have done it. It is not tax or legal advice. Indian tax and GST rules change often; confirm the current thresholds with a chartered accountant before you invoice.)

**Payment structure:** Get paid via Deel, Wise, or Payoneer into an Indian account. Do not accept payment via PayPal (fees are punishing). The standard for Indian contractors is: invoice in USD, receive in USD, convert via Wise at market rate, land in your Indian account. You'll pay income tax on the INR equivalent at your slab rate. Keep invoices. Get a CA.

**Company structure:** If you're doing contract work regularly, register an OPC (One Person Company) or LLP. Invoice from the entity, not personally. This opens up legitimate expense deductions (hardware, software, internet, home office), lets you claim GST input credits, and makes you look more professional to international clients. The GST piece: if your annual revenue from foreign clients exceeds ₹20L, you need GST registration and should file a LUT (Letter of Undertaking) to export services at 0% GST rather than charging 18% to your foreign clients.

---

## The 5-year career arc: how to think about this

Most people optimise for the next job. The people who build strong careers optimise for the next 5 years.

**Years 0–2: Build depth and a track record**
Pick one area and get good at it. Not "familiar with." Production-quality. Ship things that users or clients use. Build your GitHub. Start writing. The salary at this stage matters less than the quality of what you're building.

**Years 2–5: Build breadth and leverage**
Take on work that stretches you. If you're an ML engineer, learn deployment. If you're a data engineer, learn ML. Start speaking at meetups, conferences, online. Start contributing to open source or building your own libraries. This is when you go from "can execute tasks" to "can own a problem."

**Years 5–8: Build reputation and options**
By this point your reputation should be doing work for you: inbound referrals, interesting opportunities finding you. This is when equity starts to matter. This is when you consider whether you want to stay IC, move to technical leadership, or start something. None of these are wrong, but decide intentionally.

**The contractor vs employee question:** At 3–5 years of experience, seriously consider a 12–18 month stint as an independent contractor for 1–2 international clients. The financial gain is real (often 2–3x), the forced autonomy makes you better at your craft, and having "ran my own consulting practice" on your resume is an asset, not a liability, when you return to employment.

---

## Common mistakes I've seen (and made)

**Optimising for the wrong market tier.** Applying only to service-company "Data Scientist" posts with 5,000 applicants when product startups with 30 applicants need the same GitHub. Different funnel, different odds.

**Treating remote premium as guaranteed.** Disclosed remote medians are often higher than on-site, but that mixes seniority, company type, and who bothers to post salary. On the bundled corpus a +15% headline gap disappears once seniority is held constant (Chapter 8). Use a measured premium for negotiation framing only after it survives that test.

**Resume without evidence.** Claims without pinned repos, deploy URLs, or installable packages. The book gives you the material for all three by Chapter 22; lead with them.

**Negotiating only base salary.** Joining bonus, ESOP, remote days, and learning budget often move when base is fixed.

---

## Interview questions

**Q1: I've been applying for 3 months with no responses. What am I doing wrong?**

Template answer: "Three most common reasons: (1) resume not passing ATS: use job description keywords, simple formatting, no tables or graphics. (2) Applying to the wrong tier: service company roles have thousands of applicants; product company roles at Series B startups might have 30. Target the latter with direct outreach, not just apply buttons. (3) Profile doesn't have enough proof: a resume without a GitHub or portfolio is asking someone to trust claims with no evidence. Fix the GitHub first, then reapply."

**Q2: How do I negotiate salary when I have no competing offer?**

Template answer: "You don't need a competing offer to negotiate. You need a target number and a reason. Research the market with TalentLens data, Glassdoor, LinkedIn Salary Insights, Level.fyi for MNCs. Find the P75 for your role, city, and experience. Counter 15–20% above the first offer. The reason can be: 'Based on my research on comparable roles at similar companies, I was targeting X. Is there flexibility?' Most hiring managers expect negotiation. The ones who rescind offers for negotiating politely were never a good fit anyway."

**Q3: Should I take the service company offer or wait for a product company?**

Template answer: "It depends on your financial situation and timeline. If you need income now, take it, but set a hard exit plan of 18 months and start interviewing at 12 months. Don't let the stability become a trap. If you can afford to wait 3–6 months, hold out for a product company or startup where you'll do real work. The first 2 years of your career compound into the next 10. The quality of what you build matters more than the brand name on your resume at this stage."

**Q4: How do I get a remote job if I don't have prior remote experience?**

Template answer: "Remote experience is a chicken-and-egg problem: everyone wants it but few have it. The workaround: create the evidence that predicts remote success without having the title. Contribute to open source (async by nature). Write technical content publicly (demonstrates communication). Build a public project end-to-end (demonstrates autonomy). Then in your cover letter or outreach, name these explicitly: 'I've been contributing to [project] asynchronously for 8 months, written [N] technical articles, and built [project] independently from spec to deployment. I work well in async, distributed environments.' You're making the claim and providing the proof in the same sentence."

**Q5: Is a Master's degree worth it for an AI career in India?**

Template answer: "For research roles at top labs or MNC research divisions, yes, often required. For product/startup AI engineering roles, no, it's not a differentiator. What matters at those companies is your GitHub, your ability to ship, and whether you interview well. A Master's from a top institute (IITs, IISc, IISERs) will get your resume past the first filter at MNCs. A Master's from an unknown university probably won't. The alternative to a Master's that has the same filtering effect: a strong publication, a library with real users, a deployed product with real traffic. These are harder to fake than a degree and more signal to a technical hiring manager."

---

## What's next

You've finished the book. You have TalentLens running end to end, a deployable API, a packaged library, a GitHub that tells a coherent story, and now a framework for how to use all of it to build a career.

The last step isn't reading more. It's shipping something, applying somewhere, and having one conversation you've been putting off.

Start there.

---

## TalentLens checkpoint

At the end of this chapter, you should have:

- [ ] GitHub profile with 3–5 pinned repos, each with a real README
- [ ] LinkedIn headline updated to the formula above
- [ ] Resume trimmed to 2 pages with result-oriented bullets
- [ ] TalentLens analysis run on your target market segment (`python book/ch24/ch24_career_market_analysis.py`)
- [ ] `book/ch24/reports/career_intelligence_report.md` and figures under `book/ch24/reports/figures/`
- [ ] Tests passing from repo root: `pytest tests/test_ch24.py -v`
- [ ] One cold outreach drafted using the 3-sentence formula

**Concepts you own:**

- Reading a market through disclosed data vs anecdote: know which claims in this chapter are script-measured and which are author judgment
- Separating measured findings (remote premium on your CSV) from reference bands (five markets, negotiation table) so you do not over-fit a synthetic corpus
- Negotiation as information asymmetry: the party that names a number first usually loses; bands are framing tools, not offer predictors

Run the analysis script to get personalised market intelligence: measured remote premium plus a report template you can re-run on live data:

```bash
python book/ch24/ch24_career_market_analysis.py
```
