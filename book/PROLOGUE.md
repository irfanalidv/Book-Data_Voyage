# Prologue: What We're Building

Imagine this is Monday morning.

You're a data science student in your final year, or maybe a software engineer two years into a job you've outgrown. You open LinkedIn. The first three posts in your feed are AI Engineers at companies you've never heard of, all hiring, all paying ₹35–60 lakh, all asking for things you half-recognise: RAG, fine-tuning, vector databases, agentic workflows, FastAPI, Docker, observability.

You've done the courses. You can train a logistic regression model. You once got 94% accuracy on the Titanic dataset. None of it tells you how to get from where you are to where those job posts live.

This book is the bridge.

---

We're going to build one thing together, end to end. It's called **TalentLens**, a job market intelligence platform. It collects job postings from public APIs, cleans them, classifies them by role, extracts the skills they demand, ranks them by semantic similarity to your CV, explains each match with an LLM, and serves search as an API you can deploy with one command.

When you finish this book:

- You'll have a working product built on job-market data. Not Iris. Not Titanic. The book ships a realistic synthetic corpus so everything runs offline and reproduces exactly; switch on the live collectors and the same code analyses real postings.
- You'll have a FastAPI service in a production-shaped Docker image, with health checks, rate limiting, and a CI/CD pipeline that tests every push and deploys to Render.
- You'll have a Python package, `talentlens-core`, with a release workflow that publishes to PyPI the way real maintainers do.
- You'll have a way to read the Indian AI/ML job market: which claims you can measure, which are judgement, and how to tell a real salary premium from a statistical illusion.
- You'll know how to answer the questions the career chapter asks (What's a fair salary for a Senior ML Engineer in Bangalore? Do remote roles really pay more? Which skills are growing?), and which of them your data can answer today, and which need months of history first.

Every chapter is one step. By the end of Chapter 5 you have a data collector. By Chapter 9 you have a classifier. By Chapter 16 you have semantic search. By Chapter 19 you have an API. By Chapter 21 you have a deployment pipeline that ships your code to production every time you push to main.

---

This is not a survey course. We are not going to cover every algorithm. We are going to cover the ones you need to build TalentLens, and we are going to cover them properly, with the parameters explained in plain English, the alternatives compared, and the failure modes named explicitly.

The chapters that follow are written the way I wish someone had written this material when I was starting: assuming you're smart, telling you the truth, showing you what works in production, and pointing out the places where the textbook answer and the real answer diverge.

Let's start with Chapter 1, where we look at the field we're entering: what the five roles in modern data and AI do, what they get paid, and where TalentLens fits.
