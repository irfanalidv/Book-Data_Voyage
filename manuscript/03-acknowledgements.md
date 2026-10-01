# Acknowledgements

This book exists because of the open-source ecosystem that made the work in it possible.

## Open-source tools this book depends on

Every chapter uses libraries built and maintained by engineers who publish their work freely. The core ones are pandas, NumPy, scikit-learn, PyTorch, FastAPI, sentence-transformers, Pydantic, joblib, and uvicorn. The authors of these tools, among them Wes McKinney, the scikit-learn contributors, Sebastián Ramírez and Nils Reimers, built the infrastructure that the rest of us stand on.

The RAG chapter uses pgvector and the SQLite ecosystem. The CI/CD chapter runs on GitHub Actions. The deployment chapter uses Render. None of this costs anything to start. That accessibility is not accidental. It is the result of deliberate decisions by people who believed the tools should be free to use.

## The communities that made this practical

The Adzuna API (Jobs by Adzuna) and RemoteOK's public API, and the datasets that ship with scikit-learn, gave the book real data to work with. The Hugging Face model hub provided pretrained models that would have taken months to train from scratch. Stack Overflow answered the questions that every engineer hits at 1 AM when something breaks in a way no documentation anticipated. The India and Nepal AI/ML community, particularly the practitioners building real systems in Bangalore, Hyderabad, Pune, and Kathmandu, shaped what this book considers important.

## The people who shaped this directly

The clients of DataCortex IQ, particularly Sumit Pokhrel, whose real production requirements made the deployment chapters honest rather than hypothetical. CA Sadiq Shariff, for handling the compliance and corporate work that lets DataCortex exist as a real business and not a side project. The IISER Tirupati faculty who taught the foundations that made the harder material approachable.

Krishna Prasad Chitrapura, whose introduction opened doors I couldn't have opened on my own, and whose long view of the AI field shaped how I think about which problems are worth working on. Vijay Mulani, for the kind of warm-network conversations that turned this book from "something I might write someday" into something I actually finished.

The engineers in Nepal and India who reached out after seeing the open-source libraries and asked questions that revealed what the field needs from a book like this. The Robotics Association of Nepal, AIAN, and the team at MOCIT, for showing me what AI infrastructure looks like outside the usual centres of attention.

My mother, whose belief in this work made everything else possible.

## On sources and research

The salary benchmarks in Chapter 24 are my read of the early-2026 Indian AI/ML market, informed by seven years of offers, contracts, and hiring conversations, cross-checked against public compensation data. TalentLens is the method readers can use to verify against live postings: the chapter script measures remote-vs-on-site medians on your corpus; the five-market and negotiation bands are reference material until you have enough disclosed salaries to compute your own. Those figures will shift. Treat them as a starting point for your own research, not as fixed facts.

The technical content reflects production patterns from real systems: Reflecta (a voice-first AI wellness platform), Godam (an FMCG inventory platform serving Nepal), RAGNav (open-source hybrid retrieval library), and various client projects under DataCortex IQ. When something is described as working in production, it has been running in production.

The open-source libraries cited throughout (RAGNav, ragfallback, AgentEnsemble, AgentCare, scrapeflow-py, and others) are all available on PyPI under my GitHub account. Use them, fork them, send patches.

---

*Irfan Ali*
*Siliguri, 2026*
