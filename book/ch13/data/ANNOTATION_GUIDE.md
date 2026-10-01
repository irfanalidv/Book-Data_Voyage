# Annotation Guide for Chapter 13's Skill Extraction Eval Set

This guide describes how the `eval_set.jsonl` file is hand-labelled.
The labels are the ground truth for measuring three skill extraction
techniques in Chapter 13. The guide is committed alongside the
labels because the *rules* are part of the measurement — different
annotators using different rules would produce different "ground
truth," and the chapter is honest about that.

## What we're labelling

For each posting in `eval_set.jsonl`, the `skills_verified` field
is a list of canonical skill names (from `CANONICAL_SKILLS` in
`talentlens/skills.py`) that the posting mentions. Mentions can be:

- **Direct**: the description or `skills_raw` field contains the
  canonical name literally ("Python", "PostgreSQL").
- **Alias**: the posting mentions a known alias of a canonical
  skill ("k8s" for Kubernetes, "TF" for TensorFlow, "Postgres"
  for PostgreSQL). The labelled value is the **canonical name**,
  not the alias.

Each row's `skills_verified` is a JSON list, e.g.:

    "skills_verified": ["Python", "PostgreSQL", "Kubernetes"]

Order doesn't matter.

## Rules

These rules are committed because consistency matters more than
any individual rule. Pick a rule, stick to it.

1. **Canonical names only.** If the posting mentions "k8s", the
   label is "Kubernetes". Aliases never appear in the labels.

2. **Description + skills_raw together.** A skill counts if it
   appears in *either* field. The eval set's purpose is to measure
   extractors that work from both.

3. **Required vs nice-to-have: both count.** "Python required" and
   "Python a plus" both label as Python. The extractor doesn't
   distinguish; the eval set shouldn't either.

4. **Negation does NOT count.** "No experience with PyTorch needed"
   does not label as PyTorch. Rare in practice, but be deliberate
   when it happens.

5. **Skill families that aren't in CANONICAL_SKILLS are skipped.**
   If the posting mentions "Snowflake" but Snowflake isn't in our
   canonical vocabulary, we don't label it. The vocabulary is a
   fixed boundary; expanding it is a separate effort.

6. **Ambiguous mentions (judgement call).** When the posting uses
   a generic term like "data pipelines" — does that imply Airflow?
   Spark? No specific tool? **Default: don't label.** The eval
   set is conservative. False negatives in the labels hurt all
   three extractors equally; false positives unfairly help
   extractors that fabricate.

7. **The `notes` field is for "I made a judgement call here."**
   Use it. Reading 30 postings carefully is the chapter; recording
   what was hard is what makes the measurement honest.

## Process

For each row:

1. Read the title.
2. Read the description (truncated to 2000 chars in the eval file).
3. Read the `skills_raw` field.
4. Build the `skills_verified` list, mentally applying the rules
   above.
5. If anything was ambiguous, note it in the `notes` field.
6. Move to the next row.

A pass through 30 postings takes ~2 hours at a careful pace. Don't
rush. The labels are committed to the repo and become the
chapter's measurement bedrock — they'll be re-read by every reader
who tries to reproduce the results.

## After labelling

Once all 30 rows have `skills_verified` filled in:

    PYTHONPATH=. python -c "
    import json
    rows = [json.loads(l) for l in open('book/ch13/data/eval_set.jsonl')]
    unlabelled = [r for r in rows if r.get('skills_verified') is None]
    print(f'Total: {len(rows)}, unlabelled: {len(unlabelled)}')
    labelled_skills = [s for r in rows if r.get('skills_verified') for s in r['skills_verified']]
    print(f'Total skill labels: {len(labelled_skills)}')
    from collections import Counter
    print('Top 10 skills:', Counter(labelled_skills).most_common(10))
    "

Run this after labelling to confirm completeness. Then iteration 2
of Chapter 13 can begin.

## Inter-annotator agreement

With one annotator (the book author), inter-annotator agreement is
undefined. The chapter prose acknowledges this honestly: labels
represent one careful pass by one person, not a consensus across
annotators. A reader who disagrees with a label is welcome to
open an issue.
