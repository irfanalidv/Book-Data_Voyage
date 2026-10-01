# Chapter 13 evaluation set

**Source: The Adzuna API.** Job listings in this folder are **Jobs by [Adzuna](https://www.adzuna.in)**.

This folder holds the labelled evaluation set Chapter 13 uses to benchmark skill extraction:

| File | Contents |
|---|---|
| `eval_set.jsonl` | 200 job listings with verified skill labels |
| `eval_set_llm_labelled.jsonl` | The same listings with the intermediate LLM labels |
| `ANNOTATION_GUIDE.md` | How the labels were assigned |

## Where the listings come from

The listings were retrieved through the Adzuna API under Adzuna's API terms, for personal research into skill extraction. Each record keeps only what the benchmark needs: the job title and an excerpt of at most 500 characters of the description. Recruiter names, email addresses, and other personal contact details have been removed.

Every record carries its attribution:

- `source`: `"Jobs by Adzuna"`
- `source_url`: a link to the original listing on Adzuna. Listings expire, so older links may no longer open.

## Rights and permitted use

The listing text belongs to Adzuna and the original advertisers. It is **not** covered by this repository's MIT License or by the book's text license. It is included only so that readers can reproduce Chapter 13's evaluation. Please do not reuse it for any other purpose. To work with job data yourself, register for your own free key at [developer.adzuna.com](https://developer.adzuna.com) and run `make collect-dataset`.

The skill labels (`skills_verified`) and the annotation guide are the author's work and are released under the MIT License.

## Removal

If you are Adzuna or one of the advertisers and would like a listing, or the whole set, removed, email irfan@datacortex.in and it will be taken down promptly.
