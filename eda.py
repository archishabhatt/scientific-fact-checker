"""
SciFact: download + exploratory analysis (replaces the original scifact_eda.py).

Sections
  1. Corpus statistics
  2. Claim / label statistics
  3. Rationale size, position, and Stage-2 class balance
  4. Qualitative examples
  5. Wordpiece length analysis (truncation check for MiniLM / SciBERT)
  6. BM25 retrieval: any-gold and all-gold Recall@k, failure analysis
  7. Text-quality check (run-together words) + BM25 on raw vs cleaned text

Setup:
    pip install pandas numpy matplotlib rank-bm25 requests tqdm transformers

Run:
    python scifact_eda.py
Outputs: printed stats, figures in ./figures/, CSVs in ./outputs/
(Section 5 needs `transformers` and downloads two small tokenizers;
 it is skipped automatically if unavailable.)
"""
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Avoid UnicodeEncodeError on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
OUT_DIR = Path("outputs")
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

KS = (1, 3, 5, 10, 20, 50)

STOPWORDS = set("""
a an the of in on at to for from by with and or but not no nor is are was were be been being
this that these those it its as than then so such into over under between among within
during after before about against can could may might will would should do does did has have had
which who whom whose what when where why how there their they them we our us you your he she his her
also more most less least very much many some any all each both other another only same
""".split())


# ----------------------------------------------------------------------
# Download + load
# ----------------------------------------------------------------------
def download_scifact():
    if (DATA_DIR / "corpus.jsonl").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    tar_path = Path("scifact_data.tar.gz")
    print("Downloading SciFact...")
    r = requests.get(DATA_URL, stream=True)
    r.raise_for_status()
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with tarfile.open(tar_path) as tar:
        tar.extractall(".")  # extracts into ./data
    print("Done.")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_data():
    corpus = {d["doc_id"]: d for d in read_jsonl(DATA_DIR / "corpus.jsonl")}
    claims = {
        "train": read_jsonl(DATA_DIR / "claims_train.jsonl"),
        "dev": read_jsonl(DATA_DIR / "claims_dev.jsonl"),
        "test": read_jsonl(DATA_DIR / "claims_test.jsonl"),  # no labels
    }
    return corpus, claims


def doc_text(d):
    return d["title"] + " " + " ".join(d["abstract"])


# ----------------------------------------------------------------------
# 1. Corpus statistics
# ----------------------------------------------------------------------
def corpus_stats(corpus):
    rows = []
    for d in corpus.values():
        sents = d["abstract"]
        rows.append({
            "doc_id": d["doc_id"],
            "n_sents": len(sents),
            "n_tokens": sum(len(s.split()) for s in sents),
            "title_len": len(d["title"].split()),
            "structured": d.get("structured", False),
        })
    df = pd.DataFrame(rows)
    print("\n=== CORPUS ===")
    print(f"Documents: {len(df)}")
    print(df[["n_sents", "n_tokens", "title_len"]].describe().round(1))
    print(f"Structured abstracts: {df['structured'].mean():.1%}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["n_sents"], bins=30, color="steelblue")
    ax[0].set(title="Sentences per abstract", xlabel="# sentences", ylabel="# abstracts")
    ax[1].hist(df["n_tokens"], bins=40, color="darkorange")
    ax[1].set(title="Tokens per abstract (whitespace)", xlabel="# tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corpus_lengths.png", dpi=150)
    plt.close()
    return df


# ----------------------------------------------------------------------
# 2. Claim / label statistics
# ----------------------------------------------------------------------
def claim_label(claim):
    """SciFact uses CONTRADICT (= REFUTE in the project plan).
    A claim with no evidence is NOT_ENOUGH_INFO."""
    ev = claim.get("evidence", {})
    if not ev:
        return "NOT_ENOUGH_INFO"
    labels = {e["label"] for doc_ev in ev.values() for e in doc_ev}
    return labels.pop() if len(labels) == 1 else "MIXED"


def claim_stats(claims):
    print("\n=== CLAIMS ===")
    rows = []
    for split in ["train", "dev"]:
        for c in claims[split]:
            ev = c.get("evidence", {})
            rationale_sents = [len(e["sentences"]) for doc_ev in ev.values() for e in doc_ev]
            rows.append({
                "split": split,
                "id": c["id"],
                "label": claim_label(c),
                "claim_len": len(c["claim"].split()),
                "n_cited_docs": len(c.get("cited_doc_ids", [])),
                "n_evidence_docs": len(ev),
                "n_rationale_sets": len(rationale_sents),
                "max_rationale_sents": max(rationale_sents) if rationale_sents else 0,
            })
    df = pd.DataFrame(rows)

    print(f"Train: {len(claims['train'])} | Dev: {len(claims['dev'])} | "
          f"Test (unlabeled): {len(claims['test'])}")
    print("\nLabel distribution (claim-level):")
    dist = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(dist)
    print((dist.div(dist.sum(axis=1), axis=0) * 100).round(1))
    print("\nClaim length (tokens):")
    print(df["claim_len"].describe().round(1))

    ev_df = df[df["label"] != "NOT_ENOUGH_INFO"]
    print("\nRationale size (sentences per rationale set), claims with evidence:")
    print(ev_df["max_rationale_sents"].value_counts().sort_index())
    print(f"Share with <=3 rationale sentences: "
          f"{(ev_df['max_rationale_sents'] <= 3).mean():.1%}")
    multi = (ev_df["n_evidence_docs"] > 1).sum()
    print(f"Claims with evidence in more than one document: {multi}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    dist.T.plot(kind="bar", ax=ax[0], rot=0)
    ax[0].set(title="Label distribution", ylabel="# claims")
    ax[1].hist(df["claim_len"], bins=25, color="seagreen")
    ax[1].set(title="Claim length", xlabel="# tokens")
    ev_df["max_rationale_sents"].value_counts().sort_index().plot(
        kind="bar", ax=ax[2], rot=0, color="slateblue")
    ax[2].set(title="Max rationale size per claim", xlabel="# sentences")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "claim_stats.png", dpi=150)
    plt.close()
    return df


# ----------------------------------------------------------------------
# 3. Rationale position + Stage-2 class balance
# ----------------------------------------------------------------------
def rationale_position(claims, corpus):
    rel_pos = []
    for c in claims["train"] + claims["dev"]:
        for doc_id, ev_list in c.get("evidence", {}).items():
            n = len(corpus[int(doc_id)]["abstract"])
            for e in ev_list:
                for s in e["sentences"]:
                    rel_pos.append(s / max(n - 1, 1))
    print("\n=== RATIONALE POSITION ===")
    print(f"Total rationale sentences: {len(rel_pos)}")
    print(f"Mean relative position: {np.mean(rel_pos):.2f} (0=start, 1=end)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel_pos, bins=20, color="firebrick")
    ax.set(title="Relative position of rationale sentences", xlabel="position in abstract")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rationale_position.png", dpi=150)
    plt.close()


def sentence_class_balance(claims, corpus):
    pos = neg = 0
    for c in claims["train"]:
        for doc_id, ev_list in c.get("evidence", {}).items():
            n = len(corpus[int(doc_id)]["abstract"])
            ev_idx = {s for e in ev_list for s in e["sentences"]}
            pos += len(ev_idx)
            neg += n - len(ev_idx)
    print("\n=== STAGE 2 CLASS BALANCE (train, gold docs only) ===")
    print(f"Evidence sentences: {pos} | Non-evidence: {neg} | "
          f"Positive rate: {pos / (pos + neg):.1%}")
    print("Note: at inference Stage 2 also sees sentences from wrong retrieved docs, "
          "so the real positive rate is lower. Use hard negatives from retrieved non-gold docs.")


# ----------------------------------------------------------------------
# 4. Qualitative examples
# ----------------------------------------------------------------------
def show_examples(corpus, claims, per_label=2):
    print("\n=== EXAMPLES ===")
    shown = Counter()
    for c in claims["train"]:
        lab = claim_label(c)
        if shown[lab] >= per_label or lab == "MIXED":
            continue
        shown[lab] += 1
        print(f"\n[{lab}] Claim {c['id']}: {c['claim']}")
        for doc_id, ev_list in c.get("evidence", {}).items():
            d = corpus[int(doc_id)]
            print(f"  Doc {doc_id}: {d['title']}")
            for e in ev_list:
                for s in e["sentences"]:
                    print(f"    - ({s}) {d['abstract'][s]}")
        if not c.get("evidence"):
            print(f"  Cited docs (no evidence found): {c.get('cited_doc_ids')}")


# ----------------------------------------------------------------------
# 5. Wordpiece length analysis
# ----------------------------------------------------------------------
def wordpiece_analysis(corpus, claims):
    print("\n=== WORDPIECE LENGTH ANALYSIS ===")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("`transformers` not installed; skipping. (pip install transformers)")
        return

    models = {
        "MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", 256),
        "SciBERT": ("allenai/scibert_scivocab_uncased", 512),
    }
    doc_texts = [doc_text(d) for d in corpus.values()]

    # Stage 2 pairs (claim + each sentence in gold docs) and
    # Stage 3 inputs (claim + all rationale sentences of one rationale set)
    pair_claims, pair_sents, s3_claims, s3_rats = [], [], [], []
    for c in claims["train"] + claims["dev"]:
        for doc_id, ev_list in c.get("evidence", {}).items():
            sents = corpus[int(doc_id)]["abstract"]
            for s in sents:
                pair_claims.append(c["claim"])
                pair_sents.append(s)
            for e in ev_list:
                s3_claims.append(c["claim"])
                s3_rats.append(" ".join(sents[i] for i in e["sentences"]))

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    for ax, (name, (hf_name, default_limit)) in zip(np.atleast_1d(axes), models.items()):
        try:
            tok = AutoTokenizer.from_pretrained(hf_name)
        except Exception as e:
            print(f"Could not load tokenizer {hf_name}: {e}")
            continue
        tok.model_max_length = 10 ** 6  # silence over-length warnings

        def lens(texts, text_pair=None):
            enc = tok(texts, text_pair, add_special_tokens=True, truncation=False)
            return np.array([len(x) for x in enc["input_ids"]])

        doc_len = lens(doc_texts)
        print(f"\n--- {name} ({hf_name}) ---")
        print(f"Whole abstract (title + text, incl. special tokens):")
        print(f"  mean {doc_len.mean():.0f} | median {np.median(doc_len):.0f} | "
              f"p95 {np.percentile(doc_len, 95):.0f} | max {doc_len.max()}")
        for lim in sorted({256, 512, default_limit}):
            print(f"  > {lim} wordpieces: {(doc_len > lim).mean():.1%} of abstracts")

        p2 = lens(pair_claims, pair_sents)
        print(f"Stage 2 pairs [claim, sentence] (n={len(p2)}): "
              f"mean {p2.mean():.0f} | p99 {np.percentile(p2, 99):.0f} | max {p2.max()} | "
              f"> 128: {(p2 > 128).mean():.2%} | > 256: {(p2 > 256).mean():.2%}")
        p3 = lens(s3_claims, s3_rats)
        print(f"Stage 3 inputs [claim, rationale] (n={len(p3)}): "
              f"mean {p3.mean():.0f} | p99 {np.percentile(p3, 99):.0f} | max {p3.max()} | "
              f"> 128: {(p3 > 128).mean():.2%} | > 256: {(p3 > 256).mean():.2%}")

        ax.hist(np.clip(doc_len, 0, 1000), bins=50, color="teal")
        for lim, col in [(256, "orange"), (512, "red")]:
            ax.axvline(lim, color=col, linestyle="--", label=f"{lim}")
        ax.set(title=f"{name}: abstract length", xlabel="wordpieces (clipped at 1000)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wordpiece_lengths.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# 6. BM25 retrieval + failure analysis
# ----------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


def evaluate_bm25(corpus, claims, tok_fn, desc="BM25"):
    """Returns one record per dev claim that has gold evidence.
    ranks: 1-indexed BM25 rank of each gold document."""
    doc_ids = list(corpus.keys())
    bm25 = BM25Okapi([tok_fn(doc_text(corpus[i])) for i in doc_ids])
    pos_of = {d: i for i, d in enumerate(doc_ids)}

    records = []
    for c in tqdm(claims["dev"], desc=desc):
        gold = [int(d) for d in c.get("evidence", {}).keys()]
        if not gold:
            continue
        scores = bm25.get_scores(tok_fn(c["claim"]))
        order = np.argsort(-scores)
        rank_of = np.empty(len(order), dtype=int)
        rank_of[order] = np.arange(1, len(order) + 1)
        ranks = {g: int(rank_of[pos_of[g]]) for g in gold}
        records.append({
            "id": c["id"], "claim": c["claim"], "gold": gold, "ranks": ranks,
            "best_rank": min(ranks.values()), "worst_rank": max(ranks.values()),
        })
    return records


def recall_table(records, ks=KS):
    n = len(records)
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "any_gold_recall": sum(r["best_rank"] <= k for r in records) / n,
            "all_gold_recall": sum(r["worst_rank"] <= k for r in records) / n,
        })
    return pd.DataFrame(rows).set_index("k")


def content_tokens(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def coverage(claim, target_text):
    ct = set(content_tokens(claim))
    if not ct:
        return 0.0
    return len(ct & set(content_tokens(target_text))) / len(ct)


def retrieval_analysis(corpus, claims):
    print("\n=== BM25 RETRIEVAL (dev, claims with gold evidence) ===")
    records = evaluate_bm25(corpus, claims, tokenize, desc="BM25 (raw text)")
    n = len(records)
    print(f"\nClaims evaluated: {n}  "
          f"(claims with >1 gold doc: {sum(len(r['gold']) > 1 for r in records)})")
    table = recall_table(records)
    print(table.round(3))
    print("any_gold = at least one gold doc retrieved; "
          "all_gold = every gold doc retrieved (stricter).")

    # Failure analysis
    dev_by_id = {c["id"]: c for c in claims["dev"]}
    groups = {"found@5": [], "rank 6-50": [], "rank >50": []}
    for r in records:
        if r["best_rank"] <= 5:
            groups["found@5"].append(r)
        elif r["best_rank"] <= 50:
            groups["rank 6-50"].append(r)
        else:
            groups["rank >50"].append(r)

    print("\n--- Lexical overlap between claim and gold evidence, by retrieval outcome ---")
    print("(fraction of the claim's content words that appear in the text)")
    rows, miss_rows = [], []
    for gname, recs in groups.items():
        cov_rat, cov_abs = [], []
        for r in recs:
            c = dev_by_id[r["id"]]
            best_doc = min(r["ranks"], key=r["ranks"].get)
            d = corpus[best_doc]
            rat_idx = {i for e in c["evidence"][str(best_doc)] for i in e["sentences"]}
            rat_text = " ".join(d["abstract"][i] for i in sorted(rat_idx))
            cov_rat.append(coverage(r["claim"], rat_text))
            cov_abs.append(coverage(r["claim"], doc_text(d)))
            if gname != "found@5":
                miss_rows.append({
                    "group": gname, "claim_id": r["id"], "claim": r["claim"],
                    "gold_doc": best_doc, "best_gold_rank": r["best_rank"],
                    "gold_title": d["title"], "gold_rationale": rat_text,
                    "claim_coverage_in_rationale": round(cov_rat[-1], 2),
                })
        rows.append({
            "group": gname, "n_claims": len(recs),
            "mean claim coverage in rationale": np.mean(cov_rat) if recs else np.nan,
            "mean claim coverage in abstract": np.mean(cov_abs) if recs else np.nan,
        })
    print(pd.DataFrame(rows).set_index("group").round(3))

    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(OUT_DIR / "bm25_dev_misses.csv", index=False, encoding="utf-8")
    print(f"\nSaved {len(miss_df)} claims with best gold rank > 5 to "
          f"{OUT_DIR / 'bm25_dev_misses.csv'}")

    severe = [m for m in miss_rows if m["group"] == "rank >50"]
    print(f"\n--- Examples where no gold doc is in the BM25 top 50 "
          f"({len(severe)} claims; showing up to 8) ---")
    for m in severe[:8]:
        print(f"\nClaim {m['claim_id']}: {m['claim']}")
        print(f"  Gold doc {m['gold_doc']} (rank {m['best_gold_rank']}): {m['gold_title']}")
        print(f"  Rationale: {m['gold_rationale'][:300]}")
        print(f"  Claim-word coverage in rationale: {m['claim_coverage_in_rationale']}")
    return records


# ----------------------------------------------------------------------
# 7. Text quality: run-together words + cleaned-vs-raw BM25
# ----------------------------------------------------------------------
def alpha_tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def find_merged_tokens(corpus, claims, min_len=8, part_min_freq=50, whole_max_freq=2):
    """Heuristic: a rare token that splits into two frequent corpus words
    (e.g. 'inductiveproperties' -> 'inductive' + 'properties') is probably a
    missing-space extraction artifact. Frequencies come from the corpus itself.
    Expect a few false positives; inspect the printed examples."""
    counts = Counter()
    for d in corpus.values():
        counts.update(alpha_tokens(doc_text(d)))

    candidates = set()
    for d in corpus.values():
        candidates.update(alpha_tokens(doc_text(d)))
    for split in ("train", "dev", "test"):
        for c in claims[split]:
            candidates.update(alpha_tokens(c["claim"]))
    candidates = {t for t in candidates if len(t) >= min_len and counts[t] <= whole_max_freq}

    merged = {}
    for t in candidates:
        best, best_score = None, 0
        for i in range(3, len(t) - 2):
            a, b = t[:i], t[i:]
            if counts[a] >= part_min_freq and counts[b] >= part_min_freq:
                score = min(counts[a], counts[b])
                if score > best_score:
                    best, best_score = (a, b), score
        if best:
            merged[t] = best
    return merged


def text_quality_analysis(corpus, claims, raw_records):
    print("\n=== TEXT QUALITY: RUN-TOGETHER WORDS ===")
    merged = find_merged_tokens(corpus, claims)
    print(f"Distinct suspected merged tokens: {len(merged)}")

    docs_aff = sum(any(t in merged for t in alpha_tokens(doc_text(d))) for d in corpus.values())
    n_tok = sum(len(alpha_tokens(doc_text(d))) for d in corpus.values())
    occ = sum(t in merged for d in corpus.values() for t in alpha_tokens(doc_text(d)))
    all_claims = [c for s in ("train", "dev", "test") for c in claims[s]]
    claims_aff = sum(any(t in merged for t in alpha_tokens(c["claim"])) for c in all_claims)
    print(f"Abstracts with >=1 suspected merge: {docs_aff} / {len(corpus)} "
          f"({docs_aff / len(corpus):.1%})")
    print(f"Token occurrences affected: {occ} / {n_tok} ({occ / n_tok:.3%})")
    print(f"Claims with >=1 suspected merge: {claims_aff} / {len(all_claims)} "
          f"({claims_aff / len(all_claims):.1%})")

    print("\nSample of suspected merges (inspect for false positives):")
    for t in sorted(merged, key=len, reverse=True)[:5] + sorted(merged)[:15]:
        print(f"  {t}  ->  {merged[t][0]} + {merged[t][1]}")
    pd.DataFrame(
        [{"token": t, "part_a": a, "part_b": b} for t, (a, b) in sorted(merged.items())]
    ).to_csv(OUT_DIR / "suspected_merged_tokens.csv", index=False, encoding="utf-8")

    def clean_tokenize(text):
        out = []
        for t in tokenize(text):
            if t in merged:
                out.extend(merged[t])
            else:
                out.append(t)
        return out

    print("\n--- BM25 recall: raw vs. cleaned (merged words split) ---")
    clean_records = evaluate_bm25(corpus, claims, clean_tokenize, desc="BM25 (cleaned)")
    raw_t, clean_t = recall_table(raw_records), recall_table(clean_records)
    comp = pd.DataFrame({
        "raw_any": raw_t["any_gold_recall"], "clean_any": clean_t["any_gold_recall"],
        "raw_all": raw_t["all_gold_recall"], "clean_all": clean_t["all_gold_recall"],
    })
    print(comp.round(3))
    comp.to_csv(OUT_DIR / "bm25_raw_vs_clean.csv", encoding="utf-8")
    print("Differences of a point or two on ~188 claims are within noise; "
          "report this as 'no meaningful change' unless the gap is larger.")


# ----------------------------------------------------------------------
def main():
    download_scifact()
    corpus, claims = load_data()
    corpus_stats(corpus)
    claim_stats(claims)
    rationale_position(claims, corpus)
    sentence_class_balance(claims, corpus)
    show_examples(corpus, claims)
    wordpiece_analysis(corpus, claims)
    raw_records = retrieval_analysis(corpus, claims)
    text_quality_analysis(corpus, claims, raw_records)
    print(f"\nFigures saved to {FIG_DIR.resolve()}")
    print(f"CSVs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()