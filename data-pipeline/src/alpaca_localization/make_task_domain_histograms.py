#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e


def plot_hist(counter: Counter, title: str, out_path: str, xlabel: str):
    # Sort by count (desc) then label (asc) for stable output
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Count task+domain labels in a labeled JSONL and save histograms for primary/secondary."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to labeled JSONL (must contain primary_task, secondary_tasks, primary_domain, secondary_domains).",
    )

    # Task histogram outputs
    parser.add_argument("--task-primary-out", default="task_primary_hist.png", help="Output PNG for primary task histogram.")
    parser.add_argument("--task-secondary-out", default="task_secondary_hist.png", help="Output PNG for secondary task histogram.")

    # Domain histogram outputs
    parser.add_argument("--domain-primary-out", default="domain_primary_hist.png", help="Output PNG for primary domain histogram.")
    parser.add_argument("--domain-secondary-out", default="domain_secondary_hist.png", help="Output PNG for secondary domain histogram.")

    parser.add_argument("--top", type=int, default=0, help="If >0, print only top-N labels for each histogram.")
    args = parser.parse_args()

    # Counters
    task_primary_counts: Counter = Counter()
    task_secondary_counts: Counter = Counter()
    domain_primary_counts: Counter = Counter()
    domain_secondary_counts: Counter = Counter()

    # Diagnostics
    n_rows = 0
    n_missing_task_primary = 0
    n_missing_task_secondary = 0
    n_missing_domain_primary = 0
    n_missing_domain_secondary = 0

    for row in iter_jsonl(args.input):
        n_rows += 1

        # ---- tasks ----
        t_primary = row.get("primary_task", None)
        if isinstance(t_primary, str) and t_primary.strip():
            task_primary_counts[t_primary.strip()] += 1
        else:
            n_missing_task_primary += 1

        t_secondary = row.get("secondary_tasks", None)
        if isinstance(t_secondary, list):
            for t in t_secondary:
                if isinstance(t, str) and t.strip():
                    task_secondary_counts[t.strip()] += 1
        else:
            n_missing_task_secondary += 1

        # ---- domains ----
        d_primary = row.get("primary_domain", None)
        if isinstance(d_primary, str) and d_primary.strip():
            domain_primary_counts[d_primary.strip()] += 1
        else:
            n_missing_domain_primary += 1

        d_secondary = row.get("secondary_domains", None)
        if isinstance(d_secondary, list):
            for d in d_secondary:
                if isinstance(d, str) and d.strip():
                    domain_secondary_counts[d.strip()] += 1
        else:
            n_missing_domain_secondary += 1

    print(f"Rows: {n_rows}")
    print(f"Missing/invalid primary_task: {n_missing_task_primary}")
    print(f"Missing/invalid secondary_tasks: {n_missing_task_secondary}")
    print(f"Missing/invalid primary_domain: {n_missing_domain_primary}")
    print(f"Missing/invalid secondary_domains: {n_missing_domain_secondary}")
    print()

    def print_counts(name: str, c: Counter):
        items = c.most_common(args.top) if args.top and args.top > 0 else c.most_common()
        print(name)
        for k, v in items:
            print(f"  {k:24s} {v}")
        print()

    print_counts("Primary task counts:", task_primary_counts)
    print_counts("Secondary task counts:", task_secondary_counts)
    print_counts("Primary domain counts:", domain_primary_counts)
    print_counts("Secondary domain counts:", domain_secondary_counts)

    # ---- plots ----
    if task_primary_counts:
        plot_hist(task_primary_counts, "Primary task distribution", args.task_primary_out, xlabel="Task type")
        print(f"[saved] {args.task_primary_out}")
    else:
        print("[warn] No primary_task labels found; primary task histogram not created.")

    if task_secondary_counts:
        plot_hist(task_secondary_counts, "Secondary task distribution", args.task_secondary_out, xlabel="Task type")
        print(f"[saved] {args.task_secondary_out}")
    else:
        print("[warn] No secondary_tasks labels found; secondary task histogram not created.")

    if domain_primary_counts:
        plot_hist(domain_primary_counts, "Primary domain distribution", args.domain_primary_out, xlabel="Domain")
        print(f"[saved] {args.domain_primary_out}")
    else:
        print("[warn] No primary_domain labels found; primary domain histogram not created.")

    if domain_secondary_counts:
        plot_hist(domain_secondary_counts, "Secondary domain distribution", args.domain_secondary_out, xlabel="Domain")
        print(f"[saved] {args.domain_secondary_out}")
    else:
        print("[warn] No secondary_domains labels found; secondary domain histogram not created.")


if __name__ == "__main__":
    main()
