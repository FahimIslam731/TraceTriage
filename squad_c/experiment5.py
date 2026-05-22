"""Experiment 5: Cost-Aware Utility Analysis.

Computes:
  1. Budget curves     — recovery_success vs cumulative_cost as budget grows
  2. Pareto frontier   — which policies are non-dominated on (success, cost)
  3. Lambda sensitivity — which policy wins at each cost-weight lambda
  4. Cost regime       — budget ranges where trace_triage leads
  5. Action sensitivity — robustness when action costs are scaled up/down

Run:
    python -m squad_c.experiment5

Outputs to squad_c/results/:
    experiment5.json         — all computed data
    experiment5_summary.txt  — human-readable key findings
"""
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("squad_c/results")
RESULTS_JSONL = RESULTS_DIR / "recovery_results.jsonl"
LABELS_CSV = Path("squad_a/audit_results/all_1212_labels.csv")
CLF_DIR = RESULTS_DIR / "squad_b_best_classifier" / "Gemini"
DB_PATH = Path("data/causal_runs.sqlite")
OUT_JSON = RESULTS_DIR / "experiment5.json"
OUT_SUMMARY = RESULTS_DIR / "experiment5_summary.txt"

RETRIEVE_MORE_DOMAINS = {"SealQA", "MedBrowseComp", "BrowseComp"}
DOMAIN_MODAL = {
    "GSM8K": "LOCAL_REPAIR",
    "MBPP": "LOCAL_REPAIR",
    "SealQA": "REPLAN",
    "MedBrowseComp": "RETRIEVE_MORE",
}
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
ACTION_SCALE_FACTORS = [0.5, 1.0, 2.0, 5.0]  # cost multipliers for sensitivity analysis


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    # Recovery results
    results = defaultdict(dict)
    with open(RESULTS_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            results[r["trace_id"]][r["action"]] = {
                "success": r["success"],
                "cost": r["cost_usd"],
            }

    # Squad A labels
    with open(LABELS_CSV, encoding="utf-8") as f:
        squad_a = {row["trace_id"]: row["human_majority"] for row in csv.DictReader(f)}

    # Classifier predictions
    clf = {}
    for fp in sorted(CLF_DIR.glob("*.json")):
        for tid, rec in json.loads(fp.read_text(encoding="utf-8")).items():
            clf[tid] = rec["action"]

    # Domain map
    conn = sqlite3.connect(DB_PATH)
    domain_map = dict(conn.execute(
        "SELECT trace_id, domain FROM traces WHERE is_ablation=0"
    ).fetchall())
    conn.close()

    return results, squad_a, clf, domain_map


# ---------------------------------------------------------------------------
# Policy assignment — returns (cost, success) per trace for each policy
# ---------------------------------------------------------------------------

def assign_policies(results, squad_a, clf, domain_map, cost_scale=None):
    """Build {policy: [(cost, success), ...]} for all 1204 traces.

    cost_scale: optional {action: multiplier} for sensitivity analysis.
    """
    def pick(tr, action):
        r = tr.get(action)
        if not r:
            return None
        cost = r["cost"] * cost_scale.get(action, 1.0) if cost_scale else r["cost"]
        return (cost, r["success"])

    policies = {k: [] for k in [
        "always_retry", "always_local_repair", "always_replan",
        "always_retrieve_more", "domain_policy",
        "trace_triage", "trace_triage_clf", "oracle",
    ]}

    for tid, tr in results.items():
        domain = domain_map.get(tid, "")

        for pol, action in [
            ("always_retry", "RETRY"),
            ("always_local_repair", "LOCAL_REPAIR"),
            ("always_replan", "REPLAN"),
        ]:
            p = pick(tr, action)
            if p:
                policies[pol].append(p)

        arm = (pick(tr, "RETRIEVE_MORE") if domain in RETRIEVE_MORE_DOMAINS and "RETRIEVE_MORE" in tr
               else pick(tr, "LOCAL_REPAIR"))
        if arm:
            policies["always_retrieve_more"].append(arm)

        modal = DOMAIN_MODAL.get(domain, "ESCALATE")
        dp = pick(tr, modal) or pick(tr, "LOCAL_REPAIR")
        if dp:
            policies["domain_policy"].append(dp)

        label = squad_a.get(tid)
        if label:
            p = pick(tr, label)
            if p:
                policies["trace_triage"].append(p)

        pred = clf.get(tid)
        if pred:
            p = pick(tr, pred)
            if p:
                policies["trace_triage_clf"].append(p)

        succ = [(pick(tr, a)[0], True) for a in tr if tr[a]["success"] and pick(tr, a)]
        if succ:
            policies["oracle"].append(min(succ, key=lambda x: x[0]))
        else:
            esc = tr.get("ESCALATE", list(tr.values())[0])
            cost = esc["cost"] * cost_scale.get("ESCALATE", 1.0) if cost_scale else esc["cost"]
            policies["oracle"].append((cost, esc["success"]))

    return policies


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_summary(policies):
    """Aggregate metrics per policy."""
    out = {}
    for pol, data in policies.items():
        n = len(data)
        if n == 0:
            continue
        n_succ = sum(1 for c, s in data if s)
        total_cost = sum(c for c, s in data)
        avg_cost = total_cost / n
        rate = n_succ / n
        out[pol] = {
            "n": n,
            "n_success": n_succ,
            "recovery_rate": round(rate, 4),
            "total_cost": round(total_cost, 6),
            "avg_cost_per_trace": round(avg_cost, 8),
            "cost_per_success": round(total_cost / n_succ, 6) if n_succ else None,
            "success_per_dollar": round(n_succ / total_cost, 2) if total_cost > 0 else None,
        }
        for lam in LAMBDAS:
            key = f"utility_L{lam}".replace(".", "_")
            out[pol][key] = round(rate - lam * avg_cost, 6)
    return out


def budget_curves(policies, n_points=50):
    """For each policy, compute (cumulative_cost, success_rate) curve sorted by cost asc."""
    curves = {}
    for pol, data in policies.items():
        sorted_data = sorted(data, key=lambda x: x[0])
        cum_cost, cum_succ = 0.0, 0
        points = []
        for i, (cost, success) in enumerate(sorted_data):
            cum_cost += cost
            cum_succ += int(success)
            # Sample n_points evenly + always include last point
            step = max(1, len(sorted_data) // n_points)
            if i % step == 0 or i == len(sorted_data) - 1:
                points.append({
                    "cum_cost": round(cum_cost, 6),
                    "n_traces": i + 1,
                    "n_success": cum_succ,
                    "success_rate": round(cum_succ / (i + 1), 4),
                })
        curves[pol] = points
    return curves


def pareto_frontier(summary):
    """Find Pareto-optimal policies on (success_rate maximise, avg_cost minimise)."""
    points = [(pol, m["recovery_rate"], m["avg_cost_per_trace"])
              for pol, m in summary.items()]
    frontier = []
    for pol, rate, cost in points:
        dominated = any(
            r2 >= rate and c2 <= cost and (r2 > rate or c2 < cost)
            for _, r2, c2 in points
        )
        frontier.append({"policy": pol, "recovery_rate": rate,
                          "avg_cost": cost, "on_frontier": not dominated})
    frontier.sort(key=lambda x: x["recovery_rate"], reverse=True)
    return frontier


def lambda_sensitivity(summary):
    """For each lambda, rank policies by utility score."""
    results = {}
    for lam in LAMBDAS:
        key = f"utility_L{lam}".replace(".", "_")
        ranked = sorted(summary.items(), key=lambda x: x[1].get(key, 0), reverse=True)
        results[str(lam)] = [
            {"policy": pol, "utility": round(m.get(key, 0), 6),
             "recovery_rate": m["recovery_rate"]}
            for pol, m in ranked
        ]
    return results


def cost_regime_analysis(policies):
    """Find budget thresholds where trace_triage leads over domain_policy."""
    tt_data = sorted(policies["trace_triage"], key=lambda x: x[0])
    dp_data = sorted(policies["domain_policy"], key=lambda x: x[0])

    budgets = [round(i * 0.01, 3) for i in range(1, 201)]  # $0.01 to $2.00
    regime = []
    for budget in budgets:
        def success_at_budget(data):
            cum, succ = 0.0, 0
            for cost, success in data:
                if cum + cost > budget:
                    break
                cum += cost
                succ += int(success)
            return succ

        tt_succ = success_at_budget(tt_data)
        dp_succ = success_at_budget(dp_data)
        regime.append({
            "budget": budget,
            "trace_triage_success": tt_succ,
            "domain_policy_success": dp_succ,
            "tt_leads": tt_succ > dp_succ,
            "gap": tt_succ - dp_succ,
        })
    return regime


def action_cost_sensitivity(results, squad_a, clf, domain_map):
    """Scale each action's cost independently and recheck policy rankings."""
    actions = ["RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "LOCAL_REPAIR"]
    sensitivity = {}
    for action in actions:
        sensitivity[action] = {}
        for factor in ACTION_SCALE_FACTORS:
            scale = {action: factor}
            pols = assign_policies(results, squad_a, clf, domain_map, cost_scale=scale)
            summ = compute_summary(pols)
            # Record utility at lambda=1 and rank
            ranked = sorted(summ.items(),
                            key=lambda x: x[1].get("utility_L1_0", 0), reverse=True)
            sensitivity[action][str(factor)] = {
                "winner": ranked[0][0],
                "trace_triage_rank": next(i + 1 for i, (p, _) in enumerate(ranked)
                                          if p == "trace_triage"),
                "trace_triage_utility": round(summ["trace_triage"].get("utility_L1_0", 0), 6),
                "domain_policy_utility": round(summ["domain_policy"].get("utility_L1_0", 0), 6),
            }
    return sensitivity


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Consistent colours and styles per policy
POLICY_STYLE = {
    "oracle":              {"color": "#2ca02c", "lw": 2.5, "ls": "--", "zorder": 5},
    "trace_triage":        {"color": "#1f77b4", "lw": 2.5, "ls": "-",  "zorder": 4},
    "trace_triage_clf":    {"color": "#aec7e8", "lw": 1.8, "ls": "-",  "zorder": 3},
    "always_retrieve_more":{"color": "#ff7f0e", "lw": 1.5, "ls": "-",  "zorder": 2},
    "domain_policy":       {"color": "#9467bd", "lw": 1.5, "ls": "-",  "zorder": 2},
    "always_local_repair": {"color": "#8c564b", "lw": 1.2, "ls": ":",  "zorder": 1},
    "always_replan":       {"color": "#bcbd22", "lw": 1.2, "ls": ":",  "zorder": 1},
    "always_retry":        {"color": "#d62728", "lw": 1.2, "ls": ":",  "zorder": 1},
}

LABEL = {
    "oracle": "Oracle",
    "trace_triage": "Trace Triage (human labels)",
    "trace_triage_clf": "Trace Triage (classifier)",
    "always_retrieve_more": "Always RetrieveMore",
    "domain_policy": "Domain Policy",
    "always_local_repair": "Always LocalRepair",
    "always_replan": "Always Replan",
    "always_retry": "Always Retry",
}


def plot_budget_curves(curves, out_path):
    """Plot 1 — recovery success rate vs cumulative cost budget."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for pol, points in curves.items():
        xs = [p["cum_cost"] for p in points]
        ys = [p["success_rate"] for p in points]
        s = POLICY_STYLE.get(pol, {"color": "grey", "lw": 1, "ls": "-", "zorder": 0})
        ax.plot(xs, ys, label=LABEL.get(pol, pol),
                color=s["color"], linewidth=s["lw"],
                linestyle=s["ls"], zorder=s["zorder"])

    ax.set_xlabel("Cumulative Cost (USD)", fontsize=12)
    ax.set_ylabel("Recovery Success Rate", fontsize=12)
    ax.set_title("Recovery Success vs Budget — All Policies", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pareto_frontier(frontier, summary, out_path):
    """Plot 2 — scatter of (avg_cost, success_rate) with Pareto frontier highlighted."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for p in frontier:
        pol = p["policy"]
        s = POLICY_STYLE.get(pol, {"color": "grey", "zorder": 0})
        marker = "*" if p["on_frontier"] else "o"
        size = 180 if p["on_frontier"] else 80
        ax.scatter(p["avg_cost"], p["recovery_rate"],
                   color=s["color"], marker=marker, s=size,
                   zorder=s.get("zorder", 1) + 2,
                   label=LABEL.get(pol, pol))

    # Draw frontier line (sorted by cost)
    front_pts = sorted([p for p in frontier if p["on_frontier"]], key=lambda x: x["avg_cost"])
    if len(front_pts) >= 2:
        fx = [p["avg_cost"] for p in front_pts]
        fy = [p["recovery_rate"] for p in front_pts]
        ax.step(fx, fy, where="post", color="black", lw=1.2, ls="--", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Avg Cost per Trace (USD)", fontsize=12)
    ax.set_ylabel("Recovery Success Rate", fontsize=12)
    ax.set_title("Pareto Frontier: Success vs Cost", fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_lambda_sensitivity(lam_sens, out_path):
    """Plot 3 — utility score per policy across lambda values."""
    fig, ax = plt.subplots(figsize=(9, 6))

    lambdas = [float(k) for k in lam_sens.keys()]
    policies_ordered = list(POLICY_STYLE.keys())

    for pol in policies_ordered:
        ys = []
        for lam_key in lam_sens:
            entry = next((x for x in lam_sens[lam_key] if x["policy"] == pol), None)
            ys.append(entry["utility"] if entry else None)
        if all(y is None for y in ys):
            continue
        s = POLICY_STYLE.get(pol, {"color": "grey", "lw": 1, "ls": "-", "zorder": 0})
        ax.plot(lambdas, ys, label=LABEL.get(pol, pol),
                color=s["color"], linewidth=s["lw"],
                linestyle=s["ls"], zorder=s["zorder"])

    ax.set_xlabel("Lambda (cost penalty weight)", fontsize=12)
    ax.set_ylabel("Utility  (success_rate − λ × avg_cost)", fontsize=12)
    ax.set_title("Utility Sensitivity to Cost Weight (λ)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cost_regime(regime, out_path):
    """Plot 4 — gap between trace_triage and domain_policy across budget levels."""
    fig, ax = plt.subplots(figsize=(9, 5))

    budgets = [r["budget"] for r in regime]
    gaps = [r["gap"] for r in regime]
    colors = ["#1f77b4" if g > 0 else "#d62728" for g in gaps]

    ax.bar(budgets, gaps, width=0.009, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)

    blue_patch = mpatches.Patch(color="#1f77b4", alpha=0.8, label="Trace Triage leads")
    red_patch = mpatches.Patch(color="#d62728", alpha=0.8, label="Domain Policy leads")
    ax.legend(handles=[blue_patch, red_patch], fontsize=9)

    ax.set_xlabel("Total Budget (USD)", fontsize=12)
    ax.set_ylabel("Trace Triage − Domain Policy  (extra successes)", fontsize=11)
    ax.set_title("Cost Regime: Where Trace Triage Wins", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_policy_comparison(summary, out_path):
    """Plot 6 — horizontal bar chart of recovery rate for all 8 policies."""
    # Sort by recovery rate descending
    sorted_pols = sorted(summary.items(), key=lambda x: x[1]["recovery_rate"], reverse=True)
    names = [LABEL.get(p, p) for p, _ in sorted_pols]
    rates = [m["recovery_rate"] for _, m in sorted_pols]
    colors = [POLICY_STYLE.get(p, {"color": "grey"})["color"] for p, _ in sorted_pols]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, rates, color=colors, edgecolor="white", height=0.6)

    # Value labels at end of each bar
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Recovery Success Rate", fontsize=12)
    ax.set_title("Policy Comparison — Recovery Success Rate (Stage B, 1204 traces)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, max(rates) * 1.15)
    ax.invert_yaxis()  # highest rate at top
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_action_sensitivity(action_sens, out_path):
    """Plot 5 — trace_triage utility vs domain_policy as each action cost is scaled."""
    actions = list(action_sens.keys())
    factors = ACTION_SCALE_FACTORS
    n_actions = len(actions)

    fig, axes = plt.subplots(1, n_actions, figsize=(3.5 * n_actions, 5), sharey=True)
    if n_actions == 1:
        axes = [axes]

    for ax, action in zip(axes, actions):
        tt_vals = [action_sens[action][str(f)]["trace_triage_utility"] for f in factors]
        dp_vals = [action_sens[action][str(f)]["domain_policy_utility"] for f in factors]

        x = range(len(factors))
        ax.plot(x, tt_vals, "o-", color="#1f77b4", lw=2, label="Trace Triage")
        ax.plot(x, dp_vals, "s--", color="#9467bd", lw=1.5, label="Domain Policy")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{f}x" for f in factors], fontsize=9)
        ax.set_title(action, fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Utility (λ=1)", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle("Action Cost Sensitivity: Trace Triage vs Domain Policy (λ=1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def write_summary(summary, frontier, lam_sens, regime, out_path):
    oracle_rate = summary["oracle"]["recovery_rate"]
    tt_rate = summary["trace_triage"]["recovery_rate"]
    clf_rate = summary["trace_triage_clf"]["recovery_rate"]
    best_fixed = max(
        [summary[p] for p in ["always_retrieve_more", "domain_policy",
                               "always_local_repair", "always_replan", "always_retry"]],
        key=lambda x: x["recovery_rate"]
    )
    best_fixed_name = next(p for p, m in summary.items() if m == best_fixed)
    best_fixed_rate = best_fixed["recovery_rate"]
    best_fixed_cost = best_fixed["avg_cost_per_trace"]

    tt_cost = summary["trace_triage"]["avg_cost_per_trace"]
    cost_savings_pct = (best_fixed_cost - tt_cost) / best_fixed_cost * 100

    oracle_pct = tt_rate / oracle_rate * 100
    target_90 = oracle_rate * 0.9

    frontier_policies = [p["policy"] for p in frontier if p["on_frontier"]]

    budget_wins = [r for r in regime if r["tt_leads"]]
    budget_win_range = (
        f"${budget_wins[0]['budget']:.2f}–${budget_wins[-1]['budget']:.2f}"
        if budget_wins else "none"
    )

    winner_L1 = lam_sens["1.0"][0]["policy"]
    winner_L5 = lam_sens["5.0"][0]["policy"]

    lines = [
        "=== Experiment 5: Cost-Aware Utility Analysis ===",
        "",
        "--- 1. Key targets vs actuals ---",
        f"  Target: trace_triage achieves 90% of oracle     => need {target_90:.4f}",
        f"  Actual: trace_triage = {tt_rate:.4f}  ({oracle_pct:.1f}% of oracle)  {'PASS' if tt_rate >= target_90 else 'MISS'}",
        "",
        f"  Target: 30-40% cost savings vs next-best fixed",
        f"  Actual: trace_triage ${tt_cost:.6f} vs {best_fixed_name} ${best_fixed_cost:.6f}",
        f"          => {cost_savings_pct:.1f}% savings  {'PASS' if cost_savings_pct >= 30 else 'MISS'}",
        "",
        "--- 2. Pareto frontier ---",
        f"  Policies on frontier: {', '.join(frontier_policies)}",
        f"  trace_triage on frontier: {'YES' if 'trace_triage' in frontier_policies else 'NO'}",
        "",
        "--- 3. Lambda sensitivity (winner at each lambda) ---",
    ]
    for lam in LAMBDAS:
        winner = lam_sens[str(lam)][0]["policy"]
        tt_rank = next(i + 1 for i, p in enumerate(lam_sens[str(lam)]) if p["policy"] == "trace_triage")
        lines.append(f"  lambda={lam:<5} winner={winner:<24} trace_triage rank={tt_rank}")

    lines += [
        "",
        "--- 4. Budget regime ---",
        f"  trace_triage leads domain_policy at budgets: {budget_win_range}",
        f"  Number of budget points where TT leads: {len(budget_wins)}/200",
        "",
        "--- 5. Verdict ---",
        f"  trace_triage (human labels) = {tt_rate:.1%}  |  oracle = {oracle_rate:.1%}  |  best fixed = {best_fixed_rate:.1%}",
        f"  trace_triage_clf = {clf_rate:.1%}  (below best fixed — classifier needs improvement)",
        f"  Oracle gap: {oracle_rate - tt_rate:.1%} ppt that a better classifier could capture",
        "",
        f"  Paper claim status:",
        f"    90% of oracle: {'MET' if tt_rate >= target_90 else f'NOT MET ({oracle_pct:.0f}% achieved)'}",
        f"    30-40% cost savings: {'MET' if cost_savings_pct >= 30 else f'NOT MET ({cost_savings_pct:.0f}% achieved)'}",
        f"    Triage on Pareto frontier: {'YES' if 'trace_triage' in frontier_policies else 'NO'}",
        f"    Triage wins at lambda=1: {'YES' if winner_L1 == 'trace_triage' else f'NO (winner: {winner_L1})'}",
        f"    Triage wins at lambda=5: {'YES' if winner_L5 == 'trace_triage' else f'NO (winner: {winner_L5})'}",
    ]

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    results, squad_a, clf, domain_map = load_data()

    print("Assigning policies...")
    policies = assign_policies(results, squad_a, clf, domain_map)

    print("Computing summary metrics...")
    summary = compute_summary(policies)

    print("Computing budget curves...")
    curves = budget_curves(policies)

    print("Computing Pareto frontier...")
    frontier = pareto_frontier(summary)

    print("Computing lambda sensitivity...")
    lam_sens = lambda_sensitivity(summary)

    print("Computing cost regime analysis...")
    regime = cost_regime_analysis(policies)

    print("Computing action cost sensitivity...")
    action_sens = action_cost_sensitivity(results, squad_a, clf, domain_map)

    output = {
        "summary": summary,
        "budget_curves": curves,
        "pareto_frontier": frontier,
        "lambda_sensitivity": lam_sens,
        "cost_regime": regime,
        "action_cost_sensitivity": action_sens,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nFull results written to {OUT_JSON}")
    print(f"\n{'='*60}\n")

    write_summary(summary, frontier, lam_sens, regime, OUT_SUMMARY)
    print(f"\nSummary written to {OUT_SUMMARY}")

    print("\nGenerating plots...")
    plot_budget_curves(curves,    RESULTS_DIR / "plot_budget_curves.png")
    plot_pareto_frontier(frontier, summary, RESULTS_DIR / "plot_pareto_frontier.png")
    plot_lambda_sensitivity(lam_sens, RESULTS_DIR / "plot_lambda_sensitivity.png")
    plot_cost_regime(regime,      RESULTS_DIR / "plot_cost_regime.png")
    plot_action_sensitivity(action_sens, RESULTS_DIR / "plot_action_sensitivity.png")
    plot_policy_comparison(summary,      RESULTS_DIR / "plot_policy_comparison.png")
    print("All plots saved to squad_c/results/")


if __name__ == "__main__":
    main()
