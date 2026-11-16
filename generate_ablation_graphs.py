from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "risk_fusion_ablation.json"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "ablation_graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


processed = data["images_processed"]

baseline_actions = data[
    "baseline"
]["actions"]

proposed_actions = data[
    "proposed_risk_fusion"
]["actions"]

risk_levels = data[
    "proposed_risk_fusion"
]["risk_levels"]

agent_conflicts = data[
    "proposed_risk_fusion"
]["agent_conflicts"]


# ============================================================
# ACTION COUNTS
# ============================================================

actions = [
    "GO",
    "SLOW",
    "STOP"
]

baseline_counts = [
    baseline_actions.get(
        action,
        0
    )
    for action in actions
]

proposed_counts = [
    proposed_actions.get(
        action,
        0
    )
    for action in actions
]


baseline_percentages = [
    count / processed * 100
    for count in baseline_counts
]

proposed_percentages = [
    count / processed * 100
    for count in proposed_counts
]


# ============================================================
# GRAPH 1
# BASELINE VS PROPOSED ACTION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

x = range(
    len(actions)
)

width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    baseline_percentages,
    width=width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    proposed_percentages,
    width=width,
    label="Proposed Risk Fusion"
)

plt.xticks(
    list(x),
    actions
)

plt.ylabel(
    "Percentage of Images (%)"
)

plt.xlabel(
    "Final Driving Action"
)

plt.title(
    "Baseline vs Risk-Fusion Decision Distribution"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

for i, value in enumerate(
    baseline_percentages
):

    plt.text(
        i - width / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


for i, value in enumerate(
    proposed_percentages
):

    plt.text(
        i + width / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "baseline_vs_proposed_actions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2
# DECISION SHIFT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

difference = [
    proposed_percentages[i]
    - baseline_percentages[i]
    for i in range(
        len(actions)
    )
]

plt.bar(
    actions,
    difference
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel(
    "Change in Decision Frequency (percentage points)"
)

plt.xlabel(
    "Final Driving Action"
)

plt.title(
    "Decision Shift Introduced by Risk Fusion"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for action, value in zip(
    actions,
    difference
):

    plt.text(
        action,
        value + (
            1
            if value >= 0
            else -2
        ),
        f"{value:+.2f}",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "decision_shift.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 3
# PROPOSED RISK DISTRIBUTION
# ============================================================

risk_names = [
    "LOW",
    "MODERATE",
    "HIGH"
]

risk_counts = [
    risk_levels.get(
        risk,
        0
    )
    for risk in risk_names
]

risk_percentages = [
    count / processed * 100
    for count in risk_counts
]


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    risk_names,
    risk_percentages
)

plt.xlabel(
    "Fused Risk Level"
)

plt.ylabel(
    "Percentage of Images (%)"
)

plt.title(
    "Risk-Level Distribution of Proposed Framework"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for name, value in zip(
    risk_names,
    risk_percentages
):

    plt.text(
        name,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "risk_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 4
# AGENT CONFLICT RESULT
# ============================================================

no_conflicts = (
    processed
    - agent_conflicts
)

conflict_names = [
    "No Conflict",
    "Conflict"
]

conflict_counts = [
    no_conflicts,
    agent_conflicts
]

conflict_percentages = [
    value / processed * 100
    for value in conflict_counts
]


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    conflict_names,
    conflict_percentages
)

plt.xlabel(
    "Agent Agreement State"
)

plt.ylabel(
    "Percentage of Images (%)"
)

plt.title(
    "Agent Conflict Occurrence in 10K BDD100K Evaluation"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for name, value in zip(
    conflict_names,
    conflict_percentages
):

    plt.text(
        name,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "agent_conflict_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 5
# PROCESSING COMPARISON
#
# IMPORTANT:
# These are only the decision-layer timings,
# NOT end-to-end YOLO throughput.
# ============================================================

baseline_fps = data[
    "baseline"
]["fps"]

proposed_fps = data[
    "proposed_risk_fusion"
]["fps"]


plt.figure(
    figsize=(9, 6)
)

systems = [
    "Baseline\nDecision Layer",
    "Proposed\nRisk Fusion Layer"
]

fps_values = [
    baseline_fps,
    proposed_fps
]

plt.bar(
    systems,
    fps_values
)

plt.ylabel(
    "Decision-Layer Throughput (FPS)"
)

plt.title(
    "Decision-Layer Processing Throughput"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for system, value in zip(
    systems,
    fps_values
):

    plt.text(
        system,
        value + max(
            fps_values
        ) * 0.02,
        f"{value:.2f}",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "decision_layer_throughput.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE NUMERICAL SUMMARY
# ============================================================

summary = {

    "images_processed":
        processed,

    "baseline":
        {
            "GO":
                round(
                    baseline_percentages[0],
                    2
                ),

            "SLOW":
                round(
                    baseline_percentages[1],
                    2
                ),

            "STOP":
                round(
                    baseline_percentages[2],
                    2
                )
        },

    "proposed":
        {
            "GO":
                round(
                    proposed_percentages[0],
                    2
                ),

            "SLOW":
                round(
                    proposed_percentages[1],
                    2
                ),

            "STOP":
                round(
                    proposed_percentages[2],
                    2
                ),

            "LOW_RISK":
                round(
                    risk_percentages[0],
                    2
                ),

            "MODERATE_RISK":
                round(
                    risk_percentages[1],
                    2
                ),

            "HIGH_RISK":
                round(
                    risk_percentages[2],
                    2
                )
        },

    "decision_shift_percentage_points":
        {
            action:
                round(
                    difference[i],
                    2
                )
            for i, action
            in enumerate(actions)
        },

    "agent_conflicts":
        agent_conflicts,

    "agent_conflict_percentage":
        round(
            agent_conflicts
            / processed
            * 100,
            2
        ),

    "decision_layer_fps":
        {
            "baseline":
                round(
                    baseline_fps,
                    2
                ),

            "proposed":
                round(
                    proposed_fps,
                    2
                )
        }
}


summary_file = (
    OUTPUT_DIR
    / "ablation_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("ABLATION GRAPHS GENERATED")
print("=" * 70)

print()
print(
    f"Images processed: {processed}"
)

print()
print("Baseline:")
print(
    f"  GO   : {baseline_percentages[0]:.2f}%"
)
print(
    f"  SLOW : {baseline_percentages[1]:.2f}%"
)
print(
    f"  STOP : {baseline_percentages[2]:.2f}%"
)

print()
print("Proposed Risk Fusion:")
print(
    f"  GO   : {proposed_percentages[0]:.2f}%"
)
print(
    f"  SLOW : {proposed_percentages[1]:.2f}%"
)
print(
    f"  STOP : {proposed_percentages[2]:.2f}%"
)

print()
print("Risk Distribution:")
print(
    f"  LOW      : {risk_percentages[0]:.2f}%"
)
print(
    f"  MODERATE : {risk_percentages[1]:.2f}%"
)
print(
    f"  HIGH     : {risk_percentages[2]:.2f}%"
)

print()
print(
    f"Agent conflicts: "
    f"{agent_conflicts}"
)

print()
print("Graphs saved to:")

print(
    OUTPUT_DIR
)

print()
print("=" * 70)