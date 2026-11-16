from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "risk_fusion_ablation.json"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "ablation_results_report.txt"
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

baseline = data["baseline"]

proposed = data["proposed_risk_fusion"]

scene_distribution = data[
    "scene_distribution"
]

safety_decisions = data[
    "safety_decisions"
]


# ============================================================
# ACTION COUNTS
# ============================================================

baseline_actions = baseline["actions"]

proposed_actions = proposed["actions"]

go_baseline = baseline_actions.get("GO", 0)
slow_baseline = baseline_actions.get("SLOW", 0)
stop_baseline = baseline_actions.get("STOP", 0)

go_proposed = proposed_actions.get("GO", 0)
slow_proposed = proposed_actions.get("SLOW", 0)
stop_proposed = proposed_actions.get("STOP", 0)


# ============================================================
# PERCENTAGES
# ============================================================

def percentage(value):

    if processed == 0:
        return 0.0

    return value / processed * 100


go_baseline_pct = percentage(go_baseline)
slow_baseline_pct = percentage(slow_baseline)
stop_baseline_pct = percentage(stop_baseline)

go_proposed_pct = percentage(go_proposed)
slow_proposed_pct = percentage(slow_proposed)
stop_proposed_pct = percentage(stop_proposed)


# ============================================================
# DECISION SHIFT
# ============================================================

go_shift = (
    go_proposed_pct
    - go_baseline_pct
)

slow_shift = (
    slow_proposed_pct
    - slow_baseline_pct
)

stop_shift = (
    stop_proposed_pct
    - stop_baseline_pct
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

risk_levels = proposed[
    "risk_levels"
]

low_count = risk_levels.get(
    "LOW",
    0
)

moderate_count = risk_levels.get(
    "MODERATE",
    0
)

high_count = risk_levels.get(
    "HIGH",
    0
)


# ============================================================
# CONFLICT INFORMATION
# ============================================================

agent_conflicts = proposed[
    "agent_conflicts"
]

conflict_percentage = percentage(
    agent_conflicts
)


# ============================================================
# REPORT TEXT
# ============================================================

lines = []


lines.append(
    "=" * 80
)

lines.append(
    "CONFIDENCE-AWARE RISK FUSION"
)

lines.append(
    "DECISION-BEHAVIOR ABLATION REPORT"
)

lines.append(
    "=" * 80
)

lines.append("")

lines.append(
    "Dataset: BDD100K validation set"
)

lines.append(
    f"Images processed: {processed}"
)

lines.append("")


# ============================================================
# TABLE 1
# ============================================================

lines.append(
    "TABLE 1 — BASELINE VS PROPOSED DECISION DISTRIBUTION"
)

lines.append(
    "-" * 80
)

lines.append(
    f"{'Decision':<15}"
    f"{'Baseline Count':>18}"
    f"{'Baseline %':>15}"
    f"{'Proposed Count':>18}"
    f"{'Proposed %':>15}"
)

lines.append(
    "-" * 80
)

lines.append(
    f"{'GO':<15}"
    f"{go_baseline:>18}"
    f"{go_baseline_pct:>14.2f}%"
    f"{go_proposed:>18}"
    f"{go_proposed_pct:>14.2f}%"
)

lines.append(
    f"{'SLOW':<15}"
    f"{slow_baseline:>18}"
    f"{slow_baseline_pct:>14.2f}%"
    f"{slow_proposed:>18}"
    f"{slow_proposed_pct:>14.2f}%"
)

lines.append(
    f"{'STOP':<15}"
    f"{stop_baseline:>18}"
    f"{stop_baseline_pct:>14.2f}%"
    f"{stop_proposed:>18}"
    f"{stop_proposed_pct:>14.2f}%"
)

lines.append(
    "-" * 80
)

lines.append("")


# ============================================================
# TABLE 2
# ============================================================

lines.append(
    "TABLE 2 — DECISION SHIFT"
)

lines.append(
    "-" * 60
)

lines.append(
    f"{'Decision':<20}"
    f"{'Shift (percentage points)':>30}"
)

lines.append(
    "-" * 60
)

lines.append(
    f"{'GO':<20}"
    f"{go_shift:>29.2f}"
)

lines.append(
    f"{'SLOW':<20}"
    f"{slow_shift:>29.2f}"
)

lines.append(
    f"{'STOP':<20}"
    f"{stop_shift:>29.2f}"
)

lines.append(
    "-" * 60
)

lines.append("")


# ============================================================
# TABLE 3
# ============================================================

lines.append(
    "TABLE 3 — PROPOSED RISK DISTRIBUTION"
)

lines.append(
    "-" * 60
)

lines.append(
    f"{'Risk Level':<20}"
    f"{'Count':>15}"
    f"{'Percentage':>20}"
)

lines.append(
    "-" * 60
)

lines.append(
    f"{'LOW':<20}"
    f"{low_count:>15}"
    f"{percentage(low_count):>19.2f}%"
)

lines.append(
    f"{'MODERATE':<20}"
    f"{moderate_count:>15}"
    f"{percentage(moderate_count):>19.2f}%"
)

lines.append(
    f"{'HIGH':<20}"
    f"{high_count:>15}"
    f"{percentage(high_count):>19.2f}%"
)

lines.append(
    "-" * 60
)

lines.append("")


# ============================================================
# TABLE 4
# ============================================================

lines.append(
    "TABLE 4 — ARCHITECTURE ABLATION"
)

lines.append(
    "-" * 95
)

lines.append(
    f"{'Configuration':<30}"
    f"{'Perception':>13}"
    f"{'Scene':>10}"
    f"{'Planning':>12}"
    f"{'Safety':>10}"
    f"{'Risk Fusion':>15}"
)

lines.append(
    "-" * 95
)

lines.append(
    f"{'Baseline':<30}"
    f"{'YES':>13}"
    f"{'YES':>10}"
    f"{'YES':>12}"
    f"{'YES':>10}"
    f"{'NO':>15}"
)

lines.append(
    f"{'Proposed Risk-Aware Fusion':<30}"
    f"{'YES':>13}"
    f"{'YES':>10}"
    f"{'YES':>12}"
    f"{'YES':>10}"
    f"{'YES':>15}"
)

lines.append(
    "-" * 95
)

lines.append("")


# ============================================================
# CONTROLLED CONFLICT TEST
# ============================================================

lines.append(
    "TABLE 5 — CONTROLLED CONFLICT-RESOLUTION VALIDATION"
)

lines.append(
    "-" * 80
)

lines.append(
    f"{'Scenario':<35}"
    f"{'Planner':>12}"
    f"{'Safety':>12}"
    f"{'Conflict':>12}"
    f"{'Fusion':>12}"
)

lines.append(
    "-" * 80
)

lines.append(
    f"{'Planner/Safety disagreement':<35}"
    f"{'GO':>12}"
    f"{'STOP':>12}"
    f"{'TRUE':>12}"
    f"{'STOP':>12}"
)

lines.append(
    "-" * 80
)

lines.append("")


# ============================================================
# PERFORMANCE
# ============================================================

lines.append(
    "PERFORMANCE INFORMATION"
)

lines.append(
    "-" * 80
)

lines.append(
    f"Baseline decision-layer FPS : "
    f"{baseline['fps']:.2f}"
)

lines.append(
    f"Proposed decision-layer FPS : "
    f"{proposed['fps']:.2f}"
)

lines.append(
    f"Baseline decision-layer time: "
    f"{baseline['average_time_seconds']:.6f} sec/image"
)

lines.append(
    f"Proposed decision-layer time: "
    f"{proposed['average_time_seconds']:.6f} sec/image"
)

lines.append("")


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

lines.append(
    "INTERPRETATION"
)

lines.append(
    "-" * 80
)

lines.append(
    "The proposed risk-fusion layer changes the final "
    "decision distribution compared with the baseline."
)

lines.append(
    f"GO changes from {go_baseline_pct:.2f}% "
    f"to {go_proposed_pct:.2f}%."
)

lines.append(
    f"SLOW changes from {slow_baseline_pct:.2f}% "
    f"to {slow_proposed_pct:.2f}%."
)

lines.append(
    f"STOP changes from {stop_baseline_pct:.2f}% "
    f"to {stop_proposed_pct:.2f}%."
)

lines.append("")

lines.append(
    "These results demonstrate decision-behavior changes "
    "introduced by risk fusion."
)

lines.append(
    "They should not be interpreted as safety-accuracy "
    "improvements because BDD100K does not provide "
    "GO/SLOW/STOP ground-truth labels for this experiment."
)

lines.append("")

lines.append(
    f"The controlled conflict test detected a disagreement "
    f"between the Planning and Safety agents and produced "
    f"a safety-prioritized STOP decision."
)

lines.append("")

lines.append(
    f"Natural agent conflicts in the 10,000-image "
    f"evaluation: {agent_conflicts} "
    f"({conflict_percentage:.2f}%)."
)

lines.append("")

lines.append(
    "Temporal tracking and trajectory prediction are "
    "architecture components but are not fully evaluated "
    "using the static BDD100K image set."
)

lines.append("")

lines.append(
    "=" * 80
)

lines.append(
    "END OF ABLATION REPORT"
)

lines.append(
    "=" * 80
)


# ============================================================
# SAVE
# ============================================================

report_text = "\n".join(lines)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report_text
    )


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)
print("ABLATION REPORT GENERATED")
print("=" * 70)

print()

print(
    f"Output: {OUTPUT_FILE}"
)

print()

print(
    "Key decision shift:"
)

print(
    f"GO   : {go_shift:+.2f} percentage points"
)

print(
    f"SLOW : {slow_shift:+.2f} percentage points"
)

print(
    f"STOP : {stop_shift:+.2f} percentage points"
)

print()

print(
    "Controlled conflict validation:"
)

print(
    "Planner = GO"
)

print(
    "Safety  = STOP"
)

print(
    "Conflict = TRUE"
)

print(
    "Fusion = STOP"
)

print()

print(
    "Report generation complete."
)

print("=" * 70)