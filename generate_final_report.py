import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"

JSON_FILE = RESULTS_DIR / "batch_results_10000.json"

REPORT_FILE = RESULTS_DIR / "final_evaluation_report.txt"


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 75)
print("BDD100K MULTI-AGENT AUTONOMOUS DRIVING")
print("FINAL SYSTEM EVALUATION")
print("=" * 75)

if not JSON_FILE.exists():

    raise FileNotFoundError(
        f"\nFinal results not found:\n{JSON_FILE}"
    )


with open(
    JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# DATA
# ============================================================

dataset = data.get(
    "dataset",
    "BDD100K"
)

split = data.get(
    "split",
    "validation"
)

available_images = data.get(
    "available_images",
    10000
)

processed_images = data.get(
    "processed_images",
    0
)

confidence = data.get(
    "confidence",
    0.25
)

average_objects = data.get(
    "average_objects_per_image",
    0
)

average_time = data.get(
    "average_processing_time_seconds",
    0
)

fps = data.get(
    "approximate_fps",
    0
)

actions = data.get(
    "action_distribution",
    {}
)

risks = data.get(
    "risk_distribution",
    {}
)

safety = data.get(
    "safety_distribution",
    {}
)

scenes = data.get(
    "scene_distribution",
    {}
)


# ============================================================
# HELPER
# ============================================================

def percentage(
    value,
    total
):

    if total == 0:

        return 0

    return (
        value /
        total *
        100
    )


# ============================================================
# PRINT DATASET
# ============================================================

print()
print("=" * 75)
print("DATASET")
print("-" * 75)

print(
    f"Dataset                  : {dataset}"
)

print(
    f"Split                    : {split}"
)

print(
    f"Available images         : {available_images:,}"
)

print(
    f"Images processed         : {processed_images:,}"
)

print(
    f"Confidence threshold     : {confidence}"
)


# ============================================================
# PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("SYSTEM PERFORMANCE")
print("-" * 75)

print(
    f"Average objects/image    : "
    f"{average_objects:.2f}"
)

print(
    f"Average processing time  : "
    f"{average_time:.4f} seconds"
)

print(
    f"Approximate throughput   : "
    f"{fps:.2f} FPS"
)


# ============================================================
# ACTION
# ============================================================

print()
print("=" * 75)
print("FINAL DRIVING ACTION")
print("-" * 75)

for action in [
    "GO",
    "SLOW",
    "STOP"
]:

    count = actions.get(
        action,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    print(
        f"{action:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# RISK
# ============================================================

print()
print("=" * 75)
print("SCENE RISK")
print("-" * 75)

for risk in [
    "LOW",
    "MODERATE",
    "HIGH"
]:

    count = risks.get(
        risk,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    print(
        f"{risk:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# SAFETY
# ============================================================

print()
print("=" * 75)
print("SAFETY AGENT")
print("-" * 75)

for decision in [
    "APPROVE",
    "OVERRIDE"
]:

    count = safety.get(
        decision,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    print(
        f"{decision:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# SCENE TYPES
# ============================================================

print()
print("=" * 75)
print("SCENE TYPES")
print("-" * 75)

for scene, count in sorted(
    scenes.items(),
    key=lambda x: x[1],
    reverse=True
):

    pct = percentage(
        count,
        processed_images
    )

    print(
        f"{scene:<35}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# KEY OBSERVATIONS
# ============================================================

most_common_action = max(
    actions,
    key=actions.get
)

most_common_risk = max(
    risks,
    key=risks.get
)

most_common_scene = max(
    scenes,
    key=scenes.get
)


print()
print("=" * 75)
print("KEY OBSERVATIONS")
print("-" * 75)

print(
    f"1. The complete multi-agent pipeline "
    f"processed {processed_images:,} BDD100K validation images."
)

print(
    f"2. The most frequent driving decision was "
    f"{most_common_action} "
    f"({actions[most_common_action]:,} images, "
    f"{percentage(actions[most_common_action], processed_images):.2f}%)."
)

print(
    f"3. The most frequent scene risk level was "
    f"{most_common_risk}."
)

print(
    f"4. The most common scene type was "
    f"'{most_common_scene}'."
)

print(
    f"5. The Safety Agent approved "
    f"{safety.get('APPROVE', 0):,} decisions and "
    f"overrode "
    f"{safety.get('OVERRIDE', 0):,} planner decisions."
)

print(
    f"6. Average processing time was "
    f"{average_time:.4f} seconds/image, "
    f"corresponding to approximately "
    f"{fps:.2f} FPS."
)


# ============================================================
# ARCHITECTURE
# ============================================================

print()
print("=" * 75)
print("MULTI-AGENT ARCHITECTURE")
print("-" * 75)

print(
"""
BDD100K IMAGE
      |
      v
+-----------------------+
|   PERCEPTION AGENT    |
|       YOLOv8n         |
|    Object Detection   |
+-----------+-----------+
            |
            v
+-----------------------+
| SCENE UNDERSTANDING   |
|        AGENT          |
| Scene + Traffic +     |
| Risk Analysis         |
+-----------+-----------+
            |
            v
+-----------------------+
|    PLANNING AGENT     |
|   GO / SLOW / STOP    |
+-----------+-----------+
            |
            v
+-----------------------+
|     SAFETY AGENT      |
| Approve / Override    |
+-----------+-----------+
            |
            v
+-----------------------+
| EXPLAINABILITY AGENT  |
| Decision Explanation  |
+-----------+-----------+
            |
            v
+-----------------------+
| FINAL DRIVING ACTION  |
|     GO / SLOW / STOP  |
+-----------------------+
"""
)


# ============================================================
# REPORT CONTENT
# ============================================================

report_lines = []

report_lines.append(
    "=" * 75
)

report_lines.append(
    "BDD100K MULTI-AGENT AUTONOMOUS DRIVING"
)

report_lines.append(
    "FINAL SYSTEM EVALUATION"
)

report_lines.append(
    "=" * 75
)

report_lines.append("")

report_lines.append("DATASET")

report_lines.append("-" * 75)

report_lines.append(
    f"Dataset                  : {dataset}"
)

report_lines.append(
    f"Split                    : {split}"
)

report_lines.append(
    f"Available images         : {available_images:,}"
)

report_lines.append(
    f"Images processed         : {processed_images:,}"
)

report_lines.append(
    f"Confidence threshold     : {confidence}"
)

report_lines.append("")

report_lines.append(
    "SYSTEM PERFORMANCE"
)

report_lines.append("-" * 75)

report_lines.append(
    f"Average objects/image    : {average_objects:.2f}"
)

report_lines.append(
    f"Average processing time  : {average_time:.4f} seconds"
)

report_lines.append(
    f"Approximate throughput   : {fps:.2f} FPS"
)

report_lines.append("")

report_lines.append(
    "FINAL DRIVING ACTION"
)

report_lines.append("-" * 75)

for action in [
    "GO",
    "SLOW",
    "STOP"
]:

    count = actions.get(
        action,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    report_lines.append(
        f"{action:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


report_lines.append("")

report_lines.append(
    "SCENE RISK"
)

report_lines.append("-" * 75)

for risk in [
    "LOW",
    "MODERATE",
    "HIGH"
]:

    count = risks.get(
        risk,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    report_lines.append(
        f"{risk:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


report_lines.append("")

report_lines.append(
    "SAFETY AGENT"
)

report_lines.append("-" * 75)

for decision in [
    "APPROVE",
    "OVERRIDE"
]:

    count = safety.get(
        decision,
        0
    )

    pct = percentage(
        count,
        processed_images
    )

    report_lines.append(
        f"{decision:<10}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


report_lines.append("")

report_lines.append(
    "SCENE TYPES"
)

report_lines.append("-" * 75)

for scene, count in sorted(
    scenes.items(),
    key=lambda x: x[1],
    reverse=True
):

    pct = percentage(
        count,
        processed_images
    )

    report_lines.append(
        f"{scene:<35}: "
        f"{count:>6} "
        f"({pct:6.2f}%)"
    )


report_lines.append("")

report_lines.append(
    "KEY OBSERVATIONS"
)

report_lines.append("-" * 75)

report_lines.append(
    f"1. The complete multi-agent pipeline processed "
    f"{processed_images:,} BDD100K validation images."
)

report_lines.append(
    f"2. The most frequent driving decision was "
    f"{most_common_action} "
    f"({actions[most_common_action]:,} images, "
    f"{percentage(actions[most_common_action], processed_images):.2f}%)."
)

report_lines.append(
    f"3. The most frequent scene risk level was "
    f"{most_common_risk}."
)

report_lines.append(
    f"4. The most common scene type was "
    f"'{most_common_scene}'."
)

report_lines.append(
    f"5. The Safety Agent approved "
    f"{safety.get('APPROVE', 0):,} decisions and "
    f"overrode "
    f"{safety.get('OVERRIDE', 0):,} planner decisions."
)

report_lines.append(
    f"6. Average processing time was "
    f"{average_time:.4f} seconds/image, "
    f"corresponding to approximately "
    f"{fps:.2f} FPS."
)

report_lines.append("")

report_lines.append(
    "MULTI-AGENT ARCHITECTURE"
)

report_lines.append("-" * 75)

report_lines.append(
"""
BDD100K IMAGE
      |
      v
PERCEPTION AGENT
(YOLOv8n)
      |
      v
SCENE UNDERSTANDING AGENT
      |
      v
PLANNING AGENT
(GO / SLOW / STOP)
      |
      v
SAFETY AGENT
(APPROVE / OVERRIDE)
      |
      v
EXPLAINABILITY AGENT
      |
      v
FINAL DRIVING ACTION
"""
)

report_lines.append(
    "=" * 75
)

report_lines.append(
    "END OF FINAL SYSTEM EVALUATION"
)

report_lines.append(
    "=" * 75
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


print()
print("=" * 75)
print("FINAL REPORT CREATED")
print("=" * 75)

print()
print("Saved to:")

print(
    REPORT_FILE
)

print("=" * 75)