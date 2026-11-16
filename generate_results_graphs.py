import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"

# FINAL 20K TEST RESULTS
JSON_FILE = RESULTS_DIR / "batch_results_20000.json"


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("GENERATING FINAL PROJECT RESULT GRAPHS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nResults file:")
print(JSON_FILE)

if not JSON_FILE.exists():
    raise FileNotFoundError(
        f"\nResults file not found:\n{JSON_FILE}\n\n"
        "Make sure the 20,000-image final test evaluation "
        "has been completed."
    )

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nLoaded results successfully.")


# ============================================================
# EXTRACT DATA
# ============================================================

action_distribution = data["action_distribution"]
risk_distribution = data["risk_distribution"]
safety_distribution = data["safety_distribution"]
scene_distribution = data["scene_distribution"]

processed_images = data["processed_images"]

if "average_objects_per_image" in data:
    average_objects = data["average_objects_per_image"]
else:
    average_objects = data["average_objects"]

if "average_processing_time_seconds" in data:
    average_time = data["average_processing_time_seconds"]
else:
    average_time = data["average_processing_time"]

fps = data["approximate_fps"]


# ============================================================
# PRINT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL 20K TEST DATASET INFORMATION")
print("=" * 70)

print(f"Images processed       : {processed_images:,}")
print(f"Average objects/image  : {average_objects:.2f}")
print(f"Average processing time: {average_time:.4f} seconds")
print(f"Approximate FPS        : {fps:.2f}")


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_bar_chart(
    values,
    title,
    xlabel,
    ylabel,
    filename,
    rotate_labels=False
):

    plt.figure(figsize=(10, 6))

    labels = list(values.keys())
    numbers = list(values.values())

    bars = plt.bar(
        range(len(labels)),
        numbers
    )

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        xlabel,
        fontsize=12
    )

    plt.ylabel(
        ylabel,
        fontsize=12
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45 if rotate_labels else 0,
        ha="right" if rotate_labels else "center"
    )

    for bar, value in zip(
        bars,
        numbers
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    output = RESULTS_DIR / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output}")


# ============================================================
# 1. ACTION DISTRIBUTION
# ============================================================

print("\nGenerating action distribution...")

save_bar_chart(
    action_distribution,
    "Final Driving Action Distribution - BDD100K 20K Test",
    "Driving Action",
    "Number of Images",
    "action_distribution.png"
)


# ============================================================
# 2. RISK DISTRIBUTION
# ============================================================

print("\nGenerating risk distribution...")

save_bar_chart(
    risk_distribution,
    "Scene Risk Distribution - BDD100K 20K Test",
    "Risk Level",
    "Number of Images",
    "risk_distribution.png"
)


# ============================================================
# 3. SAFETY DISTRIBUTION
# ============================================================

print("\nGenerating safety distribution...")

save_bar_chart(
    safety_distribution,
    "Safety Agent Decision Distribution - BDD100K 20K Test",
    "Safety Decision",
    "Number of Images",
    "safety_distribution.png"
)


# ============================================================
# 4. SCENE DISTRIBUTION
# ============================================================

print("\nGenerating scene distribution...")

save_bar_chart(
    scene_distribution,
    "Scene Type Distribution - BDD100K 20K Test",
    "Scene Type",
    "Number of Images",
    "scene_distribution.png",
    rotate_labels=True
)


# ============================================================
# 5. PERFORMANCE SUMMARY
# ============================================================

print("\nGenerating performance summary...")

metrics = {

    "Images\nProcessed":
        processed_images,

    "Avg Objects\nper Image":
        average_objects,

    "FPS":
        fps,

}

plt.figure(
    figsize=(10, 6)
)

labels = list(
    metrics.keys()
)

values = list(
    metrics.values()
)

bars = plt.bar(
    range(len(labels)),
    values
)

plt.title(
    "Multi-Agent Autonomous Driving System Performance - 20K Test",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel(
    "Value",
    fontsize=12
)

plt.xticks(
    range(len(labels)),
    labels
)

for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

output = (
    RESULTS_DIR /
    "performance_summary.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Created: {output}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL 20K TEST GRAPH GENERATION COMPLETE")
print("=" * 70)

print("\n20K TEST RESULTS USED:")
print(
    f"  {JSON_FILE}"
)

print("\nGenerated files:")

for file in [

    "action_distribution.png",
    "risk_distribution.png",
    "safety_distribution.png",
    "scene_distribution.png",
    "performance_summary.png",

]:

    print(
        f"  results\\{file}"
    )

print("\n" + "=" * 70)