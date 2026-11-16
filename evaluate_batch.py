from pathlib import Path
import sys
import json
import csv
import time
from collections import Counter


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
AGENTS_DIR = PROJECT_ROOT / "Agents"

sys.path.insert(0, str(AGENTS_DIR))


# ================================================================
# IMPORT ACTUAL AGENTS
# ================================================================

from perception import PerceptionAgent
from scene_agent import SceneUnderstandingAgent
from planning_agent import PlanningAgent
from safety_agent import SafetyAgent
from explainability_agent import ExplainabilityAgent


# ================================================================
# FIND FULL 20K BDD100K TEST DATASET
# ================================================================

POSSIBLE_DATASET_PATHS = [

    # Current laptop
    PROJECT_ROOT
    / "Dataset"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "test",

    # Previous laptop structure
    PROJECT_ROOT
    / "Dataset"
    / "BDD100K_Extracted"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "test",

]


def find_test_dataset():

    for path in POSSIBLE_DATASET_PATHS:

        if path.exists():

            images = list(
                path.glob("*.jpg")
            )

            if len(images) > 0:

                return path

    raise FileNotFoundError(
        "\nBDD100K TEST dataset was not found.\n\n"
        "Expected one of:\n"
        + "\n".join(
            str(p)
            for p in POSSIBLE_DATASET_PATHS
        )
    )


BDD100K_TEST = find_test_dataset()


# ================================================================
# FIND TRAINED YOLO MODEL
# ================================================================

MODEL_SEARCH_PATHS = [

    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "bdd100k_yolov8n-2"
    / "weights"
    / "best.pt",

    PROJECT_ROOT
    / "runs"
    / "bdd100k_yolov8n"
    / "weights"
    / "best.pt",

]


def find_model():

    for path in MODEL_SEARCH_PATHS:

        if path.exists():

            return path

    models = list(
        (PROJECT_ROOT / "runs").rglob(
            "best.pt"
        )
    )

    if models:

        models.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return models[0]

    raise FileNotFoundError(
        "\nNo YOLO best.pt model found inside:\n"
        f"{PROJECT_ROOT / 'runs'}"
    )


MODEL_PATH = find_model()


# ================================================================
# CONFIGURATION
# ================================================================

# None = entire 20K test dataset
NUM_IMAGES = None

CONFIDENCE = 0.25


# ================================================================
# INITIALIZE AGENTS
# ================================================================

def initialize_agents():

    print()
    print("=" * 70)
    print("INITIALIZING MULTI-AGENT SYSTEM")
    print("=" * 70)

    print()
    print("YOLO model:")
    print(MODEL_PATH)

    print()
    print("BDD100K TEST dataset:")
    print(BDD100K_TEST)

    print()
    print("Confidence threshold:")
    print(CONFIDENCE)

    # ------------------------------------------------------------
    # PERCEPTION AGENT
    # ------------------------------------------------------------

    perception = PerceptionAgent(
        confidence=CONFIDENCE,
        model_path=str(MODEL_PATH)
    )

    # ------------------------------------------------------------
    # OTHER AGENTS
    # ------------------------------------------------------------

    scene = SceneUnderstandingAgent()

    planning = PlanningAgent()

    safety = SafetyAgent()

    explainability = ExplainabilityAgent()

    print()
    print("All agents initialized successfully.")

    print("=" * 70)

    return (
        perception,
        scene,
        planning,
        safety,
        explainability
    )


# ================================================================
# PROCESS ONE IMAGE
# ================================================================

def process_image(
    image_path,
    perception,
    scene_agent,
    planning,
    safety,
    explainability
):

    start = time.perf_counter()

    # ------------------------------------------------------------
    # PERCEPTION
    # ------------------------------------------------------------

    perception_result = perception.run(
        image_path
    )

    # ------------------------------------------------------------
    # SCENE UNDERSTANDING
    # ------------------------------------------------------------

    scene_result = scene_agent.analyze(
        perception_result
    )

    # ------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------

    planning_result = planning.plan(
        scene_result
    )

    # ------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------

    safety_result = safety.verify(
        scene_result,
        planning_result
    )

    # ------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------

    explanation_result = explainability.explain(
        perception_result,
        scene_result,
        planning_result,
        safety_result
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {

        "image":
            image_path.name,

        "objects":
            perception_result.get(
                "num_objects",
                0
            ),

        "object_summary":
            perception_result.get(
                "summary",
                {}
            ),

        "scene_type":
            scene_result.get(
                "scene_type",
                "unknown"
            ),

        "traffic_density":
            scene_result.get(
                "traffic_density",
                "unknown"
            ),

        "risk_level":
            scene_result.get(
                "risk_level",
                "unknown"
            ),

        "vehicle_count":
            scene_result.get(
                "vehicle_count",
                0
            ),

        "vulnerable_users":
            scene_result.get(
                "vulnerable_users",
                0
            ),

        "traffic_controls":
            scene_result.get(
                "traffic_controls",
                0
            ),

        "planner_action":
            planning_result.get(
                "action",
                "UNKNOWN"
            ),

        "safety_decision":
            safety_result.get(
                "decision",
                "UNKNOWN"
            ),

        "safety_level":
            safety_result.get(
                "safety_level",
                "UNKNOWN"
            ),

        "final_action":
            safety_result.get(
                "final_action",
                "UNKNOWN"
            ),

        "processing_time":
            round(
                elapsed,
                4
            ),

        "explanation":
            explanation_result.get(
                "explanation",
                ""
            )
    }


# ================================================================
# SAVE JSON
# ================================================================

def save_json(
    results_dir,
    results,
    total_images,
    action_counter,
    risk_counter,
    scene_counter,
    safety_counter,
    average_objects,
    average_time,
    fps
):

    json_path = (
        results_dir /
        "batch_results_20000.json"
    )

    summary = {

        "project":
            "BDD100K Multi-Agent Autonomous Driving",

        "dataset":
            "BDD100K",

        "split":
            "test",

        "dataset_path":
            str(BDD100K_TEST),

        "model_path":
            str(MODEL_PATH),

        "available_images":
            total_images,

        "processed_images":
            len(results),

        "confidence":
            CONFIDENCE,

        "average_objects_per_image":
            round(
                average_objects,
                4
            ),

        "average_processing_time_seconds":
            round(
                average_time,
                4
            ),

        "approximate_fps":
            round(
                fps,
                4
            ),

        "action_distribution":
            dict(action_counter),

        "risk_distribution":
            dict(risk_counter),

        "safety_distribution":
            dict(safety_counter),

        "scene_distribution":
            dict(scene_counter),

        "results":
            results
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    return json_path


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    results_dir,
    results
):

    csv_path = (
        results_dir /
        "batch_results_20000.csv"
    )

    fieldnames = [

        "image",

        "objects",

        "scene_type",

        "traffic_density",

        "risk_level",

        "vehicle_count",

        "vulnerable_users",

        "traffic_controls",

        "planner_action",

        "safety_decision",

        "safety_level",

        "final_action",

        "processing_time"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({

                field: result.get(
                    field,
                    ""
                )

                for field in fieldnames
            })

    return csv_path


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print(
        "BDD100K MULTI-AGENT FINAL TEST EVALUATION"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    print()
    print("Dataset:")
    print("BDD100K")

    print()
    print("TEST directory:")
    print(BDD100K_TEST)

    images = sorted(
        BDD100K_TEST.glob("*.jpg")
    )

    total_images = len(
        images
    )

    print()
    print(
        f"TEST images available : "
        f"{total_images:,}"
    )

    if total_images == 0:

        raise RuntimeError(
            "No JPG images were found."
        )

    # ------------------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------------------

    if total_images != 20000:

        print()
        print(
            "WARNING: Expected 20,000 test images."
        )

        print(
            f"Found: {total_images:,}"
        )

        raise RuntimeError(
            "The selected TEST directory "
            "does not contain exactly 20,000 images."
        )

    # ------------------------------------------------------------
    # SELECT DATASET
    # ------------------------------------------------------------

    if NUM_IMAGES is None:

        selected_images = images

    else:

        selected_images = images[
            :min(
                NUM_IMAGES,
                total_images
            )
        ]

    num_images = len(
        selected_images
    )

    print()
    print(
        f"Images selected for evaluation : "
        f"{num_images:,}"
    )

    print()
    print(
        "FULL 20K TEST DATASET WILL BE PROCESSED."
    )

    # ------------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------------

    (
        perception,
        scene_agent,
        planning,
        safety,
        explainability
    ) = initialize_agents()

    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------

    results = []

    action_counter = Counter()

    risk_counter = Counter()

    scene_counter = Counter()

    safety_counter = Counter()

    total_objects = 0

    total_time = 0.0

    errors = []

    # ------------------------------------------------------------
    # PROCESS DATASET
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING 20K FINAL TEST PROCESSING")
    print("=" * 70)

    overall_start = time.perf_counter()

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        try:

            result = process_image(
                image_path,
                perception,
                scene_agent,
                planning,
                safety,
                explainability
            )

            results.append(
                result
            )

            # ----------------------------------------------------
            # STATISTICS
            # ----------------------------------------------------

            action_counter[
                result["final_action"]
            ] += 1

            risk_counter[
                result["risk_level"]
            ] += 1

            scene_counter[
                result["scene_type"]
            ] += 1

            safety_counter[
                result["safety_decision"]
            ] += 1

            total_objects += (
                result["objects"]
            )

            total_time += (
                result["processing_time"]
            )

            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            print(
                f"[{index:05d}/{num_images:05d}] "
                f"{image_path.name} | "
                f"objects={result['objects']} | "
                f"risk={result['risk_level']} | "
                f"action={result['final_action']} | "
                f"time={result['processing_time']:.3f}s"
            )

        except Exception as e:

            error_message = (
                f"{image_path.name}: {str(e)}"
            )

            errors.append(
                error_message
            )

            print(
                f"[{index:05d}/{num_images:05d}] "
                f"{image_path.name} | "
                f"ERROR: {e}"
            )

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    processed = len(
        results
    )

    failed = len(
        errors
    )

    if processed == 0:

        raise RuntimeError(
            "No images were successfully processed."
        )

    average_objects = (
        total_objects /
        processed
    )

    average_time = (
        total_time /
        processed
    )

    fps = (
        1.0 /
        average_time
        if average_time > 0
        else 0
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("20K FINAL TEST EVALUATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Dataset images available : "
        f"{total_images:,}"
    )

    print(
        f"Images requested         : "
        f"{num_images:,}"
    )

    print(
        f"Images processed         : "
        f"{processed:,}"
    )

    print(
        f"Images failed            : "
        f"{failed:,}"
    )

    print(
        f"Average objects/image    : "
        f"{average_objects:.2f}"
    )

    print(
        f"Average agent time/image : "
        f"{average_time:.4f} seconds"
    )

    print(
        f"Approximate agent FPS    : "
        f"{fps:.2f}"
    )

    print(
        f"Total evaluation time    : "
        f"{overall_elapsed / 60:.2f} minutes"
    )

    # ============================================================
    # ACTION DISTRIBUTION
    # ============================================================

    print()
    print("FINAL ACTION DISTRIBUTION")
    print("-" * 70)

    for action in [
        "GO",
        "SLOW",
        "STOP"
    ]:

        count = action_counter.get(
            action,
            0
        )

        percentage = (
            count /
            processed *
            100
        )

        print(
            f"{action:<10}: "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    # ============================================================
    # RISK DISTRIBUTION
    # ============================================================

    print()
    print("RISK DISTRIBUTION")
    print("-" * 70)

    for risk in [
        "LOW",
        "MODERATE",
        "HIGH"
    ]:

        count = risk_counter.get(
            risk,
            0
        )

        percentage = (
            count /
            processed *
            100
        )

        print(
            f"{risk:<12}: "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    # ============================================================
    # SAFETY DISTRIBUTION
    # ============================================================

    print()
    print("SAFETY DECISIONS")
    print("-" * 70)

    for decision in [
        "APPROVE",
        "OVERRIDE"
    ]:

        count = safety_counter.get(
            decision,
            0
        )

        percentage = (
            count /
            processed *
            100
        )

        print(
            f"{decision:<12}: "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    # ============================================================
    # SCENE TYPES
    # ============================================================

    print()
    print("SCENE TYPES")
    print("-" * 70)

    for scene_type, count in sorted(
        scene_counter.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = (
            count /
            processed *
            100
        )

        print(
            f"{scene_type:<35}: "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    # ============================================================
    # ERROR LOG
    # ============================================================

    if errors:

        print()
        print("PROCESSING ERRORS")
        print("-" * 70)

        for error in errors[:20]:

            print(error)

        error_path = (
            PROJECT_ROOT
            / "results"
            / "batch_errors_20000.txt"
        )

        with open(
            error_path,
            "w",
            encoding="utf-8"
        ) as f:

            for error in errors:

                f.write(
                    error + "\n"
                )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_dir = (
        PROJECT_ROOT /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = save_json(
        results_dir,
        results,
        total_images,
        action_counter,
        risk_counter,
        scene_counter,
        safety_counter,
        average_objects,
        average_time,
        fps
    )

    csv_path = save_csv(
        results_dir,
        results
    )

    # ============================================================
    # COMPLETE
    # ============================================================

    print()
    print("=" * 70)
    print("20K FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("JSON results:")
    print(json_path)

    print()
    print("CSV results:")
    print(csv_path)

    print()
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()