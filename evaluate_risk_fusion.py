from pathlib import Path
import json
import time
from collections import Counter

from Agents.perception import PerceptionAgent
from Agents.scene_agent import SceneUnderstandingAgent
from Agents.planning_agent import PlanningAgent
from Agents.safety_agent import SafetyAgent
from Agents.tracker_agent import TrackerAgent
from Agents.trajectory_agent import TrajectoryPredictionAgent
from Agents.risk_fusion_agent import RiskFusionAgent


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VAL_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "val"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# CONFIGURATION
# ================================================================

MAX_IMAGES = 10000

CONFIDENCE = 0.25

IMAGE_WIDTH = 1280

IMAGE_HEIGHT = 720


# ================================================================
# FIND IMAGES
# ================================================================

def get_images():

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    images = []

    for extension in extensions:

        images.extend(
            VAL_DIR.glob(extension)
        )

    return sorted(images)[:MAX_IMAGES]


# ================================================================
# INITIALIZE AGENTS
# ================================================================

def initialize_agents():

    print()
    print("=" * 70)
    print("INITIALIZING EVALUATION AGENTS")
    print("=" * 70)

    perception = PerceptionAgent(
        model_name="yolov8"
    )

    scene = SceneUnderstandingAgent()

    planning = PlanningAgent()

    safety = SafetyAgent()

    tracker = TrackerAgent()

    trajectory = TrajectoryPredictionAgent()

    fusion = RiskFusionAgent()

    print("=" * 70)

    return (
        perception,
        scene,
        planning,
        safety,
        tracker,
        trajectory,
        fusion
    )


# ================================================================
# BASELINE PIPELINE
# ================================================================

def run_baseline(
    perception_result,
    scene_agent,
    planning_agent,
    safety_agent
):

    scene_result = scene_agent.analyze(
        perception_result
    )

    planning_result = planning_agent.plan(
        scene_result
    )

    safety_result = safety_agent.verify(
        scene_result,
        planning_result
    )

    final_action = safety_result[
        "final_action"
    ]

    return {

        "scene":
            scene_result,

        "planning":
            planning_result,

        "safety":
            safety_result,

        "final_action":
            final_action
    }


# ================================================================
# PROPOSED PIPELINE
# ================================================================

def run_proposed(
    perception_result,
    scene_agent,
    planning_agent,
    safety_agent,
    tracker_agent,
    trajectory_agent,
    fusion_agent
):

    scene_result = scene_agent.analyze(
        perception_result
    )

    planning_result = planning_agent.plan(
        scene_result
    )

    safety_result = safety_agent.verify(
        scene_result,
        planning_result
    )

    # ------------------------------------------------------------
    # Single-frame tracking
    # ------------------------------------------------------------

    tracker_agent.reset()

    tracks = tracker_agent.update(
        perception_result[
            "detections"
        ],
        fps=1.0
    )

    # ------------------------------------------------------------
    # Single-frame trajectory baseline
    # ------------------------------------------------------------

    trajectory_predictions = (
        trajectory_agent.predict(
            tracks
        )
    )

    # ------------------------------------------------------------
    # Risk fusion
    # ------------------------------------------------------------

    fusion_result = fusion_agent.fuse(

        perception_result=
            perception_result,

        scene_result=
            scene_result,

        planning_result=
            planning_result,

        safety_result=
            safety_result,

        trajectory_predictions=
            trajectory_predictions,

        image_width=
            IMAGE_WIDTH,

        image_height=
            IMAGE_HEIGHT
    )

    return {

        "scene":
            scene_result,

        "planning":
            planning_result,

        "safety":
            safety_result,

        "tracking":
            tracks,

        "trajectory":
            trajectory_predictions,

        "risk_fusion":
            fusion_result,

        "final_action":
            fusion_result[
                "final_action"
            ]
    }


# ================================================================
# MAIN EVALUATION
# ================================================================

def main():

    print()
    print("=" * 70)
    print("BDD100K RISK-FUSION ABLATION EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    images = get_images()

    if not images:

        raise FileNotFoundError(
            f"No validation images found:\n{VAL_DIR}"
        )

    print()
    print(
        f"Validation images available: "
        f"{len(images)}"
    )

    # ------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------

    (
        perception_agent,
        scene_agent,
        planning_agent,
        safety_agent,
        tracker_agent,
        trajectory_agent,
        fusion_agent
    ) = initialize_agents()

    # ------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------

    baseline_actions = Counter()

    proposed_actions = Counter()

    baseline_risks = Counter()

    proposed_risks = Counter()

    scene_counter = Counter()

    safety_decisions = Counter()

    conflicts = 0

    total_objects = 0

    baseline_time = 0.0

    proposed_time = 0.0

    processed = 0

    errors = []

    # ------------------------------------------------------------
    # Detailed records
    # ------------------------------------------------------------

    baseline_records = []

    proposed_records = []

    # ============================================================
    # PROCESS DATASET
    # ============================================================

    overall_start = time.time()

    for index, image_path in enumerate(images, start=1):

        try:

            # ----------------------------------------------------
            # PERCEPTION
            # ----------------------------------------------------

            perception_start = time.time()

            perception_result = (
                perception_agent.run(
                    str(image_path)
                )
            )

            perception_time = (
                time.time()
                - perception_start
            )

            total_objects += (
                perception_result[
                    "num_objects"
                ]
            )

            # ====================================================
            # BASELINE
            # ====================================================

            baseline_start = time.time()

            baseline = run_baseline(

                perception_result,

                scene_agent,

                planning_agent,

                safety_agent
            )

            baseline_elapsed = (
                time.time()
                - baseline_start
            )

            baseline_time += (
                baseline_elapsed
            )

            baseline_action = (
                baseline[
                    "final_action"
                ]
            )

            baseline_actions[
                baseline_action
            ] += 1

            scene_counter[
                baseline[
                    "scene"
                ][
                    "scene_type"
                ]
            ] += 1

            safety_decisions[
                baseline[
                    "safety"
                ][
                    "decision"
                ]
            ] += 1

            baseline_records.append(
                {

                    "image":
                        image_path.name,

                    "action":
                        baseline_action,

                    "scene":
                        baseline[
                            "scene"
                        ][
                            "scene_type"
                        ],

                    "risk":
                        baseline[
                            "scene"
                        ][
                            "risk_level"
                        ]
                }
            )

            # ====================================================
            # PROPOSED
            # ====================================================

            proposed_start = time.time()

            proposed = run_proposed(

                perception_result,

                scene_agent,

                planning_agent,

                safety_agent,

                tracker_agent,

                trajectory_agent,

                fusion_agent
            )

            proposed_elapsed = (
                time.time()
                - proposed_start
            )

            proposed_time += (
                proposed_elapsed
            )

            proposed_action = (
                proposed[
                    "final_action"
                ]
            )

            proposed_actions[
                proposed_action
            ] += 1

            # ----------------------------------------------------
            # Risk level
            # ----------------------------------------------------

            risk_score = (
                proposed[
                    "risk_fusion"
                ][
                    "risk_score"
                ]
            )

            if risk_score >= 0.70:

                risk_level = "HIGH"

            elif risk_score >= 0.30:

                risk_level = "MODERATE"

            else:

                risk_level = "LOW"

            proposed_risks[
                risk_level
            ] += 1

            # ----------------------------------------------------
            # Agent conflict
            # ----------------------------------------------------

            if proposed[
                "risk_fusion"
            ][
                "agent_conflict"
            ]:

                conflicts += 1

            # ----------------------------------------------------
            # Detailed record
            # ----------------------------------------------------

            proposed_records.append(
                {

                    "image":
                        image_path.name,

                    "action":
                        proposed_action,

                    "risk_score":
                        risk_score,

                    "risk_level":
                        risk_level,

                    "agent_conflict":
                        proposed[
                            "risk_fusion"
                        ][
                            "agent_conflict"
                        ],

                    "scene":
                        proposed[
                            "scene"
                        ][
                            "scene_type"
                        ],

                    "objects":
                        perception_result[
                            "num_objects"
                        ]
                }
            )

            processed += 1

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                index == 1
                or
                index % 100 == 0
                or
                index == len(images)
            ):

                elapsed = (
                    time.time()
                    - overall_start
                )

                fps = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[{index}/{len(images)}] "
                    f"{image_path.name} | "
                    f"Baseline={baseline_action} | "
                    f"Proposed={proposed_action} | "
                    f"Risk={risk_score:.2f} | "
                    f"FPS={fps:.2f}"
                )

        except Exception as error:

            errors.append(
                {
                    "image":
                        image_path.name,

                    "error":
                        str(error)
                }
            )

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )

    # ============================================================
    # FINAL STATISTICS
    # ============================================================

    total_time = (
        time.time()
        - overall_start
    )

    avg_objects = (
        total_objects / processed
        if processed > 0
        else 0
    )

    baseline_fps = (
        processed / baseline_time
        if baseline_time > 0
        else 0
    )

    proposed_fps = (
        processed / proposed_time
        if proposed_time > 0
        else 0
    )

    overall_fps = (
        processed / total_time
        if total_time > 0
        else 0
    )

    # ============================================================
    # REPORT
    # ============================================================

    report = {

        "dataset":
            "BDD100K",

        "split":
            "validation",

        "images_requested":
            len(images),

        "images_processed":
            processed,

        "errors":
            len(errors),

        "average_objects_per_image":
            round(
                avg_objects,
                4
            ),

        "baseline":
            {

                "actions":
                    dict(
                        baseline_actions
                    ),

                "processing_time_seconds":
                    round(
                        baseline_time,
                        4
                    ),

                "average_time_seconds":
                    round(
                        baseline_time / processed,
                        6
                    )
                    if processed
                    else 0,

                "fps":
                    round(
                        baseline_fps,
                        4
                    )
            },

        "proposed_risk_fusion":
            {

                "actions":
                    dict(
                        proposed_actions
                    ),

                "risk_levels":
                    dict(
                        proposed_risks
                    ),

                "agent_conflicts":
                    conflicts,

                "processing_time_seconds":
                    round(
                        proposed_time,
                        4
                    ),

                "average_time_seconds":
                    round(
                        proposed_time / processed,
                        6
                    )
                    if processed
                    else 0,

                "fps":
                    round(
                        proposed_fps,
                        4
                    )
            },

        "scene_distribution":
            dict(
                scene_counter
            ),

        "safety_decisions":
            dict(
                safety_decisions
            ),

        "overall":
            {

                "processing_time_seconds":
                    round(
                        total_time,
                        4
                    ),

                "fps":
                    round(
                        overall_fps,
                        4
                    )
            },

        "errors_detail":
            errors
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    output_json = (
        RESULTS_DIR
        / "risk_fusion_ablation.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ============================================================
    # SAVE RECORDS
    # ============================================================

    records_json = (
        RESULTS_DIR
        / "risk_fusion_image_records.json"
    )

    with open(
        records_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {

                "baseline":
                    baseline_records,

                "proposed":
                    proposed_records
            },

            file,

            indent=4
        )

    # ============================================================
    # PRINT FINAL REPORT
    # ============================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Images processed       : "
        f"{processed}"
    )

    print(
        f"Errors                 : "
        f"{len(errors)}"
    )

    print(
        f"Average objects/image  : "
        f"{avg_objects:.2f}"
    )

    print()
    print("BASELINE")
    print("-" * 70)

    for action, count in (
        baseline_actions.items()
    ):

        percentage = (
            count / processed * 100
            if processed
            else 0
        )

        print(
            f"{action:<10}: "
            f"{count:>6} "
            f"({percentage:>6.2f}%)"
        )

    print(
        f"Average time: "
        f"{baseline_time / processed:.4f}s"
        if processed
        else "Average time: 0"
    )

    print(
        f"FPS         : "
        f"{baseline_fps:.2f}"
    )

    print()
    print("PROPOSED RISK FUSION")
    print("-" * 70)

    for action, count in (
        proposed_actions.items()
    ):

        percentage = (
            count / processed * 100
            if processed
            else 0
        )

        print(
            f"{action:<10}: "
            f"{count:>6} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print("Risk levels:")

    for level, count in (
        proposed_risks.items()
    ):

        percentage = (
            count / processed * 100
            if processed
            else 0
        )

        print(
            f"{level:<10}: "
            f"{count:>6} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print(
        f"Agent conflicts       : "
        f"{conflicts}"
    )

    print(
        f"Average time          : "
        f"{proposed_time / processed:.4f}s"
        if processed
        else "Average time: 0"
    )

    print(
        f"FPS                   : "
        f"{proposed_fps:.2f}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        output_json
    )

    print(
        records_json
    )

    print()
    print("=" * 70)
    print("RISK-FUSION ABLATION FINISHED")
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()