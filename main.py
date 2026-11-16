from pathlib import Path
import json
import random
import time

from Agents.perception import PerceptionAgent
from Agents.scene_agent import SceneUnderstandingAgent
from Agents.planning_agent import PlanningAgent
from Agents.safety_agent import SafetyAgent
from Agents.explainability_agent import ExplainabilityAgent
from Agents.risk_fusion_agent import RiskFusionAgent
from Agents.trajectory_agent import TrajectoryPredictionAgent
from Agents.tracker_agent import TrackerAgent


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BDD100K_VAL = (
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
# INITIALIZE AGENTS
# ================================================================

print()
print("=" * 70)
print("MULTI-AGENT AUTONOMOUS DRIVING SYSTEM")
print("=" * 70)

print()
print("Initializing agents...")

perception_agent = PerceptionAgent(
    model_name="yolov8"
)

scene_agent = SceneUnderstandingAgent()

planning_agent = PlanningAgent()

safety_agent = SafetyAgent()

explainability_agent = ExplainabilityAgent()

tracker_agent = TrackerAgent()

trajectory_agent = TrajectoryPredictionAgent()

risk_fusion_agent = RiskFusionAgent()


# ================================================================
# FIND BDD100K IMAGE
# ================================================================

def get_validation_images():

    if not BDD100K_VAL.exists():

        raise FileNotFoundError(
            f"BDD100K validation folder not found:\n"
            f"{BDD100K_VAL}"
        )

    images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]:

        images.extend(
            BDD100K_VAL.glob(extension)
        )

    return sorted(images)


# ================================================================
# PROCESS ONE IMAGE
# ================================================================

def process_image(image_path):

    print()
    print("=" * 70)
    print("PROCESSING IMAGE")
    print("=" * 70)

    print(
        f"Image: {image_path.name}"
    )

    start_time = time.time()

    # ============================================================
    # 1. PERCEPTION
    # ============================================================

    print()
    print("[1/8] PERCEPTION AGENT")
    print("-" * 70)

    perception_result = perception_agent.run(
        str(image_path)
    )

    print(
        f"Model          : "
        f"{perception_result['model']}"
    )

    print(
        f"Objects detected: "
        f"{perception_result['num_objects']}"
    )

    print(
        f"Annotated image : "
        f"{perception_result['annotated_image']}"
    )

    # ============================================================
    # 2. SCENE UNDERSTANDING
    # ============================================================

    print()
    print("[2/8] SCENE UNDERSTANDING AGENT")
    print("-" * 70)

    scene_result = scene_agent.analyze(
        perception_result
    )

    print(
        f"Scene type      : "
        f"{scene_result['scene_type']}"
    )

    print(
        f"Traffic density : "
        f"{scene_result['traffic_density']}"
    )

    print(
        f"Vehicle count   : "
        f"{scene_result['vehicle_count']}"
    )

    print(
        f"Vulnerable users: "
        f"{scene_result['vulnerable_users']}"
    )

    print(
        f"Traffic controls: "
        f"{scene_result['traffic_controls']}"
    )

    print(
        f"Scene risk      : "
        f"{scene_result['risk_level']}"
    )

    # ============================================================
    # 3. TRACKER
    #
    # IMPORTANT:
    # This image pipeline has only one frame.
    # Therefore there is no genuine temporal velocity.
    #
    # We still pass detections through the tracker so the
    # architecture is ready for future video integration.
    # ============================================================

    print()
    print("[3/8] TRACKER AGENT")
    print("-" * 70)

    tracker_agent.reset()

    tracks = tracker_agent.update(
        perception_result["detections"],
        fps=1.0
    )

    print(
        f"Tracked objects : "
        f"{len(tracks)}"
    )

    print(
        "Temporal status  : "
        "Single-frame mode"
    )

    # ============================================================
    # 4. TRAJECTORY PREDICTION
    #
    # Since this is a single image, velocity is zero.
    # Therefore predicted position remains unchanged.
    #
    # Genuine trajectory prediction requires consecutive frames.
    # ============================================================

    print()
    print("[4/8] TRAJECTORY PREDICTION AGENT")
    print("-" * 70)

    trajectory_predictions = trajectory_agent.predict(
        tracks
    )

    print(
        f"Predictions generated: "
        f"{len(trajectory_predictions)}"
    )

    print(
        "Temporal prediction : "
        "Single-frame baseline"
    )

    # ============================================================
    # 5. PLANNING
    # ============================================================

    print()
    print("[5/8] PLANNING AGENT")
    print("-" * 70)

    planning_result = planning_agent.plan(
        scene_result
    )

    print(
        f"Proposed action : "
        f"{planning_result['action']}"
    )

    print(
        f"Speed level     : "
        f"{planning_result['speed_level']}"
    )

    print(
        f"Priority        : "
        f"{planning_result['priority']}"
    )

    print(
        f"Reason          : "
        f"{planning_result['reason']}"
    )

    # ============================================================
    # 6. SAFETY
    # ============================================================

    print()
    print("[6/8] SAFETY AGENT")
    print("-" * 70)

    safety_result = safety_agent.verify(
        scene_result,
        planning_result
    )

    print(
        f"Safety decision : "
        f"{safety_result['decision']}"
    )

    print(
        f"Final safety act: "
        f"{safety_result['final_action']}"
    )

    print(
        f"Safety level    : "
        f"{safety_result['safety_level']}"
    )

    # ============================================================
    # 7. RISK FUSION
    # ============================================================

    print()
    print("[7/8] CONFIDENCE-AWARE RISK FUSION")
    print("-" * 70)

    risk_result = risk_fusion_agent.fuse(

        perception_result=perception_result,

        scene_result=scene_result,

        planning_result=planning_result,

        safety_result=safety_result,

        trajectory_predictions=
            trajectory_predictions,

        image_width=1280,

        image_height=720
    )

    risk_fusion_agent.display(
        risk_result
    )

    # ============================================================
    # 8. EXPLAINABILITY
    # ============================================================

    print()
    print("[8/8] EXPLAINABILITY AGENT")
    print("-" * 70)
    explanation_result = (
    explainability_agent.explain(
        perception_result,
        scene_result,
        planning_result,
        safety_result,
        risk_fusion_result=risk_result,
        trajectory_predictions=trajectory_predictions
    )
)
    

    print(
        f"Explanation:"
    )

    print(
        explanation_result[
            "explanation"
        ]
    )

    # ============================================================
    # FINAL DECISION
    # ============================================================

    processing_time = (
        time.time()
        - start_time
    )

    final_action = (
        risk_result[
            "final_action"
        ]
    )

    print()
    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    print(
        f"Action          : "
        f"{final_action}"
    )

    print(
        f"Risk score      : "
        f"{risk_result['risk_score']:.4f}"
    )

    print(
        f"Risk percentage : "
        f"{risk_result['risk_percentage']:.2f}%"
    )

    print(
        f"Agent conflict  : "
        f"{risk_result['agent_conflict']}"
    )

    print(
        f"Processing time : "
        f"{processing_time:.3f} seconds"
    )

    print("=" * 70)

    # ============================================================
    # COMPLETE RESULT
    # ============================================================

    result = {

        "image":
            image_path.name,

        "image_path":
            str(image_path),

        "perception":
            perception_result,

        "scene":
            scene_result,

        "tracking":
            {
                "num_tracks":
                    len(tracks),

                "tracks":
                    tracks,

                "mode":
                    "single_frame"
            },

        "trajectory":
            {
                "mode":
                    "single_frame",

                "predictions":
                    trajectory_predictions
            },

        "planning":
            planning_result,

        "safety":
            safety_result,

        "risk_fusion":
            risk_result,

        "explainability":
            explanation_result,

        "final_action":
            final_action,

        "processing_time_seconds":
            round(
                processing_time,
                4
            )
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    output_file = (
        RESULTS_DIR
        / f"{image_path.stem}_enhanced_result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            default=str
        )

    print()
    print(
        f"Complete result saved to:"
    )

    print(
        output_file
    )

    return result


# ================================================================
# MAIN
# ================================================================

def main():

    images = get_validation_images()

    print()
    print(
        f"BDD100K validation images found: "
        f"{len(images)}"
    )

    if len(images) == 0:

        raise RuntimeError(
            "No validation images found."
        )

    # ------------------------------------------------------------
    # Use a fixed image for reproducibility.
    # ------------------------------------------------------------

    preferred_name = (
        "b1c66a42-6f7d68ca.jpg"
    )

    preferred_image = (
        BDD100K_VAL
        / preferred_name
    )

    if preferred_image.exists():

        image_path = preferred_image

    else:

        image_path = images[0]

    print()
    print(
        f"Selected image: "
        f"{image_path.name}"
    )

    result = process_image(
        image_path
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Architecture:")
    print()
    print(
        "BDD100K Image"
    )

    print(
        "    ↓"
    )

    print(
        "YOLOv8 Perception"
    )

    print(
        "    ↓"
    )

    print(
        "Tracker"
    )

    print(
        "    ↓"
    )

    print(
        "Trajectory Prediction"
    )

    print(
        "    ↓"
    )

    print(
        "Scene Understanding"
    )

    print(
        "    ↓"
    )

    print(
        "Planning + Safety"
    )

    print(
        "    ↓"
    )

    print(
        "Confidence-Aware Risk Fusion"
    )

    print(
        "    ↓"
    )

    print(
        "Explainability"
    )

    print(
        "    ↓"
    )

    print(
        result["final_action"]
    )

    print()
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()