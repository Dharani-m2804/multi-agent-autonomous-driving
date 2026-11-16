from pathlib import Path
import sys
import json
import time

# ================================================================
# PROJECT PATHS
# ================================================================

# Automatically detect the project root.
# This makes the code work on any laptop without hard-coded paths.

PROJECT_ROOT = Path(__file__).resolve().parent
AGENTS_DIR = PROJECT_ROOT / "Agents"

sys.path.insert(0, str(AGENTS_DIR))


# ================================================================
# IMPORT AGENTS
# ================================================================

from perception import PerceptionAgent
from scene_agent import SceneUnderstandingAgent
from planning_agent import PlanningAgent
from safety_agent import SafetyAgent
from explainability_agent import ExplainabilityAgent


# ================================================================
# BDD100K DATASET
# ================================================================

# Dataset location relative to the project folder.
BDD100K_VAL = (
    PROJECT_ROOT
    / "Dataset"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "val"
)


# ================================================================
# MULTI-AGENT SYSTEM
# ================================================================

class AutonomousDrivingSystem:

    def __init__(self, confidence=0.25):

        print()
        print("=" * 70)
        print("MULTI-AGENT AUTONOMOUS DRIVING SYSTEM")
        print("=" * 70)

        print()
        print("Initializing agents...")

        # --------------------------------------------------------
        # AGENT 1 — PERCEPTION
        # --------------------------------------------------------

        self.perception = PerceptionAgent(
            confidence=confidence
        )

        # --------------------------------------------------------
        # AGENT 2 — SCENE UNDERSTANDING
        # --------------------------------------------------------

        self.scene = SceneUnderstandingAgent()

        # --------------------------------------------------------
        # AGENT 3 — PLANNING
        # --------------------------------------------------------

        self.planning = PlanningAgent()

        # --------------------------------------------------------
        # AGENT 4 — SAFETY
        # --------------------------------------------------------

        self.safety = SafetyAgent()

        # --------------------------------------------------------
        # AGENT 5 — EXPLAINABILITY
        # --------------------------------------------------------

        self.explainability = ExplainabilityAgent()

        print()
        print("All agents initialized successfully.")

        print("=" * 70)


    # ============================================================
    # PROCESS ONE IMAGE
    # ============================================================

    def process_image(self, image_path):

        image_path = Path(image_path)

        start_time = time.time()

        print()
        print("=" * 70)
        print("PROCESSING BDD100K IMAGE")
        print("=" * 70)

        print()
        print(f"Image: {image_path.name}")

        # --------------------------------------------------------
        # AGENT 1 — PERCEPTION
        # --------------------------------------------------------

        print()
        print(">>> AGENT 1: PERCEPTION")

        perception_result = self.perception.run(
            image_path
        )

        # --------------------------------------------------------
        # AGENT 2 — SCENE UNDERSTANDING
        # --------------------------------------------------------

        print()
        print(">>> AGENT 2: SCENE UNDERSTANDING")

        scene_result = self.scene.analyze(
            perception_result
        )

        # --------------------------------------------------------
        # AGENT 3 — PLANNING
        # --------------------------------------------------------

        print()
        print(">>> AGENT 3: PLANNING")

        planning_result = self.planning.plan(
            scene_result
        )

        # --------------------------------------------------------
        # AGENT 4 — SAFETY
        # --------------------------------------------------------

        print()
        print(">>> AGENT 4: SAFETY")

        safety_result = self.safety.verify(
            scene_result,
            planning_result
        )

        # --------------------------------------------------------
        # AGENT 5 — EXPLAINABILITY
        # --------------------------------------------------------

        print()
        print(">>> AGENT 5: EXPLAINABILITY")

        explanation_result = self.explainability.explain(
            perception_result,
            scene_result,
            planning_result,
            safety_result
        )

        elapsed = time.time() - start_time

        # --------------------------------------------------------
        # FINAL SYSTEM RESULT
        # --------------------------------------------------------

        result = {

            "image": image_path.name,

            "perception": {
                "num_objects":
                    perception_result["num_objects"],

                "summary":
                    perception_result["summary"]
            },

            "scene": scene_result,

            "planning": planning_result,

            "safety": safety_result,

            "explainability": explanation_result,

            "final_action":
                safety_result["final_action"],

            "processing_time_seconds":
                round(elapsed, 3)
        }

        return result


    # ============================================================
    # DISPLAY FINAL RESULT
    # ============================================================

    def display_result(self, result):

        print()
        print("=" * 70)
        print("FINAL AUTONOMOUS DRIVING DECISION")
        print("=" * 70)

        print()

        print(
            f"Image              : "
            f"{result['image']}"
        )

        print(
            f"Objects detected   : "
            f"{result['perception']['num_objects']}"
        )

        print(
            f"Scene              : "
            f"{result['scene']['scene_type']}"
        )

        print(
            f"Traffic density    : "
            f"{result['scene']['traffic_density']}"
        )

        print(
            f"Risk level         : "
            f"{result['scene']['risk_level']}"
        )

        print(
            f"Planner action     : "
            f"{result['planning']['action']}"
        )

        print(
            f"Safety decision    : "
            f"{result['safety']['decision']}"
        )

        print(
            f"Safety level       : "
            f"{result['safety']['safety_level']}"
        )

        print()
        print("-" * 70)

        print(
            f"FINAL DRIVING ACTION: "
            f"{result['final_action']}"
        )

        print("-" * 70)

        print()

        print("EXPLANATION")

        print(
            result["explainability"]["explanation"]
        )

        print()

        print(
            f"Processing time: "
            f"{result['processing_time_seconds']} seconds"
        )

        print("=" * 70)


    # ============================================================
    # SAVE RESULT
    # ============================================================

    def save_result(self, result):

        output_dir = PROJECT_ROOT / "results"

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir
            / f"{Path(result['image']).stem}_result.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )

        return output_file


# =================================================================
# MAIN
# =================================================================

def main():

    print()
    print("=" * 70)
    print("BDD100K MULTI-AGENT AUTONOMOUS DRIVING")
    print("=" * 70)

    # -------------------------------------------------------------
    # DISPLAY PROJECT PATH
    # -------------------------------------------------------------

    print()
    print(f"Project root:")
    print(PROJECT_ROOT)

    # -------------------------------------------------------------
    # VERIFY DATASET
    # -------------------------------------------------------------

    print()
    print("Checking BDD100K validation dataset...")

    if not BDD100K_VAL.exists():

        raise FileNotFoundError(
            "\nBDD100K validation directory not found:\n"
            f"{BDD100K_VAL}\n"
        )

    # -------------------------------------------------------------
    # FIND IMAGES
    # -------------------------------------------------------------

    images = sorted(
        BDD100K_VAL.glob("*.jpg")
    )

    print()
    print(
        f"BDD100K validation images available: "
        f"{len(images):,}"
    )

    if not images:

        raise FileNotFoundError(
            "\nNo .jpg images found in:\n"
            f"{BDD100K_VAL}\n"
        )

    # -------------------------------------------------------------
    # SELECT FIRST IMAGE
    # -------------------------------------------------------------

    image_path = images[0]

    print()
    print(
        f"Selected image: "
        f"{image_path.name}"
    )

    # -------------------------------------------------------------
    # INITIALIZE SYSTEM
    # -------------------------------------------------------------

    system = AutonomousDrivingSystem(
        confidence=0.25
    )

    # -------------------------------------------------------------
    # PROCESS IMAGE
    # -------------------------------------------------------------

    result = system.process_image(
        image_path
    )

    # -------------------------------------------------------------
    # DISPLAY FINAL RESULT
    # -------------------------------------------------------------

    system.display_result(
        result
    )

    # -------------------------------------------------------------
    # SAVE RESULT
    # -------------------------------------------------------------

    output_file = system.save_result(
        result
    )

    print()
    print(
        f"Result saved to:"
    )

    print(output_file)

    # -------------------------------------------------------------
    # COMPLETION
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("SYSTEM EXECUTION COMPLETE")
    print("=" * 70)


# =================================================================
# PROGRAM ENTRY POINT
# =================================================================

if __name__ == "__main__":
    main()