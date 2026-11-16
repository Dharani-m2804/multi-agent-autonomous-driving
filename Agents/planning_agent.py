from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DrivingDecision:
    action: str
    speed_level: str
    reason: str
    priority: str


class PlanningAgent:
    """
    Planning Agent

    Converts scene information into a high-level driving action.

    Actions:
        GO
        SLOW
        STOP
    """

    def __init__(self):
        print()
        print("=" * 65)
        print("INITIALIZING PLANNING AGENT")
        print("=" * 65)
        print("Decision planning module loaded.")
        print("=" * 65)

    # =============================================================
    # DETERMINE ACTION
    # =============================================================

    def plan(self, scene: Dict) -> Dict:

        risk_level = scene.get(
            "risk_level",
            "LOW"
        ).upper()

        vulnerable_users = scene.get(
            "vulnerable_users",
            0
        )

        traffic_controls = scene.get(
            "traffic_controls",
            0
        )

        vehicle_count = scene.get(
            "vehicle_count",
            0
        )

        traffic_density = scene.get(
            "traffic_density",
            "unknown"
        ).lower()

        risk_indicators = scene.get(
            "risk_indicators",
            []
        )

        # ---------------------------------------------------------
        # PRIORITY 1: HIGH RISK
        # ---------------------------------------------------------

        if risk_level == "HIGH":

            action = "STOP"
            speed = "0 km/h"
            priority = "CRITICAL"

            reason = (
                "High-risk road situation detected. "
                "The vehicle should stop until the situation "
                "is confirmed safe."
            )

        # ---------------------------------------------------------
        # PRIORITY 2: VULNERABLE ROAD USERS
        # ---------------------------------------------------------

        elif vulnerable_users > 0:

            action = "SLOW"
            speed = "LOW"
            priority = "HIGH"

            reason = (
                f"{vulnerable_users} vulnerable road user(s) "
                "detected. Reduce speed and maintain increased "
                "caution."
            )

        # ---------------------------------------------------------
        # PRIORITY 3: TRAFFIC CONTROL
        # ---------------------------------------------------------

        elif traffic_controls > 0:

            action = "SLOW"
            speed = "LOW"
            priority = "HIGH"

            reason = (
                "Traffic control infrastructure detected. "
                "Reduce speed and prepare for a possible "
                "change in driving conditions."
            )

        # ---------------------------------------------------------
        # PRIORITY 4: HIGH TRAFFIC
        # ---------------------------------------------------------

        elif traffic_density == "high":

            action = "SLOW"
            speed = "MEDIUM"
            priority = "MEDIUM"

            reason = (
                f"High traffic density detected with "
                f"{vehicle_count} vehicles. "
                "Maintain a reduced and controlled speed."
            )

        # ---------------------------------------------------------
        # PRIORITY 5: MODERATE TRAFFIC
        # ---------------------------------------------------------

        elif traffic_density == "moderate":

            action = "SLOW"
            speed = "MEDIUM"
            priority = "MEDIUM"

            reason = (
                f"Moderate traffic detected with "
                f"{vehicle_count} vehicles. "
                "Proceed cautiously."
            )

        # ---------------------------------------------------------
        # LOW RISK
        # ---------------------------------------------------------

        else:

            action = "GO"
            speed = "NORMAL"
            priority = "LOW"

            reason = (
                "No major hazards were identified. "
                "The vehicle may proceed normally while "
                "continuing environmental monitoring."
            )

        return {
            "action": action,
            "speed_level": speed,
            "reason": reason,
            "priority": priority,
            "risk_level": risk_level,
            "vehicle_count": vehicle_count,
            "vulnerable_users": vulnerable_users,
            "traffic_controls": traffic_controls,
            "traffic_density": traffic_density,
            "risk_indicators": risk_indicators,
        }

    # =============================================================
    # DISPLAY DECISION
    # =============================================================

    def display(self, decision: Dict):

        print()
        print("=" * 65)
        print("PLANNING AGENT RESULTS")
        print("=" * 65)

        print(
            f"Action       : "
            f"{decision['action']}"
        )

        print(
            f"Speed level  : "
            f"{decision['speed_level']}"
        )

        print(
            f"Priority     : "
            f"{decision['priority']}"
        )

        print(
            f"Risk level   : "
            f"{decision['risk_level']}"
        )

        print()
        print("REASON")
        print("-" * 65)
        print(decision["reason"])

        print("=" * 65)


# =================================================================
# TEST PLANNING AGENT WITH REAL BDD100K SCENE
# =================================================================

if __name__ == "__main__":

    from pathlib import Path
    import sys

    # Allow importing scene_agent.py
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent)
    )

    from perception import PerceptionAgent
    from scene_agent import SceneUnderstandingAgent

    # -------------------------------------------------------------
    # BDD100K validation dataset
    # -------------------------------------------------------------

    dataset_root = Path(
        r"F:\MultiAgent-Autonomous-Driving"
        r"\Dataset\BDD100K_Extracted"
        r"\bdd100k\bdd100k\images\100k\val"
    )

    images = sorted(
        dataset_root.glob("*.jpg")
    )

    if not images:
        raise FileNotFoundError(
            f"No BDD100K validation images found:\n"
            f"{dataset_root}"
        )

    image_path = images[0]

    print()
    print("=" * 65)
    print("MULTI-AGENT PLANNING TEST")
    print("=" * 65)

    print(
        f"BDD100K image: {image_path.name}"
    )

    # -------------------------------------------------------------
    # AGENT 1: PERCEPTION
    # -------------------------------------------------------------

    perception_agent = PerceptionAgent(
        confidence=0.25
    )

    perception_result = perception_agent.run(
        image_path
    )

    # -------------------------------------------------------------
    # AGENT 2: SCENE UNDERSTANDING
    # -------------------------------------------------------------

    scene_agent = SceneUnderstandingAgent()

    scene = scene_agent.analyze(
        perception_result
    )

    scene_agent.display(scene)

    # -------------------------------------------------------------
    # AGENT 3: PLANNING
    # -------------------------------------------------------------

    planning_agent = PlanningAgent()

    decision = planning_agent.plan(
        scene
    )

    planning_agent.display(
        decision
    )

    print()
    print("=" * 65)
    print("PLANNING PIPELINE COMPLETE")
    print("=" * 65)