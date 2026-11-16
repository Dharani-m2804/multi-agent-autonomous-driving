from pathlib import Path
import sys


class SafetyAgent:
    """
    Safety Verification Agent.

    Independently verifies the action proposed by the
    Planning Agent using scene-level risk information.

    Possible final decisions:
        APPROVE
        MODIFY
        OVERRIDE
    """

    def __init__(self):

        print()
        print("=" * 65)
        print("INITIALIZING SAFETY AGENT")
        print("=" * 65)

        print("Independent safety verification module loaded.")

        print("=" * 65)

    # =============================================================
    # SAFETY VERIFICATION
    # =============================================================

    def verify(self, scene, planning_decision):

        proposed_action = planning_decision.get(
            "action",
            "STOP"
        ).upper()

        risk_level = scene.get(
            "risk_level",
            "HIGH"
        ).upper()

        vulnerable_users = scene.get(
            "vulnerable_users",
            0
        )

        traffic_controls = scene.get(
            "traffic_controls",
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
        # DEFAULT
        # ---------------------------------------------------------

        final_action = proposed_action
        decision = "APPROVE"
        safety_level = "SAFE"

        reasons = []

        # ---------------------------------------------------------
        # CRITICAL RULE:
        # Vulnerable road users → STOP
        # ---------------------------------------------------------

        if vulnerable_users > 0:

            final_action = "STOP"
            decision = "OVERRIDE"
            safety_level = "CRITICAL"

            reasons.append(
                "Vulnerable road user detected."
            )

            reasons.append(
                "Vehicle should stop until the road "
                "situation is confirmed safe."
            )

        # ---------------------------------------------------------
        # HIGH RISK → STOP
        # ---------------------------------------------------------

        elif risk_level == "HIGH":

            final_action = "STOP"
            decision = "OVERRIDE"
            safety_level = "CRITICAL"

            reasons.append(
                "Scene classified as HIGH risk."
            )

            reasons.append(
                "Stopping is required as the safest action."
            )

        # ---------------------------------------------------------
        # TRAFFIC CONTROL
        # ---------------------------------------------------------

        elif traffic_controls > 0:

            if proposed_action == "GO":

                final_action = "SLOW"
                decision = "MODIFY"
                safety_level = "CAUTION"

                reasons.append(
                    "Traffic control infrastructure detected."
                )

                reasons.append(
                    "GO changed to SLOW for additional caution."
                )

            else:

                final_action = proposed_action
                decision = "APPROVE"
                safety_level = "CAUTION"

                reasons.append(
                    "Traffic control infrastructure detected."
                )

        # ---------------------------------------------------------
        # HIGH TRAFFIC
        # ---------------------------------------------------------

        elif traffic_density == "high":

            if proposed_action == "GO":

                final_action = "SLOW"
                decision = "MODIFY"
                safety_level = "CAUTION"

                reasons.append(
                    "High traffic density detected."
                )

                reasons.append(
                    "Reduced speed is safer in dense traffic."
                )

            else:

                final_action = proposed_action
                decision = "APPROVE"
                safety_level = "CAUTION"

                reasons.append(
                    "High traffic density detected."
                )

                reasons.append(
                    "Planner already selected a cautious action."
                )

        # ---------------------------------------------------------
        # MODERATE RISK
        # ---------------------------------------------------------

        elif risk_level == "MODERATE":

            final_action = proposed_action
            decision = "APPROVE"
            safety_level = "CAUTION"

            reasons.append(
                "Scene has moderate risk."
            )

            reasons.append(
                "Planner action is considered acceptable."
            )

        # ---------------------------------------------------------
        # LOW RISK
        # ---------------------------------------------------------

        else:

            final_action = proposed_action
            decision = "APPROVE"
            safety_level = "SAFE"

            reasons.append(
                "No major safety hazards detected."
            )

            reasons.append(
                "Planner action is approved."
            )

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------

        return {
            "proposed_action": proposed_action,
            "final_action": final_action,
            "decision": decision,
            "safety_level": safety_level,
            "reasons": reasons,
            "risk_level": risk_level,
            "vulnerable_users": vulnerable_users,
            "traffic_controls": traffic_controls,
            "traffic_density": traffic_density,
            "risk_indicators": risk_indicators,
        }

    # =============================================================
    # DISPLAY
    # =============================================================

    def display(self, result):

        print()
        print("=" * 65)
        print("SAFETY AGENT RESULTS")
        print("=" * 65)

        print(
            f"Planner action : "
            f"{result['proposed_action']}"
        )

        print(
            f"Safety decision: "
            f"{result['decision']}"
        )

        print(
            f"Safety level   : "
            f"{result['safety_level']}"
        )

        print(
            f"Final action   : "
            f"{result['final_action']}"
        )

        print()
        print("SAFETY REASONING")
        print("-" * 65)

        for reason in result["reasons"]:
            print(f"- {reason}")

        print("=" * 65)


# =================================================================
# COMPLETE MULTI-AGENT TEST
# =================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------
    # Allow imports from Agents directory
    # -------------------------------------------------------------

    agents_directory = Path(
        __file__
    ).resolve().parent

    sys.path.insert(
        0,
        str(agents_directory)
    )

    from perception import PerceptionAgent
    from scene_agent import SceneUnderstandingAgent
    from planning_agent import PlanningAgent

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
    print("MULTI-AGENT SAFETY TEST")
    print("=" * 65)

    print(
        f"BDD100K image: "
        f"{image_path.name}"
    )

    # =============================================================
    # AGENT 1
    # PERCEPTION
    # =============================================================

    perception_agent = PerceptionAgent(
        confidence=0.25
    )

    perception_result = perception_agent.run(
        image_path
    )

    # =============================================================
    # AGENT 2
    # SCENE UNDERSTANDING
    # =============================================================

    scene_agent = SceneUnderstandingAgent()

    scene = scene_agent.analyze(
        perception_result
    )

    scene_agent.display(
        scene
    )

    # =============================================================
    # AGENT 3
    # PLANNING
    # =============================================================

    planning_agent = PlanningAgent()

    planning_decision = planning_agent.plan(
        scene
    )

    planning_agent.display(
        planning_decision
    )

    # =============================================================
    # AGENT 4
    # SAFETY
    # =============================================================

    safety_agent = SafetyAgent()

    safety_result = safety_agent.verify(
        scene,
        planning_decision
    )

    safety_agent.display(
        safety_result
    )

    # =============================================================
    # FINAL
    # =============================================================

    print()
    print("=" * 65)
    print("MULTI-AGENT SAFETY PIPELINE COMPLETE")
    print("=" * 65)

    print(
        f"FINAL DRIVING ACTION: "
        f"{safety_result['final_action']}"
    )

    print("=" * 65)