from Agents.risk_fusion_agent import RiskFusionAgent


def main():

    print("=" * 70)
    print("MULTI-AGENT CONFLICT RESOLUTION TEST")
    print("=" * 70)

    # ============================================================
    # SYNTHETIC PERCEPTION RESULT
    # ============================================================

    perception_result = {

        "num_objects": 8,

        "detections": [

            {
                "class": "pedestrian",
                "confidence": 0.92,
                "bbox": [560, 280, 700, 600]
            },

            {
                "class": "car",
                "confidence": 0.91,
                "bbox": [300, 300, 500, 600]
            },

            {
                "class": "car",
                "confidence": 0.88,
                "bbox": [750, 300, 950, 600]
            },

            {
                "class": "car",
                "confidence": 0.86,
                "bbox": [100, 320, 260, 600]
            },

            {
                "class": "traffic light",
                "confidence": 0.94,
                "bbox": [600, 80, 660, 180]
            },

            {
                "class": "traffic sign",
                "confidence": 0.89,
                "bbox": [900, 100, 960, 180]
            },

            {
                "class": "car",
                "confidence": 0.82,
                "bbox": [1000, 300, 1180, 600]
            },

            {
                "class": "car",
                "confidence": 0.79,
                "bbox": [1150, 300, 1270, 600]
            }

        ]
    }

    # ============================================================
    # SYNTHETIC SCENE RESULT
    # ============================================================

    scene_result = {

        "scene_type":
            "road with vulnerable road users",

        "traffic_density":
            "moderate",

        "vehicle_count":
            6,

        "vulnerable_user_count":
            1,

        "traffic_control_count":
            2,

        "risk_level":
            "HIGH"
    }

    # ============================================================
    # INTENTIONAL PLANNER/SAFETY CONFLICT
    #
    # Planner says GO
    # Safety says STOP
    #
    # This is deliberately constructed to test
    # the conflict-resolution mechanism.
    # ============================================================

    planning_result = {

        "action":
            "GO",

        "speed":
            "NORMAL",

        "risk_level":
            "LOW",

        "reason":
            "Roadway appears clear for forward motion"
    }

    safety_result = {

        "decision":
            "OVERRIDE",

        "final_action":
            "STOP",

        "action":
            "STOP",

        "risk_level":
            "CRITICAL",

        "safety_level":
            "CRITICAL",

        "reason":
            "Pedestrian detected in potential vehicle path"
    }

    # ============================================================
    # INITIALIZE RISK FUSION AGENT
    # ============================================================

    fusion_agent = RiskFusionAgent()

    # ============================================================
    # RUN RISK FUSION
    # ============================================================

    result = fusion_agent.fuse(

        perception_result=
            perception_result,

        scene_result=
            scene_result,

        planning_result=
            planning_result,

        safety_result=
            safety_result,

        trajectory_predictions=[],

        image_width=
            1280,

        image_height=
            720
    )

    # ============================================================
    # DISPLAY INPUT AGENT DECISIONS
    # ============================================================

    print()
    print("INPUT AGENT DECISIONS")
    print("-" * 70)

    print(
        f"Planner action : "
        f"{planning_result['action']}"
    )

    print(
        f"Planner risk   : "
        f"{planning_result['risk_level']}"
    )

    print(
        f"Safety action  : "
        f"{safety_result['final_action']}"
    )

    print(
        f"Safety decision: "
        f"{safety_result['decision']}"
    )

    print(
        f"Safety risk    : "
        f"{safety_result['risk_level']}"
    )

    # ============================================================
    # DISPLAY RISK FUSION RESULT
    # ============================================================

    print()
    print("RISK FUSION")
    print("-" * 70)

    print(
        f"Risk score     : "
        f"{result['risk_score']}"
    )

    print(
        f"Risk percentage: "
        f"{result['risk_percentage']}"
    )

    print(
        f"Final action   : "
        f"{result['final_action']}"
    )

    print(
        f"Agent conflict : "
        f"{result['agent_conflict']}"
    )

    # ============================================================
    # RISK COMPONENT BREAKDOWN
    #
    # IMPORTANT:
    # The RiskFusionAgent returns this as:
    #     result["risk_components"]
    # ============================================================

    print()
    print("Risk breakdown")
    print("-" * 70)

    risk_components = result.get(
        "risk_components",
        {}
    )

    if risk_components:

        for name, value in risk_components.items():

            print(
                f"{name:<20}: "
                f"{value:.4f}"
            )

    else:

        print(
            "No risk components returned."
        )

    # ============================================================
    # RISK REASONS
    #
    # IMPORTANT:
    # The RiskFusionAgent returns this as:
    #     result["reasons"]
    # ============================================================

    print()
    print("Risk reasons")
    print("-" * 70)

    reasons = result.get(
        "reasons",
        []
    )

    if reasons:

        for reason in reasons:

            print(
                f"- {reason}"
            )

    else:

        print(
            "No risk reasons returned."
        )

    # ============================================================
    # HIGHEST-RISK OBJECT
    # ============================================================

    print()
    print("Highest-risk object")
    print("-" * 70)

    highest_object = result.get(
        "highest_risk_object"
    )

    if highest_object:

        print(
            f"Track ID       : "
            f"{highest_object.get('track_id')}"
        )

        print(
            f"Class          : "
            f"{highest_object.get('class')}"
        )

        print(
            f"Confidence     : "
            f"{highest_object.get('confidence')}"
        )

        print(
            f"Object risk    : "
            f"{highest_object.get('object_risk')}"
        )

        print(
            f"Prediction risk: "
            f"{highest_object.get('prediction_risk')}"
        )

    else:

        print(
            "No highest-risk object available."
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    conflict_detected = (
        result.get(
            "agent_conflict",
            False
        )
        is True
    )

    final_action = result.get(
        "final_action"
    )

    stop_selected = (
        final_action == "STOP"
    )

    # ============================================================
    # FINAL TEST RESULT
    # ============================================================

    print()
    print("=" * 70)

    if conflict_detected and stop_selected:

        print(
            "TEST RESULT: PASS"
        )

        print(
            "Conflict detected and "
            "safety-prioritized STOP selected."
        )

    else:

        print(
            "TEST RESULT: FAIL"
        )

        print(
            "Expected:"
        )

        print(
            "  Agent conflict = True"
        )

        print(
            "  Final action   = STOP"
        )

        print(
            "Actual:"
        )

        print(
            f"  Agent conflict = "
            f"{result.get('agent_conflict')}"
        )

        print(
            f"  Final action   = "
            f"{final_action}"
        )

    print("=" * 70)


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()