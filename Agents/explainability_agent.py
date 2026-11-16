class ExplainabilityAgent:
    """
    Explainability Agent for the Multi-Agent Autonomous Driving System.

    Converts the outputs of:
        - Perception Agent
        - Scene Understanding Agent
        - Planning Agent
        - Safety Agent
        - Risk Fusion Agent
        - Trajectory Prediction

    into a human-readable decision explanation.
    """

    def __init__(self):

        print()
        print("=" * 65)
        print("INITIALIZING EXPLAINABILITY AGENT")
        print("=" * 65)

        print(
            "Decision explanation module loaded."
        )

        print("=" * 65)

    # =============================================================
    # OBJECT SUMMARY
    # =============================================================

    @staticmethod
    def format_object_summary(
        perception_result
    ):

        detections = perception_result.get(
            "detections",
            []
        )

        if not detections:

            return "No objects detected."

        counts = {}

        for detection in detections:

            class_name = detection.get(
                "class",
                "unknown"
            )

            counts[class_name] = (
                counts.get(
                    class_name,
                    0
                )
                + 1
            )

        parts = []

        for class_name, count in counts.items():

            parts.append(
                f"{count} {class_name}"
            )

        return ", ".join(parts)

    # =============================================================
    # RISK DESCRIPTION
    # =============================================================

    @staticmethod
    def risk_description(
        risk_score
    ):

        if risk_score >= 0.70:

            return "HIGH"

        elif risk_score >= 0.30:

            return "MODERATE"

        else:

            return "LOW"

    # =============================================================
    # EXPLAIN RISK COMPONENTS
    # =============================================================

    @staticmethod
    def explain_risk_components(
        risk_fusion_result
    ):

        if not risk_fusion_result:

            return []

        components = risk_fusion_result.get(
            "risk_components",
            {}
        )

        explanations = []

        scene_risk = components.get(
            "scene_risk",
            0.0
        )

        object_risk = components.get(
            "object_risk",
            0.0
        )

        prediction_risk = components.get(
            "prediction_risk",
            0.0
        )

        planning_risk = components.get(
            "planning_risk",
            0.0
        )

        safety_risk = components.get(
            "safety_risk",
            0.0
        )

        conflict_risk = components.get(
            "conflict_risk",
            0.0
        )

        # ---------------------------------------------------------
        # Scene
        # ---------------------------------------------------------

        if scene_risk >= 0.20:

            explanations.append(
                "High scene-level risk contributes "
                "to the final decision."
            )

        elif scene_risk > 0:

            explanations.append(
                "Scene conditions contribute "
                "moderately to the risk score."
            )

        # ---------------------------------------------------------
        # Object
        # ---------------------------------------------------------

        if object_risk >= 0.20:

            explanations.append(
                "Object-level factors contribute "
                "to the risk score."
            )

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        if prediction_risk > 0:

            explanations.append(
                "Predicted object motion approaches "
                "the estimated ego danger region."
            )

        else:

            explanations.append(
                "No predicted trajectory threat "
                "was detected."
            )

        # ---------------------------------------------------------
        # Planning
        # ---------------------------------------------------------

        if planning_risk >= 0.15:

            explanations.append(
                "The Planning Agent contributes "
                "elevated risk because it recommends STOP."
            )

        elif planning_risk > 0:

            explanations.append(
                "The Planning Agent recommends "
                "a cautious driving action."
            )

        # ---------------------------------------------------------
        # Safety
        # ---------------------------------------------------------

        if safety_risk >= 0.20:

            explanations.append(
                "The Safety Agent identifies "
                "elevated safety risk."
            )

        elif safety_risk > 0:

            explanations.append(
                "The Safety Agent indicates "
                "a cautionary condition."
            )

        # ---------------------------------------------------------
        # Conflict
        # ---------------------------------------------------------

        if conflict_risk > 0:

            explanations.append(
                "A disagreement exists between "
                "the Planning and Safety Agents."
            )

        else:

            explanations.append(
                "No disagreement exists between "
                "the Planning and Safety Agents."
            )

        return explanations

    # =============================================================
    # MAIN EXPLANATION
    # =============================================================

    def explain(
        self,
        perception_result,
        scene_result,
        planning_decision,
        safety_result,
        risk_fusion_result=None,
        trajectory_predictions=None
    ):
        """
        Generate a complete human-readable explanation.

        The first four arguments remain compatible with the
        previous ExplainabilityAgent implementation.
        """

        if trajectory_predictions is None:

            trajectory_predictions = []

        # =========================================================
        # PERCEPTION
        # =========================================================

        num_objects = perception_result.get(
            "num_objects",
            len(
                perception_result.get(
                    "detections",
                    []
                )
            )
        )

        object_summary = (
            self.format_object_summary(
                perception_result
            )
        )

        # =========================================================
        # SCENE
        # =========================================================

        scene_type = scene_result.get(
            "scene_type",
            "unknown"
        )

        traffic_density = scene_result.get(
            "traffic_density",
            "unknown"
        )

        scene_risk_level = scene_result.get(
            "risk_level",
            "LOW"
        )

        vehicle_count = scene_result.get(
            "vehicle_count",
            0
        )

        vulnerable_users = scene_result.get(
            "vulnerable_users",
            0
        )

        traffic_controls = scene_result.get(
            "traffic_controls",
            0
        )

        # =========================================================
        # PLANNING
        # =========================================================

        planning_action = planning_decision.get(
            "action",
            "UNKNOWN"
        )

        planning_reason = planning_decision.get(
            "reason",
            "No planning reason provided."
        )

        # =========================================================
        # SAFETY
        # =========================================================

        safety_decision = safety_result.get(
            "decision",
            "UNKNOWN"
        )

        safety_level = safety_result.get(
            "safety_level",
            "UNKNOWN"
        )

        safety_action = safety_result.get(
            "final_action",
            planning_action
        )

        safety_reasons = safety_result.get(
            "reasons",
            []
        )

        # =========================================================
        # RISK FUSION
        # =========================================================

        if risk_fusion_result:

            risk_score = risk_fusion_result.get(
                "risk_score",
                0.0
            )

            risk_percentage = risk_fusion_result.get(
                "risk_percentage",
                risk_score * 100
            )

            fusion_action = risk_fusion_result.get(
                "final_action",
                safety_action
            )

            agent_conflict = risk_fusion_result.get(
                "agent_conflict",
                False
            )

            risk_reasons = risk_fusion_result.get(
                "reasons",
                []
            )

            risk_level = self.risk_description(
                risk_score
            )

            risk_components_explanation = (
                self.explain_risk_components(
                    risk_fusion_result
                )
            )

        else:

            risk_score = 0.0

            risk_percentage = 0.0

            fusion_action = safety_action

            agent_conflict = (
                planning_action
                !=
                safety_action
            )

            risk_reasons = []

            risk_level = scene_risk_level

            risk_components_explanation = []

        # =========================================================
        # TRAJECTORY INFORMATION
        # =========================================================

        trajectory_threat = False

        highest_prediction_risk = 0.0

        if risk_fusion_result:

            highest_object = (
                risk_fusion_result.get(
                    "highest_risk_object"
                )
            )

            if highest_object:

                highest_prediction_risk = (
                    highest_object.get(
                        "prediction_risk",
                        0.0
                    )
                )

                trajectory_threat = (
                    highest_prediction_risk > 0
                )

        # =========================================================
        # BUILD EXPLANATION
        # =========================================================

        explanation_parts = []

        # ---------------------------------------------------------
        # Perception
        # ---------------------------------------------------------

        explanation_parts.append(

            f"The perception system detected "
            f"{num_objects} object(s) in the scene."
        )

        if object_summary:

            explanation_parts.append(

                f"Detected objects include: "
                f"{object_summary}."
            )

        # ---------------------------------------------------------
        # Scene
        # ---------------------------------------------------------

        explanation_parts.append(

            f"The scene was classified as "
            f"'{scene_type}'."
        )

        explanation_parts.append(

            f"Traffic density was classified as "
            f"'{traffic_density}'."
        )

        explanation_parts.append(

            f"The estimated scene risk level was "
            f"'{scene_risk_level}'."
        )

        explanation_parts.append(

            f"The scene contained "
            f"{vehicle_count} vehicle(s), "
            f"{vulnerable_users} vulnerable "
            f"road user(s), and "
            f"{traffic_controls} traffic "
            f"control object(s)."
        )

        # ---------------------------------------------------------
        # Planning
        # ---------------------------------------------------------

        explanation_parts.append(

            f"The Planning Agent proposed "
            f"'{planning_action}'."
        )

        explanation_parts.append(

            f"Planning reason: "
            f"{planning_reason}"
        )

        # ---------------------------------------------------------
        # Safety
        # ---------------------------------------------------------

        explanation_parts.append(

            f"The Safety Agent returned "
            f"'{safety_decision}' with safety "
            f"level '{safety_level}'."
        )

        if safety_reasons:

            explanation_parts.append(

                "Safety reasoning: "
                +
                " ".join(
                    str(reason)
                    for reason in safety_reasons
                )
            )

        # ---------------------------------------------------------
        # Risk Fusion
        # ---------------------------------------------------------

        if risk_fusion_result:

            explanation_parts.append(

                f"The Confidence-Aware Risk "
                f"Fusion Agent calculated a "
                f"normalized risk score of "
                f"{risk_score:.2f} "
                f"({risk_percentage:.1f}%)."
            )

            explanation_parts.append(

                f"The corresponding fused risk "
                f"level was '{risk_level}'."
            )

            # -----------------------------------------------------
            # Trajectory
            # -----------------------------------------------------

            if trajectory_threat:

                explanation_parts.append(

                    "A predicted trajectory "
                    "threat contributed to the "
                    "risk assessment."
                )

            else:

                explanation_parts.append(

                    "No predictive trajectory "
                    "threat contributed to the "
                    "risk assessment."
                )

            # -----------------------------------------------------
            # Conflict
            # -----------------------------------------------------

            if agent_conflict:

                explanation_parts.append(

                    "The Planning and Safety "
                    "Agents produced different "
                    "actions, so an agent conflict "
                    "was detected."
                )

            else:

                explanation_parts.append(

                    "The Planning and Safety "
                    "Agents were consistent."
                )

        # ---------------------------------------------------------
        # Final
        # ---------------------------------------------------------

        explanation_parts.append(

            f"Therefore, the final recommended "
            f"driving action is '{fusion_action}'."
        )

        # =========================================================
        # FULL EXPLANATION
        # =========================================================

        explanation = " ".join(
            explanation_parts
        )

        # =========================================================
        # RISK BREAKDOWN
        # =========================================================

        risk_breakdown = (
            risk_components_explanation
        )

        # =========================================================
        # RESULT
        # =========================================================

        return {

            "final_action":
                fusion_action,

            "explanation":
                explanation,

            "object_summary":
                object_summary,

            "scene_type":
                scene_type,

            "traffic_density":
                traffic_density,

            "risk_level":
                risk_level,

            "risk_score":
                round(
                    risk_score,
                    4
                ),

            "risk_percentage":
                round(
                    risk_percentage,
                    2
                ),

            "planning_action":
                planning_action,

            "safety_decision":
                safety_decision,

            "safety_action":
                safety_action,

            "safety_level":
                safety_level,

            "agent_conflict":
                agent_conflict,

            "trajectory_threat":
                trajectory_threat,

            "risk_breakdown":
                risk_breakdown,

            "risk_reasons":
                risk_reasons,

            "trajectory_count":
                len(
                    trajectory_predictions
                )
        }


# =================================================================
# STANDALONE TEST
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("EXPLAINABILITY AGENT TEST")
    print("=" * 65)

    agent = ExplainabilityAgent()

    # -------------------------------------------------------------
    # Perception
    # -------------------------------------------------------------

    perception = {

        "num_objects":
            3,

        "detections":
            [

                {
                    "class":
                        "car",

                    "confidence":
                        0.91
                },

                {
                    "class":
                        "car",

                    "confidence":
                        0.87
                },

                {
                    "class":
                        "pedestrian",

                    "confidence":
                        0.89
                }
            ]
    }

    # -------------------------------------------------------------
    # Scene
    # -------------------------------------------------------------

    scene = {

        "scene_type":
            "road with vulnerable road users",

        "traffic_density":
            "moderate",

        "risk_level":
            "HIGH",

        "vehicle_count":
            2,

        "vulnerable_users":
            1,

        "traffic_controls":
            0
    }

    # -------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------

    planning = {

        "action":
            "SLOW",

        "reason":
            "Vulnerable road user detected."
    }

    # -------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------

    safety = {

        "decision":
            "OVERRIDE",

        "final_action":
            "STOP",

        "safety_level":
            "CRITICAL",

        "reasons":
            [
                "Vulnerable road user detected."
            ]
    }

    # -------------------------------------------------------------
    # Risk Fusion
    # -------------------------------------------------------------

    risk_fusion = {

        "risk_score":
            0.82,

        "risk_percentage":
            82.0,

        "final_action":
            "STOP",

        "agent_conflict":
            True,

        "risk_components":
            {

                "scene_risk":
                    0.25,

                "object_risk":
                    0.30,

                "prediction_risk":
                    0.20,

                "planning_risk":
                    0.10,

                "safety_risk":
                    0.25,

                "conflict_risk":
                    0.15
            },

        "highest_risk_object":
            {

                "track_id":
                    1,

                "class":
                    "pedestrian",

                "prediction_risk":
                    0.20
            },

        "reasons":
            [
                "Elevated scene-level risk",
                "Elevated object-level risk",
                "Predicted trajectory approaches "
                "the ego danger zone",
                "Planner and safety agents disagree"
            ]
    }

    # -------------------------------------------------------------
    # Explain
    # -------------------------------------------------------------

    result = agent.explain(

        perception,

        scene,

        planning,

        safety,

        risk_fusion,

        []
    )

    print()
    print("FINAL ACTION:")
    print(
        result["final_action"]
    )

    print()
    print("RISK SCORE:")
    print(
        result["risk_score"]
    )

    print()
    print("EXPLANATION:")
    print(
        result["explanation"]
    )

    print()
    print("RISK BREAKDOWN:")

    for item in result[
        "risk_breakdown"
    ]:

        print(
            f"- {item}"
        )

    print()
    print("=" * 65)
    print("EXPLAINABILITY TEST COMPLETE")
    print("=" * 65)