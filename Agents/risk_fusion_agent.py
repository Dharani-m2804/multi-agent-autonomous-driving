class RiskFusionAgent:
    """
    Confidence-Aware Risk-Based Multi-Agent Decision Fusion.

    Combines information from:
        - Scene Understanding
        - Object perception confidence
        - Object proximity
        - Object velocity
        - Trajectory prediction
        - Planning Agent
        - Safety Agent
        - Agent disagreement

    Output:
        - Normalized risk score
        - Risk components
        - Highest-risk object
        - Agent conflict
        - Final GO / SLOW / STOP action

    IMPORTANT:
        risk_score is a normalized decision-support score.
        It is NOT a calibrated probability of collision.
    """

    def __init__(self):

        print()
        print("=" * 65)
        print("INITIALIZING RISK FUSION AGENT")
        print("=" * 65)

        print(
            "Confidence-aware multi-agent decision fusion enabled."
        )

        print("=" * 65)

    # =============================================================
    # CONFIDENCE RISK
    # =============================================================

    @staticmethod
    def confidence_risk(confidence):
        """
        Converts perception uncertainty into a small risk contribution.

        Low confidence increases uncertainty, but confidence alone
        must never cause an immediate STOP decision.
        """

        if confidence >= 0.80:
            return 0.02

        elif confidence >= 0.60:
            return 0.05

        elif confidence >= 0.40:
            return 0.10

        else:
            return 0.15

    # =============================================================
    # VULNERABLE USER RISK
    # =============================================================

    @staticmethod
    def vulnerable_user_risk(class_name):
        """
        Vulnerable road users receive additional risk because
        pedestrians/riders require stronger safety consideration.
        """

        vulnerable_classes = {
            "pedestrian",
            "rider",
            "other person"
        }

        if class_name in vulnerable_classes:
            return 0.20

        return 0.0

    # =============================================================
    # VELOCITY RISK
    # =============================================================

    @staticmethod
    def velocity_risk(speed):
        """
        Image-space velocity contribution.

        This is NOT physical vehicle speed.
        It is a relative motion indicator in pixels/second.
        """

        if speed >= 500:
            return 0.15

        elif speed >= 250:
            return 0.10

        elif speed >= 100:
            return 0.05

        return 0.0

    # =============================================================
    # PROXIMITY RISK
    # =============================================================

    @staticmethod
    def proximity_risk(
        center,
        image_width=1280
    ):
        """
        Estimates horizontal proximity to the ego-vehicle region.

        This is an image-space heuristic.
        """

        ego_x = image_width / 2.0

        distance = abs(
            center[0] - ego_x
        )

        normalized_distance = (
            distance / ego_x
        )

        if normalized_distance < 0.10:
            return 0.15

        elif normalized_distance < 0.20:
            return 0.10

        elif normalized_distance < 0.35:
            return 0.05

        return 0.0

    # =============================================================
    # TRAJECTORY RISK
    # =============================================================

    @staticmethod
    def trajectory_risk(
        prediction,
        image_width=1280,
        image_height=720
    ):
        """
        Checks predicted object positions against an approximate
        ego danger region.

        This becomes meaningful when consecutive video frames
        provide actual object velocity.

        For single-frame BDD100K processing, velocity is zero and
        therefore prediction risk normally remains zero.
        """

        if not prediction:
            return 0.0

        risk = 0.0

        predictions = prediction.get(
            "predictions",
            []
        )

        for future in predictions:

            position = future.get(
                "position",
                [0, 0]
            )

            x = position[0]
            y = position[1]

            ego_x = image_width / 2.0

            horizontal_distance = abs(
                x - ego_x
            )

            near_road_region = (
                y > image_height * 0.55
            )

            if (
                horizontal_distance
                <
                image_width * 0.12
                and
                near_road_region
            ):

                t = future.get(
                    "time_seconds",
                    2.0
                )

                if t <= 0.5:

                    risk = max(
                        risk,
                        0.35
                    )

                elif t <= 1.0:

                    risk = max(
                        risk,
                        0.30
                    )

                elif t <= 1.5:

                    risk = max(
                        risk,
                        0.20
                    )

                else:

                    risk = max(
                        risk,
                        0.10
                    )

        return risk

    # =============================================================
    # SCENE RISK
    # =============================================================

    @staticmethod
    def scene_risk(scene_result):
        """
        Converts scene-level risk into a fusion contribution.
        """

        risk_level = scene_result.get(
            "risk_level",
            "LOW"
        )

        if risk_level == "HIGH":

            return 0.25

        elif risk_level == "MODERATE":

            return 0.10

        return 0.02

    # =============================================================
    # PLANNING RISK
    # =============================================================

    @staticmethod
    def planning_risk(planning_result):
        """
        Planning action contributes to overall risk.
        """

        action = planning_result.get(
            "action",
            "GO"
        )

        if action == "STOP":

            return 0.20

        elif action == "SLOW":

            return 0.10

        return 0.02

    # =============================================================
    # SAFETY RISK
    # =============================================================

    @staticmethod
    def safety_risk(safety_result):
        """
        Converts Safety Agent output into fusion evidence.
        """

        decision = safety_result.get(
            "decision",
            "APPROVE"
        )

        safety_level = safety_result.get(
            "safety_level",
            "SAFE"
        )

        risk = 0.0

        if decision == "OVERRIDE":

            risk += 0.25

        elif decision == "MODIFY":

            risk += 0.12

        if safety_level == "CRITICAL":

            risk += 0.15

        elif safety_level == "CAUTION":

            risk += 0.03

        return min(
            risk,
            0.35
        )

    # =============================================================
    # AGENT CONFLICT
    # =============================================================

    @staticmethod
    def detect_conflict(
        planning_result,
        safety_result
    ):
        """
        Detects disagreement between planning and safety agents.
        """

        planning_action = planning_result.get(
            "action",
            "GO"
        )

        safety_action = safety_result.get(
            "final_action",
            planning_action
        )

        return (
            planning_action
            !=
            safety_action
        )

    # =============================================================
    # FINAL DECISION
    # =============================================================

    @staticmethod
    def final_decision(
        risk_score,
        planning_result,
        safety_result,
        prediction_risk
    ):
        """
        Generates final action.

        Priority:
            1. Strong predictive threat
            2. Safety STOP
            3. High fusion risk
            4. Moderate fusion risk
            5. Preserve planning action
        """

        safety_action = safety_result.get(
            "final_action",
            "GO"
        )

        planning_action = planning_result.get(
            "action",
            "GO"
        )

        # ---------------------------------------------------------
        # Strong predictive threat
        # ---------------------------------------------------------

        if prediction_risk >= 0.30:

            return "STOP"

        # ---------------------------------------------------------
        # Safety agent requires STOP
        # ---------------------------------------------------------

        if safety_action == "STOP":

            return "STOP"

        # ---------------------------------------------------------
        # High combined risk
        # ---------------------------------------------------------

        if risk_score >= 0.70:

            return "STOP"

        # ---------------------------------------------------------
        # Moderate combined risk
        # ---------------------------------------------------------

        elif risk_score >= 0.30:

            return "SLOW"

        # ---------------------------------------------------------
        # Low risk
        #
        # Preserve the existing planning decision.
        # ---------------------------------------------------------

        return planning_action

    # =============================================================
    # MAIN FUSION
    # =============================================================

    def fuse(
        self,
        perception_result,
        scene_result,
        planning_result,
        safety_result,
        trajectory_predictions=None,
        image_width=1280,
        image_height=720
    ):
        """
        Main risk-fusion function.
        """

        if trajectory_predictions is None:

            trajectory_predictions = []

        # ---------------------------------------------------------
        # Risk components
        # ---------------------------------------------------------

        risk_components = {

            "scene_risk":
                self.scene_risk(
                    scene_result
                ),

            "object_risk":
                0.0,

            "prediction_risk":
                0.0,

            "planning_risk":
                self.planning_risk(
                    planning_result
                ),

            "safety_risk":
                self.safety_risk(
                    safety_result
                ),

            "conflict_risk":
                0.0
        }

        object_risks = []

        # =========================================================
        # OBJECT-LEVEL ANALYSIS
        # =========================================================

        for prediction in trajectory_predictions:

            confidence = prediction.get(
                "confidence",
                0.0
            )

            center = prediction.get(
                "current_position",
                [image_width / 2, image_height / 2]
            )

            speed = prediction.get(
                "speed",
                0.0
            )

            class_name = prediction.get(
                "class",
                "unknown"
            )

            # -----------------------------------------------------
            # Individual risk components
            # -----------------------------------------------------

            confidence_component = (
                self.confidence_risk(
                    confidence
                )
            )

            proximity_component = (
                self.proximity_risk(
                    center,
                    image_width
                )
            )

            velocity_component = (
                self.velocity_risk(
                    speed
                )
            )

            vulnerable_component = (
                self.vulnerable_user_risk(
                    class_name
                )
            )

            prediction_component = (
                self.trajectory_risk(
                    prediction,
                    image_width,
                    image_height
                )
            )

            # -----------------------------------------------------
            # Object risk
            # -----------------------------------------------------

            object_score = (

                confidence_component

                +

                proximity_component

                +

                velocity_component

                +

                vulnerable_component
            )

            # Prevent one object from dominating
            # the complete fusion score.

            object_score = min(
                object_score,
                0.35
            )

            object_risks.append(
                {

                    "track_id":
                        prediction.get(
                            "track_id"
                        ),

                    "class":
                        class_name,

                    "confidence":
                        confidence,

                    "object_risk":
                        round(
                            object_score,
                            4
                        ),

                    "prediction_risk":
                        round(
                            prediction_component,
                            4
                        ),

                    "velocity":
                        prediction.get(
                            "velocity",
                            {
                                "x": 0.0,
                                "y": 0.0
                            }
                        ),

                    "speed":
                        speed,

                    "current_position":
                        center
                }
            )

        # =========================================================
        # HIGHEST-RISK OBJECT
        # =========================================================

        highest_object = None

        if object_risks:

            highest_object = max(
                object_risks,
                key=lambda x:
                    (
                        x["object_risk"]
                        +
                        x["prediction_risk"]
                    )
            )

            risk_components[
                "object_risk"
            ] = highest_object[
                "object_risk"
            ]

            risk_components[
                "prediction_risk"
            ] = highest_object[
                "prediction_risk"
            ]

        # =========================================================
        # AGENT CONFLICT
        # =========================================================

        conflict = self.detect_conflict(
            planning_result,
            safety_result
        )

        if conflict:

            risk_components[
                "conflict_risk"
            ] = 0.15

        # =========================================================
        # TOTAL RISK
        # =========================================================

        raw_score = sum(
            risk_components.values()
        )

        risk_score = min(
            raw_score,
            1.0
        )

        # =========================================================
        # FINAL ACTION
        # =========================================================

        final_action = self.final_decision(

            risk_score,

            planning_result,

            safety_result,

            risk_components[
                "prediction_risk"
            ]
        )

        # =========================================================
        # EXPLANATION REASONS
        # =========================================================

        reasons = []

        if (
            risk_components[
                "scene_risk"
            ] >= 0.20
        ):

            reasons.append(
                "Elevated scene-level risk"
            )

        if (
            risk_components[
                "object_risk"
            ] >= 0.20
        ):

            reasons.append(
                "Elevated object-level risk"
            )

        if (
            risk_components[
                "prediction_risk"
            ] > 0
        ):

            reasons.append(
                "Predicted trajectory approaches "
                "the ego danger zone"
            )

        if (
            planning_result.get(
                "action"
            )
            ==
            "STOP"
        ):

            reasons.append(
                "Planning agent recommends STOP"
            )

        elif (
            planning_result.get(
                "action"
            )
            ==
            "SLOW"
        ):

            reasons.append(
                "Planning agent recommends SLOW"
            )

        if (
            risk_components[
                "safety_risk"
            ] >= 0.20
        ):

            reasons.append(
                "Safety agent indicates elevated risk"
            )

        if conflict:

            reasons.append(
                "Planner and safety agents disagree"
            )

        if not reasons:

            reasons.append(
                "No significant risk indicators detected"
            )

        # =========================================================
        # FINAL RESULT
        # =========================================================

        return {

            "risk_score":
                round(
                    risk_score,
                    4
                ),

            "risk_percentage":
                round(
                    risk_score * 100,
                    2
                ),

            "final_action":
                final_action,

            "agent_conflict":
                conflict,

            "risk_components":
                {
                    key:
                        round(
                            value,
                            4
                        )

                    for key, value
                    in risk_components.items()
                },

            "highest_risk_object":
                highest_object,

            "object_risks":
                object_risks,

            "reasons":
                reasons
        }

    # =============================================================
    # DISPLAY
    # =============================================================

    def display(
        self,
        result
    ):

        print()
        print("=" * 65)
        print("CONFIDENCE-AWARE RISK FUSION")
        print("=" * 65)

        print(
            f"Risk Score      : "
            f"{result['risk_score']:.4f}"
        )

        print(
            f"Risk Percentage : "
            f"{result['risk_percentage']:.2f}%"
        )

        print(
            f"Final Action    : "
            f"{result['final_action']}"
        )

        print(
            f"Agent Conflict  : "
            f"{result['agent_conflict']}"
        )

        print()
        print("Risk Components:")

        for name, value in result[
            "risk_components"
        ].items():

            print(
                f"  {name:<20}: "
                f"{value:.4f}"
            )

        # ---------------------------------------------------------
        # Highest-risk object
        # ---------------------------------------------------------

        if result[
            "highest_risk_object"
        ]:

            obj = result[
                "highest_risk_object"
            ]

            print()
            print(
                "Highest Risk Object:"
            )

            print(
                f"  ID         : "
                f"{obj['track_id']}"
            )

            print(
                f"  Class      : "
                f"{obj['class']}"
            )

            print(
                f"  Confidence : "
                f"{obj['confidence']:.2f}"
            )

            print(
                f"  Speed      : "
                f"{obj['speed']:.2f} px/s"
            )

            print(
                f"  Object Risk: "
                f"{obj['object_risk']:.4f}"
            )

            print(
                f"  Prediction : "
                f"{obj['prediction_risk']:.4f}"
            )

        # ---------------------------------------------------------
        # Reasons
        # ---------------------------------------------------------

        print()
        print("Reasons:")

        for reason in result[
            "reasons"
        ]:

            print(
                f"  - {reason}"
            )

        print("=" * 65)


# =================================================================
# STANDALONE TEST
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("RISK FUSION AGENT TEST")
    print("=" * 65)

    fusion = RiskFusionAgent()

    # =============================================================
    # TEST SCENE
    # =============================================================

    scene = {

        "scene_type":
            "roadway",

        "traffic_density":
            "moderate",

        "vehicle_count":
            8,

        "vulnerable_users":
            0,

        "traffic_controls":
            0,

        "risk_level":
            "MODERATE"
    }

    # =============================================================
    # TEST PLANNING
    # =============================================================

    planning = {

        "action":
            "SLOW",

        "speed_level":
            "LOW",

        "risk_level":
            "MODERATE"
    }

    # =============================================================
    # TEST SAFETY
    # =============================================================

    safety = {

        "proposed_action":
            "SLOW",

        "final_action":
            "SLOW",

        "decision":
            "APPROVE",

        "safety_level":
            "CAUTION"
    }

    # =============================================================
    # TEST TRAJECTORY
    # =============================================================

    trajectory = [

        {

            "track_id":
                1,

            "class":
                "car",

            "confidence":
                0.90,

            "current_position":
                [
                    700,
                    500
                ],

            "velocity":
                {
                    "x":
                        0,

                    "y":
                        0
                },

            "speed":
                0,

            "predictions":
                [

                    {
                        "time_seconds":
                            0.5,

                        "position":
                            [
                                700,
                                500
                            ]
                    },

                    {
                        "time_seconds":
                            1.0,

                        "position":
                            [
                                700,
                                500
                            ]
                    },

                    {
                        "time_seconds":
                            1.5,

                        "position":
                            [
                                700,
                                500
                            ]
                    },

                    {
                        "time_seconds":
                            2.0,

                        "position":
                            [
                                700,
                                500
                            ]
                    }
                ]
        }
    ]

    # =============================================================
    # TEST PERCEPTION
    # =============================================================

    perception = {

        "num_objects":
            1
    }

    # =============================================================
    # RUN
    # =============================================================

    result = fusion.fuse(

        perception_result=
            perception,

        scene_result=
            scene,

        planning_result=
            planning,

        safety_result=
            safety,

        trajectory_predictions=
            trajectory,

        image_width=
            1280,

        image_height=
            720
    )

    fusion.display(
        result
    )

    print()
    print("=" * 65)
    print("RISK FUSION TEST COMPLETE")
    print("=" * 65)