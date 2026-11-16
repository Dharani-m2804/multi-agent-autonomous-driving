from pathlib import Path


class SceneUnderstandingAgent:
    """
    Scene Understanding Agent

    Converts raw object detections from the Perception Agent
    into a structured representation of the driving scene.
    """

    VEHICLE_CLASSES = {
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "other vehicle",
        "trailer",
        "train",
    }

    VULNERABLE_ROAD_USERS = {
        "pedestrian",
        "rider",
        "other person",
    }

    TRAFFIC_CONTROL_CLASSES = {
        "traffic light",
        "traffic sign",
    }

    def __init__(self):
        print()
        print("=" * 65)
        print("INITIALIZING SCENE UNDERSTANDING AGENT")
        print("=" * 65)
        print("Scene reasoning module loaded.")
        print("=" * 65)

    # =============================================================
    # COUNT OBJECT CATEGORIES
    # =============================================================

    def count_categories(self, detections):

        vehicles = 0
        vulnerable_users = 0
        traffic_controls = 0

        for detection in detections:

            class_name = detection["class"]

            if class_name in self.VEHICLE_CLASSES:
                vehicles += 1

            elif class_name in self.VULNERABLE_ROAD_USERS:
                vulnerable_users += 1

            elif class_name in self.TRAFFIC_CONTROL_CLASSES:
                traffic_controls += 1

        return {
            "vehicles": vehicles,
            "vulnerable_users": vulnerable_users,
            "traffic_controls": traffic_controls,
        }

    # =============================================================
    # DETERMINE TRAFFIC DENSITY
    # =============================================================

    def determine_traffic_density(self, vehicle_count):

        if vehicle_count == 0:
            return "empty"

        if vehicle_count <= 5:
            return "low"

        if vehicle_count <= 15:
            return "moderate"

        return "high"

    # =============================================================
    # DETERMINE SCENE TYPE
    # =============================================================

    def determine_scene_type(
        self,
        detections,
        vehicle_count,
        vulnerable_users,
        traffic_controls,
    ):

        classes = {
            detection["class"]
            for detection in detections
        }

        if "traffic light" in classes:
            return "traffic-controlled road"

        if "traffic sign" in classes:
            return "sign-controlled road"

        if vulnerable_users > 0:
            return "road with vulnerable road users"

        if vehicle_count > 15:
            return "high-density roadway"

        if vehicle_count > 0:
            return "roadway"

        return "unknown"

    # =============================================================
    # RISK INDICATORS
    # =============================================================

    def identify_risk_indicators(self, detections):

        risks = []

        for detection in detections:

            class_name = detection["class"]
            confidence = detection["confidence"]

            if class_name in self.VULNERABLE_ROAD_USERS:
                risks.append(
                    f"{class_name} detected "
                    f"(confidence={confidence:.2f})"
                )

            elif class_name == "traffic light":
                risks.append(
                    f"traffic light detected "
                    f"(confidence={confidence:.2f})"
                )

            elif class_name == "traffic sign":
                risks.append(
                    f"traffic sign detected "
                    f"(confidence={confidence:.2f})"
                )

        return risks

    # =============================================================
    # MAIN SCENE ANALYSIS
    # =============================================================

    def analyze(self, perception_result):

        detections = perception_result.get(
            "detections",
            []
        )

        summary = perception_result.get(
            "summary",
            {}
        )

        categories = self.count_categories(
            detections
        )

        vehicle_count = categories["vehicles"]
        vulnerable_users = categories["vulnerable_users"]
        traffic_controls = categories["traffic_controls"]

        traffic_density = self.determine_traffic_density(
            vehicle_count
        )

        scene_type = self.determine_scene_type(
            detections,
            vehicle_count,
            vulnerable_users,
            traffic_controls,
        )

        risk_indicators = self.identify_risk_indicators(
            detections
        )

        # ---------------------------------------------------------
        # Overall risk
        # ---------------------------------------------------------

        if vulnerable_users > 0:
            risk_level = "HIGH"

        elif traffic_controls > 0:
            risk_level = "MODERATE"

        elif vehicle_count > 15:
            risk_level = "MODERATE"

        else:
            risk_level = "LOW"

        return {
            "scene_type": scene_type,
            "traffic_density": traffic_density,
            "object_summary": summary,
            "vehicle_count": vehicle_count,
            "vulnerable_users": vulnerable_users,
            "traffic_controls": traffic_controls,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
        }

    # =============================================================
    # DISPLAY SCENE
    # =============================================================

    def display(self, scene):

        print()
        print("=" * 65)
        print("SCENE UNDERSTANDING RESULTS")
        print("=" * 65)

        print(
            f"Scene type       : "
            f"{scene['scene_type']}"
        )

        print(
            f"Traffic density  : "
            f"{scene['traffic_density']}"
        )

        print(
            f"Vehicle count    : "
            f"{scene['vehicle_count']}"
        )

        print(
            f"Vulnerable users : "
            f"{scene['vulnerable_users']}"
        )

        print(
            f"Traffic controls : "
            f"{scene['traffic_controls']}"
        )

        print(
            f"Risk level       : "
            f"{scene['risk_level']}"
        )

        print()
        print("OBJECT SUMMARY")
        print("-" * 65)

        for name, count in sorted(
            scene["object_summary"].items()
        ):
            print(
                f"{name:<25}: {count}"
            )

        print()
        print("RISK INDICATORS")
        print("-" * 65)

        if scene["risk_indicators"]:

            for risk in scene["risk_indicators"]:
                print(f"- {risk}")

        else:
            print("No major risk indicators detected.")

        print("=" * 65)


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    # Import the actual Perception Agent
    from perception import PerceptionAgent

    # -------------------------------------------------------------
    # Actual BDD100K validation dataset
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
    print("SCENE UNDERSTANDING TEST")
    print("=" * 65)

    print(
        f"BDD100K validation image: "
        f"{image_path.name}"
    )

    # -------------------------------------------------------------
    # Perception
    # -------------------------------------------------------------

    perception_agent = PerceptionAgent(
        confidence=0.25
    )

    perception_result = perception_agent.run(
        image_path
    )

    # -------------------------------------------------------------
    # Scene Understanding
    # -------------------------------------------------------------

    scene_agent = SceneUnderstandingAgent()

    scene = scene_agent.analyze(
        perception_result
    )

    scene_agent.display(scene)

    print()
    print("=" * 65)
    print("SCENE UNDERSTANDING AGENT COMPLETE")
    print("=" * 65)