import time
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# MULTI-AGENT AUTONOMOUS DRIVING - FINAL DEMO
# ============================================================

PROJECT_ROOT = Path(r"F:\MultiAgent-Autonomous-Driving")

# ------------------------------------------------------------
# BDD100K VALIDATION DATASET
# ------------------------------------------------------------

DATASET_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "BDD100K_Extracted"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "val"
)

# ------------------------------------------------------------
# TRAINED YOLO MODEL
# ------------------------------------------------------------

MODEL_PATH = Path(
    r"C:\Users\mdhar\Documents\MultiAgent-Autonomous-Driving"
    r"\runs\bdd100k_yolov8n\weights\best.pt"
)

# ------------------------------------------------------------
# FIXED DEMO IMAGE
# ------------------------------------------------------------

DEMO_IMAGE_NAME = "b1c66a42-6f7d68ca.jpg"

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DIR = RESULTS_DIR / "demo"

CONFIDENCE = 0.25


# ============================================================
# HEADER
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SCENE UNDERSTANDING AGENT
# ============================================================

def understand_scene(objects):

    vehicle_classes = {
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "other vehicle",
        "trailer",
        "train",
    }

    vulnerable_classes = {
        "pedestrian",
        "rider",
        "other person",
    }

    traffic_control_classes = {
        "traffic light",
        "traffic sign",
    }

    vehicle_count = sum(
        1
        for obj in objects
        if obj["class"] in vehicle_classes
    )

    vulnerable_count = sum(
        1
        for obj in objects
        if obj["class"] in vulnerable_classes
    )

    traffic_control_count = sum(
        1
        for obj in objects
        if obj["class"] in traffic_control_classes
    )

    total_objects = len(objects)

    # --------------------------------------------------------
    # Scene classification
    # --------------------------------------------------------

    if vulnerable_count > 0:

        scene_type = "road with vulnerable road users"

    elif traffic_control_count > 0:

        if any(
            obj["class"] == "traffic light"
            for obj in objects
        ):
            scene_type = "traffic-controlled road"
        else:
            scene_type = "sign-controlled road"

    elif vehicle_count >= 15:

        scene_type = "high-density roadway"

    elif vehicle_count > 0:

        scene_type = "roadway"

    else:

        scene_type = "unknown"

    # --------------------------------------------------------
    # Traffic density
    # --------------------------------------------------------

    if vehicle_count >= 15:

        traffic_density = "high"

    elif vehicle_count >= 5:

        traffic_density = "medium"

    else:

        traffic_density = "low"

    # --------------------------------------------------------
    # Risk estimation
    # --------------------------------------------------------

    if vulnerable_count >= 3:

        risk_level = "HIGH"

    elif vehicle_count >= 10:

        risk_level = "HIGH"

    elif vulnerable_count > 0:

        risk_level = "MODERATE"

    elif vehicle_count >= 5:

        risk_level = "MODERATE"

    else:

        risk_level = "LOW"

    return {
        "scene_type": scene_type,
        "traffic_density": traffic_density,
        "vehicle_count": vehicle_count,
        "vulnerable_count": vulnerable_count,
        "traffic_control_count": traffic_control_count,
        "risk_level": risk_level,
        "total_objects": total_objects,
    }


# ============================================================
# PLANNING AGENT
# ============================================================

def planning_agent(scene):

    risk = scene["risk_level"]
    vehicles = scene["vehicle_count"]

    if risk == "HIGH":

        action = "STOP"
        speed = "ZERO"
        priority = "HIGH"

        reason = (
            "High-risk scene detected. "
            "Stop the vehicle and wait for a safer condition."
        )

    elif risk == "MODERATE":

        action = "SLOW"
        speed = "MEDIUM"
        priority = "MEDIUM"

        reason = (
            f"Moderate traffic risk detected with "
            f"{vehicles} vehicle(s). Maintain a reduced "
            "and controlled speed."
        )

    else:

        action = "GO"
        speed = "NORMAL"
        priority = "LOW"

        reason = (
            "Low-risk roadway detected. "
            "Continue with normal driving."
        )

    return {
        "action": action,
        "speed": speed,
        "priority": priority,
        "reason": reason,
    }


# ============================================================
# SAFETY AGENT
# ============================================================

def safety_agent(scene, plan):

    risk = scene["risk_level"]
    planned_action = plan["action"]

    reasons = []

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if risk == "HIGH":

        if planned_action != "STOP":

            decision = "OVERRIDE"
            final_action = "STOP"

            reasons.append(
                "High-risk scene requires an emergency stop."
            )

        else:

            decision = "APPROVE"
            final_action = "STOP"

            reasons.append(
                "High-risk scene detected."
            )

    # --------------------------------------------------------
    # MODERATE RISK
    # --------------------------------------------------------

    elif risk == "MODERATE":

        if planned_action == "GO":

            decision = "OVERRIDE"
            final_action = "SLOW"

            reasons.append(
                "Moderate risk does not permit unrestricted GO."
            )

        else:

            decision = "APPROVE"
            final_action = planned_action

            reasons.append(
                "Moderate traffic risk detected."
            )

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        if planned_action == "STOP":

            decision = "OVERRIDE"
            final_action = "SLOW"

            reasons.append(
                "Low-risk scene does not require a full stop."
            )

        else:

            decision = "APPROVE"
            final_action = planned_action

            reasons.append(
                "Low-risk scene and planner decision are consistent."
            )

    return {
        "decision": decision,
        "final_action": final_action,
        "safety_level": (
            "EMERGENCY"
            if risk == "HIGH"
            else "CAUTION"
            if risk == "MODERATE"
            else "NORMAL"
        ),
        "reasons": reasons,
    }


# ============================================================
# EXPLAINABILITY AGENT
# ============================================================

def explain_decision(scene, plan, safety):

    explanation = (
        f"The perception system detected "
        f"{scene['total_objects']} object(s). "
        f"The scene was classified as "
        f"'{scene['scene_type']}'. "
        f"Traffic density was classified as "
        f"'{scene['traffic_density']}'. "
        f"The estimated scene risk level was "
        f"'{scene['risk_level']}'. "
        f"The scene contained "
        f"{scene['vehicle_count']} vehicle(s), "
        f"{scene['vulnerable_count']} vulnerable road user(s), "
        f"and "
        f"{scene['traffic_control_count']} traffic control object(s). "
        f"The Planning Agent proposed "
        f"'{plan['action']}'. "
        f"Planning reason: {plan['reason']} "
        f"The Safety Agent returned "
        f"'{safety['decision']}' with safety level "
        f"'{safety['safety_level']}'. "
        f"Therefore, the final recommended driving action is "
        f"'{safety['final_action']}'."
    )

    return explanation


# ============================================================
# PERCEPTION AGENT
# ============================================================

def run_perception(model, image_path):

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]

    objects = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            objects.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            })

    return objects


# ============================================================
# VISUALIZATION
# ============================================================

def draw_detections(
    image_path,
    objects,
    scene,
    plan,
    safety,
):

    image = cv2.imread(str(image_path))

    if image is None:

        raise RuntimeError(
            f"Unable to read image:\n{image_path}"
        )

    # --------------------------------------------------------
    # DRAW BOUNDING BOXES
    # --------------------------------------------------------

    for obj in objects:

        x1, y1, x2, y2 = map(
            int,
            obj["bbox"]
        )

        label = (
            f"{obj['class']} "
            f"{obj['confidence']:.2f}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    panel_height = 220

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1], panel_height),
        (0, 0, 0),
        -1,
    )

    image = cv2.addWeighted(
        overlay,
        0.65,
        image,
        0.35,
        0,
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_lines = [
        "MULTI-AGENT AUTONOMOUS DRIVING",
        f"Objects: {len(objects)}",
        f"Scene: {scene['scene_type']}",
        f"Risk: {scene['risk_level']}",
        f"Planner: {plan['action']}",
        f"Safety: {safety['decision']}",
        f"FINAL ACTION: {safety['final_action']}",
    ]

    y = 30

    for index, text in enumerate(text_lines):

        scale = 0.75 if index == 0 else 0.60

        cv2.putText(
            image,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2,
        )

        y += 28

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "BDD100K MULTI-AGENT AUTONOMOUS DRIVING DEMONSTRATION"
    )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    print()
    print("Checking dataset...")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"BDD100K validation directory not found:\n"
            f"{DATASET_DIR}"
        )

    images = sorted(
        DATASET_DIR.glob("*.jpg")
    )

    print(
        f"Validation images available : "
        f"{len(images):,}"
    )

    # ========================================================
    # FIXED IMAGE
    # ========================================================

    image_path = DATASET_DIR / DEMO_IMAGE_NAME

    if not image_path.exists():

        raise FileNotFoundError(
            f"Selected BDD100K image not found:\n"
            f"{image_path}"
        )

    print(
        f"Selected image              : "
        f"{image_path.name}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading YOLO model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n"
            f"{MODEL_PATH}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "YOLO model loaded successfully."
    )

    # ========================================================
    # PERCEPTION
    # ========================================================

    header("PERCEPTION AGENT")

    start_time = time.perf_counter()

    objects = run_perception(
        model,
        image_path,
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Objects detected : "
        f"{len(objects)}"
    )

    print(
        f"Processing time  : "
        f"{processing_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # OBJECT SUMMARY
    # --------------------------------------------------------

    object_counts = {}

    for obj in objects:

        class_name = obj["class"]

        object_counts[class_name] = (
            object_counts.get(
                class_name,
                0
            )
            + 1
        )

    print()
    print("OBJECT SUMMARY")

    for name, count in sorted(
        object_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {name:<20}: "
            f"{count}"
        )

    # ========================================================
    # SCENE UNDERSTANDING
    # ========================================================

    header(
        "SCENE UNDERSTANDING AGENT"
    )

    scene = understand_scene(
        objects
    )

    print(
        f"Scene type           : "
        f"{scene['scene_type']}"
    )

    print(
        f"Traffic density      : "
        f"{scene['traffic_density']}"
    )

    print(
        f"Vehicle count        : "
        f"{scene['vehicle_count']}"
    )

    print(
        f"Vulnerable users     : "
        f"{scene['vulnerable_count']}"
    )

    print(
        f"Traffic controls     : "
        f"{scene['traffic_control_count']}"
    )

    print(
        f"Risk level           : "
        f"{scene['risk_level']}"
    )

    # ========================================================
    # PLANNING
    # ========================================================

    header(
        "PLANNING AGENT"
    )

    plan = planning_agent(
        scene
    )

    print(
        f"Action               : "
        f"{plan['action']}"
    )

    print(
        f"Speed level          : "
        f"{plan['speed']}"
    )

    print(
        f"Priority             : "
        f"{plan['priority']}"
    )

    print()
    print("Reason:")
    print(
        plan["reason"]
    )

    # ========================================================
    # SAFETY
    # ========================================================

    header(
        "SAFETY AGENT"
    )

    safety = safety_agent(
        scene,
        plan,
    )

    print(
        f"Planner action       : "
        f"{plan['action']}"
    )

    print(
        f"Safety decision      : "
        f"{safety['decision']}"
    )

    print(
        f"Safety level         : "
        f"{safety['safety_level']}"
    )

    print(
        f"Final action         : "
        f"{safety['final_action']}"
    )

    print()
    print("Safety reasoning:")

    for reason in safety["reasons"]:

        print(
            f"  - {reason}"
        )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    header(
        "EXPLAINABILITY AGENT"
    )

    explanation = explain_decision(
        scene,
        plan,
        safety,
    )

    print()
    print(
        explanation
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    DEMO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    header(
        "GENERATING VISUAL RESULT"
    )

    annotated_image = draw_detections(
        image_path,
        objects,
        scene,
        plan,
        safety,
    )

    output_image = (
        DEMO_DIR
        / f"{image_path.stem}_demo.jpg"
    )

    success = cv2.imwrite(
        str(output_image),
        annotated_image,
    )

    if not success:

        raise RuntimeError(
            f"Could not save image:\n"
            f"{output_image}"
        )

    # ========================================================
    # CREATE JSON RESULT
    # ========================================================

    output_json = (
        DEMO_DIR
        / f"{image_path.stem}_demo.json"
    )

    result_data = {

        "project":
            "Multi-Agent Autonomous Driving",

        "dataset":
            "BDD100K",

        "split":
            "validation",

        "image":
            image_path.name,

        "image_path":
            str(image_path),

        "model":
            str(MODEL_PATH),

        "confidence":
            CONFIDENCE,

        "processing_time":
            processing_time,

        "objects_detected":
            len(objects),

        "object_summary":
            object_counts,

        "detections":
            objects,

        "scene":
            scene,

        "planning":
            plan,

        "safety":
            safety,

        "final_action":
            safety["final_action"],

        "explanation":
            explanation,
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_data,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    header(
        "FINAL AUTONOMOUS DRIVING DECISION"
    )

    print(
        f"Image                : "
        f"{image_path.name}"
    )

    print(
        f"Objects detected     : "
        f"{len(objects)}"
    )

    print(
        f"Scene                : "
        f"{scene['scene_type']}"
    )

    print(
        f"Traffic density      : "
        f"{scene['traffic_density']}"
    )

    print(
        f"Risk level           : "
        f"{scene['risk_level']}"
    )

    print(
        f"Planner action       : "
        f"{plan['action']}"
    )

    print(
        f"Safety decision      : "
        f"{safety['decision']}"
    )

    print()
    print(
        "-" * 70
    )

    print(
        f"FINAL DRIVING ACTION: "
        f"{safety['final_action']}"
    )

    print(
        "-" * 70
    )

    print()
    print("EXPLANATION")

    print(
        explanation
    )

    print()
    print(
        "Visual result saved:"
    )

    print(
        output_image
    )

    print()
    print(
        "JSON result saved:"
    )

    print(
        output_json
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DEMO COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()