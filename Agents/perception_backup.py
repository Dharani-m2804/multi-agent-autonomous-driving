from pathlib import Path
from ultralytics import YOLO
import torch
import cv2


class PerceptionAgent:
    """
    Perception Agent

    Uses the trained YOLOv8 model on BDD100K images.

    Outputs:
        1. Structured object detections
        2. Object summary
        3. Annotated image with bounding boxes,
           class labels and confidence scores
    """

    # =============================================================
    # BDD100K CLASS NAMES
    # =============================================================

    CLASS_NAMES = {
        0: "bicycle",
        1: "bus",
        2: "car",
        3: "motorcycle",
        4: "other person",
        5: "other vehicle",
        6: "pedestrian",
        7: "rider",
        8: "traffic light",
        9: "traffic sign",
        10: "trailer",
        11: "train",
        12: "truck",
    }

    # =============================================================
    # INITIALIZATION
    # =============================================================

    def __init__(self, model_path=None, confidence=0.25):

        self.project_root = (
            Path(__file__).resolve().parent.parent
        )

        # =========================================================
        # DEVICE
        # =========================================================

        if torch.cuda.is_available():

            self.device = 0
            self.device_name = (
                torch.cuda.get_device_name(0)
            )

        else:

            self.device = "cpu"
            self.device_name = "CPU"

        print()
        print("=" * 65)
        print("INITIALIZING PERCEPTION AGENT")
        print("=" * 65)

        if self.device == 0:

            print("Device: NVIDIA GPU")
            print(f"GPU: {self.device_name}")
            print(
                f"CUDA version: "
                f"{torch.version.cuda}"
            )

        else:

            print("Device: CPU")

        # =========================================================
        # MODEL PATH
        # =========================================================

        if model_path is None:

            model_path = (
                self.project_root
                / "runs"
                / "detect"
                / "runs"
                / "bdd100k_yolov8n-2"
                / "weights"
                / "best.pt"
            )

        self.model_path = Path(model_path)
        self.confidence = confidence

        print()
        print("YOLO model:")
        print(self.model_path)

        print(
            f"Confidence threshold: "
            f"{self.confidence}"
        )

        # =========================================================
        # CHECK MODEL
        # =========================================================

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"\nYOLO model not found:\n"
                f"{self.model_path}"
            )

        # =========================================================
        # LOAD MODEL
        # =========================================================

        print()
        print("Loading trained YOLO model...")

        self.model = YOLO(
            str(self.model_path)
        )

        print(
            "YOLO model loaded successfully."
        )

        print("=" * 65)

    # =============================================================
    # DETECTION
    # =============================================================

    def detect(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(
                f"\nBDD100K image not found:\n"
                f"{image_path}"
            )

        print()
        print("=" * 65)
        print("RUNNING YOLO PERCEPTION")
        print("=" * 65)

        print("BDD100K image:")
        print(image_path)

        print(
            f"Device: {self.device_name}"
        )

        print("Image size: 640")

        print(
            f"Confidence threshold: "
            f"{self.confidence}"
        )

        # =========================================================
        # YOLO PREDICTION
        # =========================================================

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False
        )

        detections = []

        if not results:

            return detections, None

        result = results[0]

        if result.boxes is None:

            return detections, result

        # =========================================================
        # EXTRACT DETECTIONS
        # =========================================================

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            class_name = (
                self.CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                )
            )

            detections.append(
                {
                    "class_id": class_id,

                    "class": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                }
            )

        return detections, result

    # =============================================================
    # CREATE ANNOTATED IMAGE
    # =============================================================

    def save_annotated_image(
        self,
        image_path,
        result,
        detections
    ):

        image_path = Path(
            image_path
        )

        # =========================================================
        # OUTPUT DIRECTORY
        # =========================================================

        output_dir = (
            self.project_root
            / "results"
            / "perception_visualizations"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =========================================================
        # OUTPUT FILE
        # =========================================================

        output_path = (
            output_dir
            / f"{image_path.stem}_detected.jpg"
        )

        # =========================================================
        # LOAD ORIGINAL IMAGE
        # =========================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Unable to read image:\n"
                f"{image_path}"
            )

        # =========================================================
        # DRAW BOUNDING BOXES
        # =========================================================

        for detection in detections:

            x1, y1, x2, y2 = map(
                int,
                detection["bbox"]
            )

            class_name = (
                detection["class"]
            )

            confidence = (
                detection["confidence"]
            )

            # -----------------------------------------------------
            # Bounding box
            # -----------------------------------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # -----------------------------------------------------
            # Label
            # -----------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.7

            thickness = 2

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # -----------------------------------------------------
            # Label background
            # -----------------------------------------------------

            label_y1 = max(
                0,
                y1 - text_height - baseline - 8
            )

            label_y2 = y1

            cv2.rectangle(
                image,
                (
                    x1,
                    label_y1
                ),
                (
                    x1 + text_width + 8,
                    label_y2
                ),
                (0, 255, 0),
                -1
            )

            # -----------------------------------------------------
            # Label text
            # -----------------------------------------------------

            cv2.putText(
                image,
                label,
                (
                    x1 + 4,
                    y1 - 6
                ),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # =========================================================
        # SAVE IMAGE
        # =========================================================

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if not success:

            raise RuntimeError(
                f"Failed to save annotated image:\n"
                f"{output_path}"
            )

        return output_path

    # =============================================================
    # SUMMARY
    # =============================================================

    def summarize(self, detections):

        summary = {}

        for detection in detections:

            class_name = (
                detection["class"]
            )

            if class_name not in summary:

                summary[class_name] = 0

            summary[class_name] += 1

        return summary

    # =============================================================
    # RUN PERCEPTION
    # =============================================================

    def run(self, image_path):

        detections, yolo_result = (
            self.detect(image_path)
        )

        summary = self.summarize(
            detections
        )

        # =========================================================
        # SAVE VISUALIZATION
        # =========================================================

        annotated_image = None

        if yolo_result is not None:

            annotated_image = (
                self.save_annotated_image(
                    image_path,
                    yolo_result,
                    detections
                )
            )

        # =========================================================
        # RETURN RESULT
        # =========================================================

        return {

            "image": str(
                image_path
            ),

            "num_objects": len(
                detections
            ),

            "detections": detections,

            "summary": summary,

            "annotated_image": (
                str(annotated_image)
                if annotated_image
                else None
            )
        }


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":

    # =============================================================
    # PROJECT ROOT
    # =============================================================

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    # =============================================================
    # BDD100K VALIDATION DATASET
    # =============================================================

    dataset_root = (
        project_root
        / "Dataset"
        / "bdd100k"
        / "bdd100k"
        / "images"
        / "100k"
        / "val"
    )

    # =============================================================
    # CHECK DATASET
    # =============================================================

    if not dataset_root.exists():

        raise FileNotFoundError(
            f"\nBDD100K validation directory "
            f"not found:\n"
            f"{dataset_root}"
        )

    # =============================================================
    # FIND IMAGES
    # =============================================================

    images = sorted(
        dataset_root.glob("*.jpg")
    )

    if not images:

        raise FileNotFoundError(
            f"\nNo BDD100K images found in:\n"
            f"{dataset_root}"
        )

    # =============================================================
    # SELECT FIRST IMAGE
    # =============================================================

    image_path = images[0]

    print()
    print("=" * 65)
    print("BDD100K DATASET INPUT")
    print("=" * 65)

    print(
        "Dataset split : VALIDATION"
    )

    print(
        f"Images found  : "
        f"{len(images):,}"
    )

    print(
        f"Selected image: "
        f"{image_path.name}"
    )

    print("=" * 65)

    # =============================================================
    # CREATE AGENT
    # =============================================================

    agent = PerceptionAgent(
        confidence=0.25
    )

    # =============================================================
    # RUN
    # =============================================================

    result = agent.run(
        image_path
    )

    # =============================================================
    # RESULTS
    # =============================================================

    print()
    print("=" * 65)
    print("PERCEPTION AGENT RESULTS")
    print("=" * 65)

    print()
    print("Dataset image:")
    print(result["image"])

    print()
    print(
        f"Total objects detected: "
        f"{result['num_objects']}"
    )

    # =============================================================
    # OBJECT SUMMARY
    # =============================================================

    print()
    print("OBJECT SUMMARY")
    print("-" * 65)

    if result["summary"]:

        for class_name, count in sorted(
            result["summary"].items()
        ):

            print(
                f"{class_name:<25}: "
                f"{count}"
            )

    else:

        print(
            "No objects detected."
        )

    # =============================================================
    # DETECTIONS
    # =============================================================

    print()
    print("DETECTIONS")
    print("-" * 65)

    if result["detections"]:

        for i, detection in enumerate(
            result["detections"],
            start=1
        ):

            print(
                f"{i:03d}. "
                f"{detection['class']:<20} "
                f"confidence="
                f"{detection['confidence']:.3f} "
                f"bbox="
                f"{detection['bbox']}"
            )

    else:

        print(
            "No detections."
        )

    # =============================================================
    # VISUAL OUTPUT
    # =============================================================

    print()
    print("VISUAL OUTPUT")
    print("-" * 65)

    if result["annotated_image"]:

        print(
            "Bounding-box image saved:"
        )

        print(
            result["annotated_image"]
        )

    else:

        print(
            "No annotated image generated."
        )

    # =============================================================
    # COMPLETE
    # =============================================================

    print()
    print("=" * 65)
    print(
        "PERCEPTION AGENT COMPLETE"
    )
    print("=" * 65)