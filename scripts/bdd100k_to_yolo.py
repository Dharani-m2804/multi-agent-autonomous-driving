from pathlib import Path
import json
import shutil


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "Dataset"

IMAGE_ROOT = (
    DATASET_ROOT
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
)

ANNOTATION_ROOT = DATASET_ROOT / "labels"

OUTPUT_ROOT = DATASET_ROOT / "BDD100K_YOLO"


# ============================================================
# BDD100K CLASSES
# ============================================================

CLASS_NAMES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "other person",
    "other vehicle",
    "pedestrian",
    "rider",
    "traffic light",
    "traffic sign",
    "trailer",
    "train",
    "truck",
]

CLASS_MAP = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# IMAGE SIZE
# BDD100K images are 1280 x 720
# ============================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    for split in ["train", "val"]:

        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CONVERT ONE ANNOTATION RECORD
# ============================================================

def convert_record(record, split):

    image_name = record["name"]

    image_source = (
        IMAGE_ROOT
        / split
        / image_name
    )

    if not image_source.exists():

        print(
            f"[WARNING] Image not found: {image_name}"
        )

        return False

    image_output = (
        OUTPUT_ROOT
        / "images"
        / split
        / image_name
    )

    label_output = (
        OUTPUT_ROOT
        / "labels"
        / split
        / f"{Path(image_name).stem}.txt"
    )

    yolo_lines = []

    for obj in record.get("labels", []):

        category = obj.get("category")

        # Ignore classes outside our 13-class mapping
        if category not in CLASS_MAP:
            continue

        box = obj.get("box2d")

        if not box:
            continue

        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])

        # ----------------------------------------------------
        # Clamp coordinates to image boundaries
        # ----------------------------------------------------

        x1 = max(0.0, min(x1, IMAGE_WIDTH))
        y1 = max(0.0, min(y1, IMAGE_HEIGHT))
        x2 = max(0.0, min(x2, IMAGE_WIDTH))
        y2 = max(0.0, min(y2, IMAGE_HEIGHT))

        # ----------------------------------------------------
        # Ignore invalid boxes
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # Convert BDD100K box → YOLO format
        #
        # YOLO:
        # class center_x center_y width height
        #
        # Values normalized between 0 and 1
        # ----------------------------------------------------

        center_x = ((x1 + x2) / 2) / IMAGE_WIDTH
        center_y = ((y1 + y2) / 2) / IMAGE_HEIGHT

        width = (x2 - x1) / IMAGE_WIDTH
        height = (y2 - y1) / IMAGE_HEIGHT

        class_id = CLASS_MAP[category]

        yolo_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # --------------------------------------------------------
    # Write label file
    # --------------------------------------------------------

    label_output.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    if not image_output.exists():

        shutil.copy2(
            image_source,
            image_output
        )

    return True


# ============================================================
# PROCESS JSON FILE
# ============================================================

def process_split(split, annotation_file):

    annotation_path = (
        ANNOTATION_ROOT
        / annotation_file
    )

    print()
    print("=" * 70)
    print(f"PROCESSING {split.upper()} DATASET")
    print("=" * 70)

    print(f"Annotation file:")
    print(annotation_path)

    if not annotation_path.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n{annotation_path}"
        )

    print("Loading JSON...")

    with open(
        annotation_path,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    print(
        f"Annotation records: {len(records):,}"
    )

    successful = 0
    failed = 0

    for index, record in enumerate(records, start=1):

        try:

            success = convert_record(
                record,
                split
            )

            if success:
                successful += 1
            else:
                failed += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {record.get('name')}: {e}"
            )

        # Progress
        if index % 1000 == 0:

            print(
                f"Progress: {index:,}/{len(records):,} "
                f"({index / len(records) * 100:.1f}%)"
            )

    print()
    print(
        f"{split.upper()} conversion complete."
    )

    print(
        f"Successful: {successful:,}"
    )

    print(
        f"Failed:     {failed:,}"
    )

    return successful


# ============================================================
# CREATE data.yaml
# ============================================================

def create_yaml():

    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}

train: images/train
val: images/val

nc: {len(CLASS_NAMES)}

names:
"""

    for index, name in enumerate(CLASS_NAMES):

        yaml_text += (
            f"  {index}: {name}\n"
        )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("YOLO DATASET CONFIGURATION")
    print("=" * 70)
    print(yaml_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BDD100K → YOLO DATASET CONVERTER")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Dataset root:")
    print(DATASET_ROOT)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_count = process_split(
        "train",
        "det_v2_train_release.json"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_count = process_split(
        "val",
        "det_v2_val_release.json"
    )

    # --------------------------------------------------------
    # data.yaml
    # --------------------------------------------------------

    create_yaml()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION FINISHED")
    print("=" * 70)

    print(
        f"Training records converted   : {train_count:,}"
    )

    print(
        f"Validation records converted : {val_count:,}"
    )

    print()
    print("YOLO dataset:")
    print(OUTPUT_ROOT)

    print()
    print("Classes:")
    
    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{index:2d} -> {name}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
      