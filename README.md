# Multi-Agent Autonomous Driving

### Confidence-Aware Risk-Based Multi-Agent Decision Fusion for Image-Based Autonomous Driving

An AI-based autonomous driving decision-support framework that combines object detection, scene understanding, planning, safety verification, tracking, trajectory prediction, and explainable risk fusion to generate driving actions.

## Project Overview

Autonomous driving systems must do more than detect objects. They must understand the road scene, assess potential risks, and make safe decisions.

This project implements a modular multi-agent framework using YOLOv8n and the BDD100K dataset. Its main contribution is a **Confidence-Aware Risk-Based Multi-Agent Decision Fusion** approach that combines information from multiple decision modules to produce an explainable final action:

- **GO** – proceed under normal conditions.
- **SLOW** – reduce speed or proceed cautiously.
- **STOP** – stop when safety conditions require it.

The current implementation is designed for **road-scene images**, rather than continuous real-time video.

## Key Features

- Road-object detection using YOLOv8n.
- Road-scene understanding and traffic-density classification.
- Rule-based driving action planning.
- Independent safety verification and override logic.
- Confidence-aware risk assessment and multi-agent decision fusion.
- Agent-conflict detection.
- Explainable decision outputs.
- Lightweight object tracking and constant-velocity trajectory prediction baselines.
- Streamlit-based interactive interface.
- Batch evaluation and result visualizations.

## System Architecture

```text
              BDD100K Road Images
                       |
                       v
                YOLOv8n Detection
                       |
                       v
              Scene Understanding
                       |
             +---------+---------+
             |                   |
             v                   v
       Planning Agent       Safety Agent
             |                   |
             +---------+---------+
                       |
                       v
             Confidence-Aware
               Risk Fusion
                       |
                       v
               Final Decision
                 GO / SLOW / STOP
                       |
                       v
              Explainability Agent

       Supporting modules:
       Tracking Agent and Trajectory Prediction Agent
```

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLO
- YOLOv8n
- BDD100K dataset
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Dataset

The project uses the BDD100K road-scene dataset.

Dataset split:

| Split | Images |
|---|---:|
| Training | 70,000 |
| Validation | 10,000 |
| Testing | 20,000 |

The object-detection annotations were converted into YOLO format for model training and evaluation.

**Note:** The BDD100K dataset is not included in this repository. Obtain it from its official source and follow its usage terms.

## Model Training and Results

YOLOv8n and YOLO11n were compared using the reported 50-epoch, 640-pixel training setup.

| Metric | YOLOv8n | YOLO11n |
|---|---:|---:|
| Precision | 0.681 | 0.679 |
| Recall | 0.319 | 0.315 |
| mAP@50 | 0.346 | 0.343 |
| mAP@50–95 | 0.195 | 0.194 |

YOLOv8n was selected because it achieved slightly higher values across all four reported metrics in this experiment. This result is specific to the experimental setup and does not establish that YOLOv8n is universally better than YOLO11n.

## Proposed Risk-Fusion Approach

The proposed risk-fusion module combines information from:

- Object detection confidence and uncertainty.
- Vulnerable road-user presence.
- Object velocity and proximity information.
- Trajectory predictions.
- Scene-level risk.
- Planning output.
- Safety verification.
- Conflicts between agent decisions.

The fusion module produces a normalized risk score and a final driving action.

**Important:** The risk score is a decision-support score. It is not a calibrated probability of collision.

## Risk-Fusion Evaluation

The risk-fusion ablation experiment processed 10,000 validation images with zero processing errors.

| Action | Baseline | Proposed |
|---|---:|---:|
| GO | 428 (4.28%) | 279 (2.79%) |
| SLOW | 7,278 (72.78%) | 5,068 (50.68%) |
| STOP | 2,294 (22.94%) | 4,653 (46.53%) |

The proposed fusion approach shifted more decisions toward STOP and fewer toward GO and SLOW in this experiment.

These figures demonstrate a change in decision behaviour; they do not independently prove improved real-world safety. The dataset does not provide ground-truth GO, SLOW, and STOP labels.

## Project Modules

| Module | Responsibility |
|---|---|
| Perception Agent | Detects road objects using YOLOv8n |
| Scene Understanding Agent | Interprets objects and road-scene conditions |
| Planning Agent | Generates an initial driving action |
| Safety Agent | Verifies or overrides the planned action |
| Risk Fusion Agent | Combines agent outputs into a final risk-aware decision |
| Explainability Agent | Provides reasons for the final decision |
| Tracker Agent | Maintains object identities across supplied frames |
| Trajectory Prediction Agent | Estimates future object positions using a constant-velocity baseline |

## Project Structure

```text
MultiAgent-Autonomous-Driving/
│
├── Agents/
│   ├── perception.py
│   ├── scene_agent.py
│   ├── planning_agent.py
│   ├── safety_agent.py
│   ├── explainability_agent.py
│   ├── tracker_agent.py
│   ├── trajectory_agent.py
│   └── risk_fusion_agent.py
│
├── weights/
│
├── results/
│
├── main.py
├── demo.py
├── web_app.py
├── evaluate_batch.py
├── evaluate_risk_fusion.py
├── test_agent_conflict.py
├── generate_final_report.py
├── generate_results_graphs.py
├── generate_ablation_graphs.py
├── generate_ablation_report.py
│
├── requirements.txt
└── README.md
```

*The structure above is a guide; adjust it to match the files actually included in your repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/multi-agent-autonomous-driving.git
cd multi-agent-autonomous-driving
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the Streamlit interface:

```bash
python -m streamlit run web_app.py
```

Open the local URL displayed in your terminal.

Make sure the required model weights and any expected input images are available at the paths configured in the project.



## Research Contribution

This project proposes a confidence-aware, risk-based multi-agent decision-fusion framework for image-based autonomous driving, in which perception confidence, scene risk, planning output, safety verification, and agent conflicts are jointly considered to generate an explainable final driving action.


Dataset: BDD100K. Please refer to the dataset's official documentation and license for usage conditions.

---
