from pathlib import Path
import sys
import json
import time
import random

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
    / "test"
)

RESULTS_DIR = PROJECT_ROOT / "results"
WEB_RESULTS_DIR = RESULTS_DIR / "web_results"
VIS_DIR = RESULTS_DIR / "perception_visualizations"

WEB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT YOUR EXISTING AGENTS
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT))

from Agents.perception import PerceptionAgent
from Agents.scene_agent import SceneUnderstandingAgent
from Agents.planning_agent import PlanningAgent
from Agents.safety_agent import SafetyAgent
from Agents.explainability_agent import ExplainabilityAgent
from Agents.tracker_agent import TrackerAgent
from Agents.trajectory_agent import TrajectoryPredictionAgent
from Agents.risk_fusion_agent import RiskFusionAgent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Autonomous Driving",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #080d19;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 25px;
        padding-bottom: 60px;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .app-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 5px;
    }

    .app-subtitle {
        text-align: center;
        color: #9ba8bf;
        font-size: 17px;
        margin-bottom: 20px;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3 {
        color: white !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #111a2c;
        border: 1px solid #293752;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #8e9ab0 !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* ========================================================
       CONTAINERS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111a2c;
        border-color: #293752 !important;
        border-radius: 15px;
    }

    /* ========================================================
       TABS
       ======================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 15px;
    }

    /* ========================================================
       IMAGE
       ======================================================== */

    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #26344e;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {
        text-align: center;
        color: #66748d;
        font-size: 12px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🚗 Multi-Agent Autonomous Driving</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Explainable AI-Based Driving Decision & Risk-Fusion System'
    '</div>',
    unsafe_allow_html=True
)

# Native Streamlit status row
status1, status2, status3, status4 = st.columns(4)

with status1:
    st.success("🟢 System Ready")

with status2:
    st.info("🤖 8 AI Modules")

with status3:
    st.info("🎯 YOLOv8")

with status4:
    st.info("🛡️ Safety Verified")


# ============================================================
# CHECK TEST DATASET
# ============================================================

if not TEST_DIR.exists():

    st.error("❌ BDD100K test folder was not found.")

    st.code(str(TEST_DIR))

    st.stop()


# ============================================================
# LOAD TEST IMAGES
# ============================================================

@st.cache_data
def load_test_images():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    images = [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ]

    images.sort(
        key=lambda x: x.name.lower()
    )

    return images


test_images = load_test_images()


if not test_images:

    st.error("❌ No BDD100K test images found.")

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ============================================================
# INITIALIZE AGENTS
# ============================================================

@st.cache_resource
def initialize_agents():

    perception = PerceptionAgent(
        confidence=0.25
    )

    scene = SceneUnderstandingAgent()

    planning = PlanningAgent()

    safety = SafetyAgent()

    explainability = ExplainabilityAgent()
    tracker = TrackerAgent()
    trajectory = TrajectoryPredictionAgent()
    risk_fusion = RiskFusionAgent()

    return (
        perception,
        scene,
        planning,
        safety,
        explainability,
        tracker,
        trajectory,
        risk_fusion
    )


try:

    (
        perception_agent,
        scene_agent,
        planning_agent,
        safety_agent,
        explainability_agent,
        tracker_agent,
        trajectory_agent,
        risk_fusion_agent
    ) = initialize_agents()

except Exception as error:

    st.error("❌ Failed to initialize AI agents.")

    st.exception(error)

    st.stop()


# ============================================================
# CURRENT IMAGE
# ============================================================

current_image = test_images[
    st.session_state.image_index
]


# ============================================================
# TEST SCENE SECTION
# ============================================================

st.divider()

st.header("🎬 Test Scene Selection")

st.caption(
    "Choose a road scene directly from the BDD100K test dataset. "
    "No folder browsing is required."
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5 = st.columns(
    [1, 1, 3, 1, 1]
)


with nav1:

    if st.button(
        "⬅ Previous",
        use_container_width=True
    ):

        st.session_state.image_index = (
            st.session_state.image_index - 1
        ) % len(test_images)

        st.session_state.analysis_result = None

        st.rerun()


with nav2:

    if st.button(
        "Next ➡",
        use_container_width=True
    ):

        st.session_state.image_index = (
            st.session_state.image_index + 1
        ) % len(test_images)

        st.session_state.analysis_result = None

        st.rerun()


with nav3:

    st.info(
        f"📷 **{current_image.name}**  |  "
        f"Image {st.session_state.image_index + 1:,} "
        f"of {len(test_images):,}"
    )


with nav4:

    if st.button(
        "🎲 Random",
        use_container_width=True
    ):

        st.session_state.image_index = random.randrange(
            len(test_images)
        )

        st.session_state.analysis_result = None

        st.rerun()


with nav5:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):

        st.session_state.image_index = 0
        st.session_state.analysis_result = None

        st.rerun()


# ============================================================
# IMAGE + PIPELINE
# ============================================================

image_col, pipeline_col = st.columns(
    [2.3, 1]
)


# ============================================================
# ROAD IMAGE
# ============================================================

with image_col:

    with st.container(border=True):

        st.subheader("📷 Selected Road Scene")

        st.image(
            str(current_image),
            use_container_width=True
        )

        st.caption(
            f"BDD100K Test Image • {current_image.name}"
        )


# ============================================================
# AI PIPELINE
# ============================================================

with pipeline_col:

    with st.container(border=True):

        st.subheader("🧠 AI Pipeline")

        st.write(
            "The selected scene is processed through "
            "eight cooperating AI modules."
        )

        st.write("")

        pipeline_items = [
            ("🔍 1. Perception", "YOLOv8n detects road objects."),
            ("🧠 2. Scene Understanding", "Interprets traffic density, scene type and risk."),
            ("🎯 3. Tracking", "Assigns object identities in the current frame."),
            ("📈 4. Trajectory", "Provides a single-frame trajectory baseline."),
            ("🗺️ 5. Planning", "Generates the recommended driving action."),
            ("🛡️ 6. Safety", "Checks the planned action against safety rules."),
            ("⚖️ 7. Risk Fusion", "Combines confidence, risk and agent evidence."),
            ("💡 8. Explainability", "Explains the final decision and its evidence.")
        ]

        for title, description in pipeline_items:
            st.markdown(f"**{title}**")
            st.caption(description)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

left, center, right = st.columns(
    [1, 2, 1]
)

with center:

    analyze_button = st.button(
        "🚀 ANALYZE DRIVING SCENE",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RUN PIPELINE
# ============================================================

if analyze_button:

    start_time = time.time()

    progress = st.progress(0)

    status = st.empty()


    try:

        # ----------------------------------------------------
        # PERCEPTION
        # ----------------------------------------------------

        status.info(
            "🔍 Perception Agent is detecting objects..."
        )

        perception_result = perception_agent.run(
            str(current_image)
        )

        progress.progress(20)


        # ----------------------------------------------------
        # SCENE
        # ----------------------------------------------------

        status.info(
            "🧠 Scene Understanding Agent is analyzing the environment..."
        )

        scene_result = scene_agent.analyze(
            perception_result
        )

        progress.progress(40)


        # ----------------------------------------------------
        # PLANNING
        # ----------------------------------------------------

        status.info(
            "🗺️ Planning Agent is generating the driving action..."
        )

        planning_result = planning_agent.plan(
            scene_result
        )

        progress.progress(60)


        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        status.info(
            "🛡️ Safety Agent is verifying the decision..."
        )

        safety_result = safety_agent.verify(
            scene_result,
            planning_result
        )

        progress.progress(80)


        # ----------------------------------------------------
        # TRACKING / SINGLE-FRAME TEMPORAL BASELINE
        # ----------------------------------------------------

        status.info("🎯 Tracking module is assigning object IDs...")

        tracker_agent.reset()
        tracks = tracker_agent.update(
            perception_result.get("detections", []),
            fps=1.0
        )
        progress.progress(85)

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        status.info("📈 Trajectory module is generating the single-frame baseline...")

        trajectory_predictions = trajectory_agent.predict(tracks)
        progress.progress(88)

        # ----------------------------------------------------
        # CONFIDENCE-AWARE RISK FUSION
        # ----------------------------------------------------

        status.info("⚖️ Confidence-aware Risk Fusion is combining agent evidence...")

        risk_fusion_result = risk_fusion_agent.fuse(
            perception_result=perception_result,
            scene_result=scene_result,
            planning_result=planning_result,
            safety_result=safety_result,
            trajectory_predictions=trajectory_predictions,
            image_width=1280,
            image_height=720
        )
        progress.progress(94)

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        status.info(
            "💡 Explainability Agent is generating the explanation..."
        )

        explanation_result = explainability_agent.explain(
            perception_result,
            scene_result,
            planning_result,
            safety_result,
            risk_fusion_result=risk_fusion_result,
            trajectory_predictions=trajectory_predictions
        )

        progress.progress(100)


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        elapsed = time.time() - start_time


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "image": current_image.name,

            "perception": perception_result,

            "scene_understanding": scene_result,

            "planning": planning_result,

            "safety": safety_result,

            "tracking": tracks,

            "trajectory": trajectory_predictions,

            "risk_fusion": risk_fusion_result,

            "explanation": explanation_result,

            "processing_time": elapsed
        }


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        result_file = (
            WEB_RESULTS_DIR
            / f"{current_image.stem}_web_result.json"
        )

        with open(
            result_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                default=str
            )


        st.session_state.analysis_result = result

        status.success(
            f"✅ Analysis completed in {elapsed:.2f} seconds"
        )

        time.sleep(0.4)

        progress.empty()


    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            "❌ Analysis failed."
        )

        st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_result:

    result = st.session_state.analysis_result

    perception = result["perception"]

    scene = result["scene_understanding"]

    planning = result["planning"]

    safety = result["safety"]

    explanation = result["explanation"]

    processing_time = result["processing_time"]


    # ========================================================
    # EXTRACT RESULTS
    # ========================================================

    num_objects = perception.get(
        "num_objects",
        0
    )

    object_summary = perception.get(
        "summary",
        {}
    )

    scene_type = scene.get(
        "scene_type",
        "UNKNOWN"
    )

    traffic_density = scene.get(
        "traffic_density",
        "UNKNOWN"
    )

    risk_level = scene.get(
        "risk_level",
        "UNKNOWN"
    )

    vehicle_count = scene.get(
        "vehicle_count",
        0
    )

    vulnerable_users = scene.get(
        "vulnerable_users",
        0
    )

    traffic_controls = scene.get(
        "traffic_controls",
        []
    )

    planner_action = planning.get(
        "action",
        "UNKNOWN"
    )

    safety_decision = safety.get(
        "decision",
        "UNKNOWN"
    )

    safety_level = safety.get(
        "safety_level",
        "UNKNOWN"
    )

    risk_fusion = result.get("risk_fusion", {})

    final_action = risk_fusion.get(
        "final_action",
        safety.get("final_action", planner_action)
    )

    risk_score = risk_fusion.get("risk_score")
    risk_percentage = risk_fusion.get("risk_percentage")
    agent_conflict = risk_fusion.get("agent_conflict", False)
    risk_components = risk_fusion.get("risk_components", {})
    risk_reasons = risk_fusion.get("reasons", [])


    # ========================================================
    # FINAL DECISION
    # ========================================================

    st.divider()

    st.header("🚦 Autonomous Driving Decision")


    decision_left, decision_center, decision_right = st.columns(
        [1, 2, 1]
    )


    with decision_center:

        with st.container(border=True):

            st.markdown(
                "<h2 style='text-align:center;'>"
                "FINAL VERIFIED ACTION"
                "</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<h1 style='text-align:center; font-size:52px;'>"
                f"🚦 {str(final_action).upper()}"
                f"</h1>",
                unsafe_allow_html=True
            )

            st.caption(
                "Planning → Safety → Confidence-Aware Risk Fusion"
            )


    # ========================================================
    # OVERVIEW
    # ========================================================

    st.header("📊 Scene Overview")


    metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)


    with metric1:

        st.metric(
            "🔎 Objects",
            num_objects
        )


    with metric2:

        st.metric(
            "🚘 Vehicles",
            vehicle_count
        )


    with metric3:

        st.metric(
            "🚦 Traffic",
            str(traffic_density).upper()
        )


    with metric4:

        st.metric(
            "⚠️ Risk",
            str(risk_level).upper()
        )


    with metric5:

        st.metric(
            "⚡ Processing",
            f"{processing_time:.2f}s"
        )

    with metric6:

        st.metric(
            "⚖️ Risk Score",
            f"{risk_score:.2f}/1.00" if risk_score is not None else "N/A",
            help="Normalized decision-support score; not a calibrated collision probability."
        )


    # ========================================================
    # DETECTION VISUALIZATION
    # ========================================================

    st.header("🎯 YOLOv8 Object Detection")


    # PerceptionAgent saves the exact annotated image path in
    # perception_result["annotated_image"]. Use that path first.
    # This avoids filename mismatches between the UI and agent.
    visualization_path = None

    annotated_path = perception.get("annotated_image")
    if annotated_path:
        candidate = Path(annotated_path)
        if candidate.exists():
            visualization_path = candidate

    # Fallbacks for older generated visualizations.
    if visualization_path is None:
        fallback_paths = [
            VIS_DIR / f"{current_image.stem}_yolov8_detected.jpg",
            VIS_DIR / f"{current_image.stem}_detected.jpg",
        ]
        for candidate in fallback_paths:
            if candidate.exists():
                visualization_path = candidate
                break

    if visualization_path is not None:
        st.image(
            str(visualization_path),
            caption="YOLOv8 Bounding-Box Detection",
            use_container_width=True
        )
    else:
        st.warning(
            "Bounding-box visualization could not be located. "
            "The detection results below are still valid."
        )


    # ========================================================
    # OBJECT SUMMARY
    # ========================================================

    st.header("🔍 Detected Objects")


    if isinstance(
        object_summary,
        dict
    ) and object_summary:

        object_items = list(
            object_summary.items()
        )

        object_columns = st.columns(
            min(
                4,
                len(object_items)
            )
        )

        for index, (
            object_name,
            count
        ) in enumerate(
            object_items
        ):

            with object_columns[
                index % len(object_columns)
            ]:

                st.metric(
                    str(object_name).title(),
                    count
                )

    else:

        st.info(
            "No object summary available."
        )


    # ========================================================
    # MULTI-AGENT PIPELINE
    # ========================================================

    st.header("🤖 Multi-Agent Decision Pipeline")

    st.caption(
        "Eight AI modules cooperate to transform perception evidence "
        "into an explainable GO / SLOW / STOP decision."
    )

    # Row 1
    row1 = st.columns(4)

    with row1[0]:
        with st.container(border=True):
            st.markdown("### 🔍 Perception")
            st.metric("Objects", num_objects)
            st.caption("YOLOv8n object detection")

    with row1[1]:
        with st.container(border=True):
            st.markdown("### 🧠 Scene")
            st.write(f"**Type:** {scene_type}")
            st.write(f"**Traffic:** {traffic_density}")
            st.caption("Environment understanding")

    with row1[2]:
        with st.container(border=True):
            st.markdown("### 🎯 Tracking")
            st.metric("Tracks", len(tracks))
            st.caption("Single-frame ID baseline")

    with row1[3]:
        with st.container(border=True):
            st.markdown("### 📈 Trajectory")
            prediction_count = len(trajectory_predictions) if isinstance(trajectory_predictions, list) else 0
            st.metric("Predictions", prediction_count)
            st.caption("Single-frame trajectory baseline")

    # Row 2
    row2 = st.columns(4)

    with row2[0]:
        with st.container(border=True):
            st.markdown("### 🗺️ Planning")
            st.success(f"Action: {str(planner_action).upper()}")
            st.caption("Driving action generation")

    with row2[1]:
        with st.container(border=True):
            st.markdown("### 🛡️ Safety")
            st.write(f"**Decision:** {str(safety_decision).upper()}")
            st.write(f"**Level:** {str(safety_level).upper()}")
            st.caption("Safety verification")

    with row2[2]:
        with st.container(border=True):
            st.markdown("### ⚖️ Risk Fusion")
            st.metric(
                "Risk Score",
                f"{risk_score:.2f}/1.00" if risk_score is not None else "N/A"
            )
            st.write(f"**Conflict:** {'YES' if agent_conflict else 'NO'}")
            st.caption("Confidence-aware decision fusion")

    with row2[3]:
        with st.container(border=True):
            st.markdown("### 💡 Explainability")
            st.success("Explanation generated")
            st.caption("Decision reasoning")

    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    st.header(
        "📋 Detailed System Analysis"
    )


    (
        tab_detection,
        tab_scene,
        tab_tracking,
        tab_trajectory,
        tab_planning,
        tab_safety,
        tab_risk,
        tab_explanation
    ) = st.tabs(
        [
            "🔍 Detection",
            "🧠 Scene",
            "🎯 Tracking",
            "📈 Trajectory",
            "🗺️ Planning",
            "🛡️ Safety",
            "⚖️ Risk Fusion",
            "💡 Explanation"
        ]
    )


    # ========================================================
    # DETECTION
    # ========================================================

    with tab_detection:

        st.subheader(
            "YOLOv8 Perception"
        )

        st.write(
            f"**Total objects detected:** {num_objects}"
        )

        if isinstance(
            object_summary,
            dict
        ):

            for name, count in object_summary.items():

                st.write(
                    f"🚘 **{name}:** {count}"
                )

        with st.expander(
            "View raw perception result"
        ):

            st.json(
                perception
            )


    # ========================================================
    # SCENE
    # ========================================================

    with tab_scene:

        st.subheader(
            "Scene Understanding"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Scene Type",
                str(scene_type)
            )

            st.metric(
                "Traffic Density",
                str(traffic_density)
            )

            st.metric(
                "Risk Level",
                str(risk_level)
            )

        with col2:

            st.metric(
                "Vehicles",
                vehicle_count
            )

            st.metric(
                "Vulnerable Users",
                vulnerable_users
            )

            st.write(
                "**Traffic Controls:**"
            )

            st.write(
                traffic_controls
            )

        with st.expander(
            "View raw scene result"
        ):

            st.json(
                scene
            )


    # ========================================================
    # TRACKING
    # ========================================================

    with tab_tracking:

        st.subheader("🎯 Object Tracking")

        st.info(
            "This BDD100K implementation is image-based. "
            "Tracking is therefore evaluated as a single-frame identity baseline "
            "and is not presented as full temporal video tracking."
        )

        st.metric("Objects assigned IDs", len(tracks))

        if tracks:
            tracking_rows = []
            for track in tracks:
                tracking_rows.append({
                    "ID": track.get("track_id", "N/A"),
                    "Class": track.get("class_name", track.get("class", "unknown")),
                    "Frames Seen": track.get("frames_seen", 1)
                })
            st.dataframe(tracking_rows, use_container_width=True, hide_index=True)

        with st.expander("View raw tracking result"):
            st.json(tracks)


    # ========================================================
    # TRAJECTORY
    # ========================================================

    with tab_trajectory:

        st.subheader("📈 Trajectory Prediction")

        st.info(
            "Trajectory prediction is shown as a single-frame baseline. "
            "Because BDD100K is used here as a static image dataset, "
            "future motion is not quantitatively evaluated against video ground truth."
        )

        prediction_count = len(trajectory_predictions) if isinstance(trajectory_predictions, list) else 0
        st.metric("Trajectory predictions", prediction_count)

        if trajectory_predictions:
            # Display trajectory information without velocity/speed fields.
            trajectory_display_rows = []

            for prediction in trajectory_predictions:
                if isinstance(prediction, dict):
                    clean_prediction = {
                        key: value
                        for key, value in prediction.items()
                        if not any(
                            term in key.lower()
                            for term in ("velocity", "speed", "vx", "vy")
                        )
                    }
                    trajectory_display_rows.append(clean_prediction)
                else:
                    trajectory_display_rows.append({"Prediction": prediction})

            st.dataframe(
                trajectory_display_rows,
                use_container_width=True,
                hide_index=True
            )

        with st.expander("View raw trajectory result"):
            st.json(trajectory_predictions)


    # ========================================================
    # PLANNING
    # ========================================================

    with tab_planning:

        st.subheader(
            "Driving Plan"
        )

        st.success(
            f"🚗 Recommended Action: {planner_action}"
        )

        with st.expander(
            "View raw planning result"
        ):

            st.json(
                planning
            )


    # ========================================================
    # SAFETY
    # ========================================================

    with tab_safety:

        st.subheader(
            "Safety Verification"
        )

        safety_col1, safety_col2 = st.columns(2)

        with safety_col1:

            st.metric(
                "Safety Decision",
                str(safety_decision)
            )

        with safety_col2:

            st.metric(
                "Safety Level",
                str(safety_level)
            )

        st.info(
            f"Final fused decision: {str(final_action).upper()}"
        )

        with st.expander(
            "View raw safety result"
        ):

            st.json(
                safety
            )


    # ========================================================
    # RISK FUSION
    # ========================================================

    with tab_risk:

        st.subheader("⚖️ Confidence-Aware Risk Fusion")

        if risk_score is not None:

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Normalized Risk Score", f"{risk_score:.4f}/1.0000")

            with c2:
                st.metric("Decision Risk", f"{risk_score * 100:.1f}/100")

            with c3:
                st.metric("Agent Conflict", "YES" if agent_conflict else "NO")

            st.info(
                "Interpretation: this is a normalized decision-support risk score. "
                "It is not a calibrated probability of collision."
            )

            st.progress(
                max(0.0, min(1.0, float(risk_score))),
                text=f"Normalized risk level: {float(risk_score) * 100:.1f}/100"
            )

            st.subheader("Risk Components")

            if risk_components:
                cols = st.columns(min(3, len(risk_components)))
                for i, (name, value) in enumerate(risk_components.items()):
                    with cols[i % len(cols)]:
                        st.metric(name.replace("_", " ").title(), f"{value:.4f}")
            else:
                st.info("No risk components available.")

            st.subheader("Risk Reasons")
            if risk_reasons:
                for reason in risk_reasons:
                    st.write(f"• {reason}")
            else:
                st.info("No specific risk reasons were returned.")

            st.subheader("Final Fused Action")
            st.success(f"🚦 {str(final_action).upper()}")

            with st.expander("View raw risk-fusion result"):
                st.json(risk_fusion)

        else:
            st.info("Risk Fusion results are not available for this analysis.")


    # ========================================================
    # EXPLANATION
    # ========================================================

    with tab_explanation:

        st.subheader(
            "💡 Why did the system make this decision?"
        )


        if isinstance(
            explanation,
            str
        ):

            explanation_text = explanation

        elif isinstance(
            explanation,
            dict
        ):

            explanation_text = (
                explanation.get(
                    "explanation",
                    explanation.get(
                        "reason",
                        explanation.get(
                            "message",
                            str(explanation)
                        )
                    )
                )
            )

        else:

            explanation_text = str(
                explanation
            )


        st.info(
            explanation_text
        )


        with st.expander(
            "View raw explanation result"
        ):

            if isinstance(
                explanation,
                dict
            ):

                st.json(
                    explanation
                )

            else:

                st.write(
                    explanation
                )


    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    st.header(
        "🔄 Complete Decision Flow"
    )


    flow1, flow2, flow3, flow4, flow5, flow6 = st.columns(6)


    with flow1:

        st.success(
            "🔍 Perception\n\n"
            "Objects detected"
        )


    with flow2:

        st.success(
            "🧠 Scene\n\n"
            "Environment understood"
        )


    with flow3:

        st.success(
            "🗺️ Planning\n\n"
            f"Action: {planner_action}"
        )


    with flow4:

        st.success(
            "🛡️ Safety\n\n"
            f"{safety_decision}"
        )


    with flow5:

        st.success(
            "⚖️ Risk Fusion\n\n"
            + (f"Score: {risk_score:.2f}/1.00" if risk_score is not None else "N/A")
        )

    with flow6:

        st.success(
            "🚦 Decision\n\n"
            f"{final_action}"
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.header(
        "💾 Save Analysis"
    )


    json_data = json.dumps(
        result,
        indent=4,
        default=str
    )


    st.download_button(
        "⬇️ Download Complete Analysis",
        data=json_data,
        file_name=(
            f"{current_image.stem}_analysis.json"
        ),
        mime="application/json",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'Multi-Agent Explainable Autonomous Driving System'
    '<br><br>'
    'BDD100K • YOLOv8 • Scene Understanding • '
    'Planning • Safety Verification • Explainability'
    '</div>',
    unsafe_allow_html=True
)