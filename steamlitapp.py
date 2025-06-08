import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import io
import torch
import collections
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(
    page_title="AuraVision | AI Defect Inspector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Custom CSS for Futuristic Look ---
def load_css():
    st.markdown("""
    <style>
    /* Main app theming */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Expander styling */
    .st-emotion-cache-p5msec {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    /* Metric styling */
    .st-emotion-cache-1b0udgb {
        background-color: rgba(46, 170, 220, 0.1);
        border: 1px solid rgba(46, 170, 220, 0.5);
        border-radius: 8px;
        padding: 1rem;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #2ea0df;
        background-color: transparent;
        color: #2ea0df;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2ea0df;
        color: white;
        border: 1px solid #2ea0df;
    }
    /* Title styling */
    h1, h2, h3 {
        color: #58a6ff;
    }
    </style>
    """, unsafe_allow_html=True)


load_css()

# --- Action Suggestions Data ---
# (Kept the same as your original request)
ACTION_SUGGESTIONS = {
    "polished_casting": {
        "description": "Excessive polishing can lead to reduced friction and inefficient braking.",
        "action": """🔧 **Check disc thickness and hardness.**<br>🔧 **Assess brake performance.**<br>🔁 **Replace if surface friction is below tolerance.**"""
    },
    "burr": {
        "description": "Small protrusions of metal, usually from improper machining.",
        "action": """🧽 **Deburr manually or with a tool.**<br>🔧 **Inspect nearby components for wear or damage.**<br>✅ **Ensure no risk of cuts or misalignment.**"""
    },
    "casting_burr": {
        "description": "Burrs formed during the initial casting process.",
        "action": """🛠 **Carefully grind or file off casting burrs.**<br>🔍 **Check for deeper casting flaws or porosity.**<br>🔄 **If burr is too large or irregular → replace component.**"""
    },
    "crack": {
        "description": "Fractures in the material, a serious safety concern.",
        "action": """⛔️ **Immediate replacement.**<br>🔍 **Inspect the surrounding system (caliper, hub, etc.).**<br>🧪 **Conduct crack propagation analysis if recurring.**"""
    },
    "pit": {
        "description": "Small holes or depressions due to corrosion or surface damage.",
        "action": """🧼 **Clean the disc thoroughly.**<br>📏 **Measure depth of pits.**<br>🔄 **Replace if pitting is extensive or compromises braking.**"""
    },
    "scratch": {
        "description": "Surface marks or grooves.",
        "action": """🪵 **Light resurfacing (turning or grinding) if shallow.**<br>🔎 **Investigate cause (e.g., debris in brake pad).**<br>🔄 **Replace disc if scratches are deep or cause vibration.**"""
    },
    "strain": {
        "description": "Deformation from stress, heat, or mechanical load.",
        "action": """🌡 **Check for warping (runout test).**<br>🔄 **Replace if disc is out of tolerance.**<br>🧪 **Review thermal exposure history.**"""
    },
    "unpolished_casting": {
        "description": "Raw, unfinished surface that may cause uneven wear.",
        "action": """🛠 **Machine or polish surface to spec.**<br>🔧 **Check for fitment issues.**<br>❗️ **If polishing doesn’t bring to spec → replace.**"""
    }
}

# --- Session State Initialization ---
if 'all_detections' not in st.session_state:
    st.session_state.all_detections = []
if 'processed_images_data' not in st.session_state:
    st.session_state.processed_images_data = []


# --- Helper Functions ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads a YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def generate_pdf_report():
    """Generates a PDF report of all detections in the current session."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AuraVision Defect Detection Report", 1, 1, 'C')
    pdf.ln(10)

    for data in st.session_state.processed_images_data:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, f"Analysis for: {data['filename']}", 0, 1)

        # Save processed image to a temporary buffer
        img_buffer = io.BytesIO()
        data['processed_image'].save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Get image dimensions to maintain aspect ratio
        with Image.open(img_buffer) as img:
            width, height = img.size
        aspect_ratio = height / width
        pdf_img_width = pdf.w - 20  # Full width minus margins
        pdf_img_height = pdf_img_width * aspect_ratio

        # Reset buffer for FPDF
        img_buffer.seek(0)

        pdf.image(img_buffer, x=10, w=pdf_img_width, h=pdf_img_height)
        pdf.ln(5)

        if not data['detections']:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 10, "No defects detected in this image.", 0, 1)
        else:
            pdf.set_font("Helvetica", "B", 10)
            # Create table header
            pdf.cell(60, 10, "Defect Class", 1)
            pdf.cell(40, 10, "Confidence", 1)
            pdf.ln()
            # Create table rows
            pdf.set_font("Helvetica", "", 10)
            for det in data['detections']:
                pdf.cell(60, 10, det['class'], 1)
                pdf.cell(40, 10, f"{det['confidence']:.2%}", 1)
                pdf.ln()

        # Check if we need to add a new page to avoid overflow
        if pdf.get_y() > 250:
            pdf.add_page()

    # The final, corrected return statement:
    return bytes(pdf.output())


# --- Sidebar ---
with st.sidebar:
    st.title("🛰️ AuraVision Inspector")

    # Model Selection
    model_path = "models/best.pt"  # Default model
    model = load_yolo_model(model_path)

    # Input Settings
    st.header("⚙️ Analysis Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

    # Device Selection
    device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    device = st.radio("Compute Device", device_options, horizontal=True)

    st.header("📥 Input Source")
    input_source = st.radio("Select source:", ["Image Upload", "Live Camera"], horizontal=True)

    st.divider()
    if st.button("Clear Session Data", use_container_width=True):
        st.session_state.all_detections.clear()
        st.session_state.processed_images_data.clear()
        st.rerun()

# --- Main Page ---
st.title("AI Defect Detection Dashboard")
st.markdown("Upload images or use a live camera feed to identify manufacturing defects with AI.")

# --- Input Handling ---
if model:
    if input_source == "Image Upload":
        uploaded_files = st.file_uploader(
            "Upload image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                results = model(image, conf=confidence_threshold, iou=iou_threshold, device=device)

                detections = []
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        detections.append({"class": class_name, "confidence": confidence})
                        st.session_state.all_detections.append({"class": class_name, "confidence": confidence})

                res_plotted = results[0].plot()
                processed_image_rgb = Image.fromarray(res_plotted[..., ::-1])
                st.session_state.processed_images_data.append({
                    "filename": uploaded_file.name,
                    "original_image": image,
                    "processed_image": processed_image_rgb,
                    "detections": detections
                })

    elif input_source == "Live Camera":
        st.info("Position the object in front of the camera and click 'Analyze Frame'.")
        camera_image_file = st.camera_input("Live Camera Feed", label_visibility="collapsed")
        if camera_image_file:
            image = Image.open(camera_image_file)
            results = model(image, conf=confidence_threshold, iou=iou_threshold, device=device)

            detections = []
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    confidence = float(box.conf)
                    detections.append({"class": class_name, "confidence": confidence})
                    st.session_state.all_detections.append({"class": class_name, "confidence": confidence})

            res_plotted = results[0].plot()
            processed_image_rgb = Image.fromarray(res_plotted[..., ::-1])

            # For camera, we overwrite the last analysis to avoid flooding the session
            st.session_state.processed_images_data = [{
                "filename": "Live Camera Frame",
                "original_image": image,
                "processed_image": processed_image_rgb,
                "detections": detections
            }]

# --- Dashboard Display ---
if not st.session_state.processed_images_data:
    st.image("https://i.imgur.com/gY91G6W.png", caption="Awaiting analysis...", use_container_width=True)
else:
    # --- Summary Metrics & Charts ---
    st.header("📊 Session Dashboard")
    total_images = len(st.session_state.processed_images_data)
    total_defects = len(st.session_state.all_detections)

    col1, col2, col3 = st.columns(3)
    col1.metric("Images Analyzed", total_images)
    col2.metric("Total Defects Found", total_defects)
    col3.metric("Avg. Defects / Image", f"{total_defects / total_images:.2f}" if total_images > 0 else "0.00")

    st.divider()

    col_chart, col_download = st.columns([2, 1])
    with col_chart:
        st.subheader("Defect Frequency")
        if total_defects > 0:
            defect_counts = collections.Counter(d['class'] for d in st.session_state.all_detections)
            df_counts = pd.DataFrame(defect_counts.items(), columns=['Defect', 'Count']).set_index('Defect')
            st.bar_chart(df_counts)
        else:
            st.info("No defects detected yet in this session.")

    with col_download:
        st.subheader("Export Results")
        if total_defects > 0:
            # CSV Download
            df_export = pd.DataFrame(st.session_state.all_detections)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_data, "defect_report.csv", "text/csv", use_container_width=True)

            # PDF Download
            pdf_data = generate_pdf_report()
            st.download_button("Download PDF Report", pdf_data, "defect_report.pdf", "application/pdf",
                               use_container_width=True)

    st.divider()

    # --- Detailed Breakdown per Image ---
    st.header("🖼️ Detailed Analysis")
    for data in reversed(st.session_state.processed_images_data):  # Show most recent first
        with st.expander(f"**{data['filename']}** | Found {len(data['detections'])} defects"):
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(data['original_image'], caption="Original Image", use_container_width=True)
            with col_img2:
                st.image(data['processed_image'], caption="Processed Image", use_container_width=True)

            if not data['detections']:
                st.success("✅ No defects found in this image.")
            else:
                st.subheader("Detected Defects & Actions")
                for i, det in enumerate(data['detections']):
                    st.markdown(f"--- \n#### Defect #{i + 1}: `{det['class']}`")
                    st.progress(det['confidence'], text=f"Confidence: {det['confidence']:.2f}")

                    suggestion = ACTION_SUGGESTIONS.get(det['class'])
                    if suggestion:
                        with st.container(border=True):
                            st.markdown(f"**Description:** {suggestion['description']}")
                            st.markdown(f"**Suggested Action:**", unsafe_allow_html=True)
                            st.markdown(suggestion['action'], unsafe_allow_html=True)