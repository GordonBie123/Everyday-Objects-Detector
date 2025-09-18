# ObjectDetector.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
import urllib.request
import hashlib
from collections import Counter

# Page config
st.set_page_config(
    page_title="Object Detector",   
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MAX_HISTORY = 100
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for better performance

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Card styling */
    .stats-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem;
    }
    
    /* Detection box styling */
    .detection-info {
        background: rgba(255,255,255,0.05);
        border-left: 2px solid rgba(255,255,255,0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 400;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #2d2d2d;
        border-color: rgba(255,255,255,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 6px;
    }
    
    /* FPS counter styling */
    .fps-counter {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.7);
        color: #00ff00;
        padding: 5px 10px;
        border-radius: 5px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo(model_version="yolov3"):
    """Load YOLO model and configuration"""
    # Create model directory
    model_dir = os.path.join(os.path.expanduser("~"), ".yolo_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Model configurations
    model_configs = {
        "yolov3": {
            "weights": "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights",
            "cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "size": 248007048  # Expected file size for weights
        },
        "yolov3-tiny": {
            "weights": "https://github.com/pjreddie/darknet/raw/master/yolov3-tiny.weights",
            "cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
            "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "size": 35434956  # Expected file size for tiny weights
        }
    }
    
    config = model_configs[model_version]
    
    # Download files with progress bar
    files_to_download = {
        f'{model_version}.weights': config['weights'],
        f'{model_version}.cfg': config['cfg'],
        'coco.names': config['names']
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            with st.spinner(f'Downloading {filename}...'):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Add headers to avoid 403 errors
                        opener = urllib.request.build_opener()
                        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                        urllib.request.install_opener(opener)
                        
                        urllib.request.urlretrieve(url, filepath)
                        
                        # Verify weights file size
                        if filename.endswith('.weights'):
                            file_size = os.path.getsize(filepath)
                            expected_size = config.get('size', 0)
                            if expected_size > 0 and abs(file_size - expected_size) > 1000000:  # 1MB tolerance
                                os.remove(filepath)
                                raise ValueError(f"Downloaded file size mismatch. Expected ~{expected_size} bytes, got {file_size}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            st.warning(f"Download attempt {attempt + 1} failed, retrying...")
                            time.sleep(2)  # Wait before retry
                        else:
                            st.error(f"Error downloading {filename}: {str(e)}")
                            st.info("Try downloading manually from: https://github.com/patrick013/Object-Detection---Yolov3")
                            raise
    
    # Load YOLO
    weights_path = os.path.join(model_dir, f'{model_version}.weights')
    cfg_path = os.path.join(model_dir, f'{model_version}.cfg')
    names_path = os.path.join(model_dir, 'coco.names')
    
    net = cv2.dnn.readNet(weights_path, cfg_path)
    
    # Load class names
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def detect_objects(image, net, output_layers, classes, confidence_threshold=0.5):
    """Detect objects in an image using YOLO""" 
    height, width, channels = image.shape
    
    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Extract information from outputs
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    detected_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            detected_objects.append({
                'class': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
    
    return detected_objects

def draw_detections(image, detections, classes):
    """Draw bounding boxes and labels on the image"""
    # Generate random colors for each class
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for detection in detections:
        x, y, w, h = detection['box']
        label = f"{detection['class']}: {detection['confidence']:.2f}"
        
        # Get color for this class
        try:
            color_idx = classes.index(detection['class'])
        except ValueError:
            color_idx = 0
        color = colors[color_idx]
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Calculate label size
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_width = label_size[0]
        label_height = label_size[1]
        
        # Draw label background
        cv2.rectangle(image, (x, y - label_height - 10), (x + label_width, y), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def draw_fps(image, fps):
    """Draw FPS counter on image"""
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(image, fps_text, (image.shape[1] - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def save_detection_image(image, detections, output_dir="detections"):
    """Save detected image with timestamp"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'unique_objects' not in st.session_state:
    st.session_state.unique_objects = set()
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'fps' not in st.session_state:
    st.session_state.fps = 0.0

# Header
st.markdown("""
<div class="header-container">
    <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 400;">Object Detector</h1>
    <p style="color: rgba(255,255,255,0.7); font-size: 1rem; margin-top: 0.5rem; font-weight: 300;">
        Real-time object detection using YOLO and OpenCV
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Detection Settings")
    
    # Model selection
    model_version = st.selectbox(
        "Model Version",
        ["yolov3", "yolov3-tiny"],
        help="YOLOv3-tiny is faster but less accurate"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )
    
    # Object filter
    st.markdown("### Object Filter")
    filter_objects = st.multiselect(
        "Show only these objects:",
        ["person", "car", "bicycle", "dog", "cat", "chair", "bottle", "cell phone"],
        help="Leave empty to show all detected objects"
    )
    
    st.markdown("### Performance")
    frame_skip = st.slider(
        "Process every N frames",
        min_value=1,
        max_value=10,
        value=PROCESS_EVERY_N_FRAMES,
        help="Higher values = better performance, lower accuracy"
    )
    
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Detections", st.session_state.total_detections)
    with col2:
        st.metric("Unique Objects", len(st.session_state.unique_objects))
    
    if st.session_state.fps > 0:
        st.metric("FPS", f"{st.session_state.fps:.1f}")
    
    st.markdown("### Detection History")
    if st.session_state.detection_history:
        for i, hist in enumerate(st.session_state.detection_history[-5:][::-1]):
            st.markdown(f"""
            <div class="detection-info">
                <strong>Detection {len(st.session_state.detection_history) - i}</strong><br>
                {', '.join(hist['objects'])}<br>
                <small>{hist['timestamp']}</small>
                <small style="color: #667eea;">Confidence: {hist['avg_confidence']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No detections yet")
    
    if st.button("Clear History"):
        st.session_state.detection_history = []
        st.session_state.total_detections = 0
        st.session_state.unique_objects = set()
        st.rerun()
    
    # Export options
    st.markdown("### Export Options")
    save_detections = st.checkbox("Save detected images", help="Save images with detections to disk")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Camera Feed")
    
    # Camera input method selection
    input_method = st.radio(
        "Select input method:",
        ["Webcam (Live)", "Upload Image"],
        horizontal=True
    )
    
    if input_method == "Webcam (Live)":
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("Start Detection", disabled=st.session_state.run_detection):
                st.session_state.run_detection = True
                st.rerun()
        
        with col_stop:
            if st.button("Stop Detection", disabled=not st.session_state.run_detection):
                st.session_state.run_detection = False
                st.rerun()
        
        if st.session_state.run_detection:
            # Load YOLO model
            try:
                net, classes, output_layers = load_yolo(model_version)
                
                # Placeholder for video feed
                video_placeholder = st.empty()
                
                # Open webcam
                cap = cv2.VideoCapture(0)
                
                # FPS calculation variables
                fps_start_time = time.time()
                fps_frame_count = 0
                frame_count = 0
                
                try:
                    while st.session_state.run_detection and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break
                        
                        frame_count += 1
                        fps_frame_count += 1
                        
                        # Calculate FPS
                        fps_elapsed_time = time.time() - fps_start_time
                        if fps_elapsed_time > 1:
                            st.session_state.fps = fps_frame_count / fps_elapsed_time
                            fps_frame_count = 0
                            fps_start_time = time.time()
                        
                        # Process frame based on frame skip setting
                        if frame_count % frame_skip == 0:
                            # Detect objects
                            detections = detect_objects(frame, net, output_layers, classes, confidence_threshold)
                            
                            # Filter objects if specified
                            if filter_objects:
                                detections = [d for d in detections if d['class'] in filter_objects]
                            
                            # Draw detections
                            frame_with_detections = draw_detections(frame.copy(), detections, classes)
                            
                            # Update statistics
                            if detections:
                                detected_classes = [d['class'] for d in detections]
                                confidences = [d['confidence'] for d in detections]
                                avg_confidence = sum(confidences) / len(confidences)
                                
                                st.session_state.total_detections += len(detections)
                                st.session_state.unique_objects.update(detected_classes)
                                
                                # Add to history with limit
                                st.session_state.detection_history.append({
                                    'objects': detected_classes,
                                    'timestamp': time.strftime("%H:%M:%S"),
                                    'count': len(detections),
                                    'avg_confidence': avg_confidence
                                })
                                
                                # Limit history size
                                if len(st.session_state.detection_history) > MAX_HISTORY:
                                    st.session_state.detection_history.pop(0)
                                
                                # Save image if enabled
                                if save_detections:
                                    save_detection_image(frame_with_detections, detections)
                        else:
                            frame_with_detections = frame
                        
                        # Draw FPS
                        frame_with_detections = draw_fps(frame_with_detections, st.session_state.fps)
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)
                
                finally:
                    cap.release()
                    st.session_state.run_detection = False
                    st.session_state.fps = 0.0
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please ensure you have an active internet connection for first-time model download.")
                st.session_state.run_detection = False
    
    else:  # Upload Image
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect objects"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Detect objects button
            if st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    try:
                        # Load YOLO model
                        net, classes, output_layers = load_yolo(model_version)
                        
                        # Detect objects
                        detections = detect_objects(image_bgr, net, output_layers, classes, confidence_threshold)
                        
                        # Filter objects if specified
                        if filter_objects:
                            detections = [d for d in detections if d['class'] in filter_objects]
                        
                        # Draw detections
                        result_image = draw_detections(image_bgr.copy(), detections, classes)
                        
                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        # Display result
                        st.image(result_rgb, caption="Detection Results", use_column_width=True)
                        
                        # Update statistics
                        if detections:
                            detected_classes = [d['class'] for d in detections]
                            confidences = [d['confidence'] for d in detections]
                            avg_confidence = sum(confidences) / len(confidences)
                            
                            st.session_state.total_detections += len(detections)
                            st.session_state.unique_objects.update(detected_classes)
                            
                            # Add to history
                            st.session_state.detection_history.append({
                                'objects': detected_classes,
                                'timestamp': time.strftime("%H:%M:%S"),
                                'count': len(detections),
                                'avg_confidence': avg_confidence
                            })
                            
                            # Limit history size
                            if len(st.session_state.detection_history) > MAX_HISTORY:
                                st.session_state.detection_history.pop(0)
                            
                            # Save image if enabled
                            if save_detections:
                                filepath = save_detection_image(result_image, detections)
                                st.success(f"Image saved to: {filepath}")
                        else:
                            st.info("No objects detected with the current confidence threshold.")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
            else:
                st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### Current Detections")
    
    if st.session_state.detection_history:
        latest = st.session_state.detection_history[-1]
        
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin: 0; color: #667eea; font-weight: 400;">Objects Found: {latest['count']}</h4>
            <div style="margin-top: 1rem;">
        """, unsafe_allow_html=True)
        
        # Count occurrences
        object_counts = Counter(latest['objects'])
        
        for obj, count in object_counts.items():
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span>{obj.title()}</span>
                <span style="color: #667eea; font-weight: 500;">{count}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        st.markdown(f"""
        <div class="stats-card">
            <p style="margin: 0; line-height: 1.8;">
                <strong>Detection Time:</strong> {latest['timestamp']}<br>
                <strong>Avg Confidence:</strong> {latest['avg_confidence']:.2%}<br>
                <strong>Confidence Threshold:</strong> {confidence_threshold}<br>
                <strong>Model:</strong> {model_version.upper()}<br>
                <strong>Frame Skip:</strong> {frame_skip}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top detected objects
        if len(st.session_state.detection_history) > 5:
            st.markdown("### Most Detected Objects")
            all_objects = []
            for hist in st.session_state.detection_history:
                all_objects.extend(hist['objects'])
            
            top_objects = Counter(all_objects).most_common(5)
            
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            for obj, count in top_objects:
                percentage = (count / len(all_objects)) * 100
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{obj.title()}</span>
                        <span>{percentage:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; margin-top: 0.25rem;">
                        <div style="background: #667eea; height: 100%; width: {percentage}%; border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No active detections. Start the camera or upload an image to begin.")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.1);">
    <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">
        Built with Streamlit, OpenCV, and YOLO<br>
        <small>Performance optimized with frame skipping and efficient state management</small>
    </p>
</div>
""", unsafe_allow_html=True)