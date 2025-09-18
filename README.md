# Object Detector

A modern, real-time object detection application built with Streamlit, OpenCV, and YOLO. This application provides an intuitive web interface for detecting objects in images and live webcam feeds with customizable settings and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### Multiple Input Methods
- **Live Webcam Detection**: Real-time object detection from your webcam
- **Image Upload**: Detect objects in uploaded images (JPG, JPEG, PNG)

### Advanced Detection Capabilities
- **YOLO Models**: Support for both YOLOv3 and YOLOv3-tiny models
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity
- **Object Filtering**: Focus on specific object types
- **Non-Max Suppression**: Eliminate duplicate detections

### Real-Time Analytics
- **FPS Counter**: Monitor processing performance
- **Detection History**: Track up to 100 recent detections
- **Statistics Dashboard**: View total detections and unique objects
- **Most Detected Objects**: Analyze detection patterns with percentages
- **Confidence Metrics**: Track average confidence scores

### Modern UI Features
- **Dark Theme**: Sleek, modern interface with gradient effects
- **Responsive Layout**: Optimized for different screen sizes
- **Live Updates**: Real-time statistics and visual feedback
- **Custom CSS Styling**: Professional appearance with smooth animations

### Performance Optimization
- **Frame Skipping**: Adjustable frame processing rate (1-10 frames)
- **Model Selection**: Choose between accuracy (YOLOv3) and speed (YOLOv3-tiny)
- **Efficient Caching**: Automatic model caching for faster subsequent runs

### Export & Storage
- **Auto-Save Detections**: Optional automatic saving of detected images
- **Timestamped Files**: Organized output with timestamp naming
- **Detection History Export**: Track and review past detections

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (for live detection)
- Internet connection (for initial model download)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/object-detector.git
cd object-detector
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run ObjectDetector.py
```

The application will open in your default web browser at `http://localhost:8501`

## Requirements

Create a `requirements.txt` file with:
```txt
streamlit>=1.25.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

## Usage

### Getting Started
1. **Launch the application** using the command above
2. **Select input method**: Choose between Webcam or Image Upload
3. **Configure settings** in the sidebar:
   - Model version (YOLOv3 or YOLOv3-tiny)
   - Confidence threshold (0.1 to 1.0)
   - Object filters (optional)
   - Frame skip rate (for performance)

### Live Webcam Detection
1. Select "Webcam (Live)" option
2. Click **"Start Detection"** button
3. Grant camera permissions if prompted
4. View real-time detections with bounding boxes
5. Monitor FPS and detection statistics
6. Click **"Stop Detection"** when finished

### Image Upload Detection
1. Select "Upload Image" option
2. Click **"Browse files"** and select an image
3. Click **"Detect Objects"** button
4. View detection results with bounding boxes and labels

### Sidebar Features
- **Detection Settings**: Adjust model and confidence parameters
- **Object Filter**: Select specific objects to detect
- **Performance**: Control frame processing rate
- **Statistics**: View real-time detection metrics
- **Detection History**: Review recent detections with timestamps
- **Export Options**: Enable automatic image saving

## Project Structure

```
object-detector/
│
├── ObjectDetector.py       # Main application file
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── detections/           # Saved detection images (auto-created)
    └── detection_*.jpg   # Timestamped detection results
```

## Configuration

### Model Selection
- **YOLOv3**: Full model, higher accuracy, slower processing (~250MB)
- **YOLOv3-tiny**: Lightweight model, faster processing, lower accuracy (~35MB)

### Performance Tuning
- **Confidence Threshold**: Higher values (0.7-0.9) for fewer, more accurate detections
- **Frame Skip**: Higher values (5-10) for better performance on slower systems
- **Model Choice**: Use YOLOv3-tiny for real-time performance on standard hardware

## Detected Object Classes

The application can detect 80 different object classes from the COCO dataset, including:
- People and animals (person, dog, cat, bird, etc.)
- Vehicles (car, truck, bicycle, motorcycle, etc.)
- Everyday objects (chair, bottle, cell phone, laptop, etc.)
- Food items (apple, banana, pizza, etc.)

## Troubleshooting

### Model Download Issues
- Ensure stable internet connection for first-time model download
- Models are cached in `~/.yolo_models/` directory
- Clear cache and restart if download fails

### Webcam Not Working
- Check camera permissions in your browser
- Ensure no other application is using the camera
- Try refreshing the page or restarting the application

### Low FPS
- Increase frame skip value in settings
- Switch to YOLOv3-tiny model
- Close other resource-intensive applications
- Reduce confidence threshold

### Import Errors
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Update pip: `pip install --upgrade pip`
- Try reinstalling OpenCV: `pip uninstall opencv-python && pip install opencv-python`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLO (You Only Look Once)](https://pjreddie.com/darknet/yolo/) for the detection models
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [COCO Dataset](https://cocodataset.org/) for object classes

---

**Made with Python, Streamlit, and OpenCV**
