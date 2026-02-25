🔥 Fire & Smoke Detection from Surveillance Cameras

📌 **Overview**
  
  This project presents a Fire & Smoke Detection System developed using computer vision techniques.
  
  The system analyzes surveillance images and classifies them as:
      - 🔥 Fire
      - 🌫 Smoke
      - ✅ Normal Scene
 
  The solution combines **Digital Image Processing (DIP) techniques with Deep Learning (YOLOv8)** to improve detection reliability and accuracy.
  
  This system can be applied in smart surveillance environments such as industrial monitoring, warehouses, smart buildings, and public safety systems.
  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
🎯 **Objectives**

  - Detect fire using HSV color segmentation and contour analysis
  - Detect smoke using texture and edge-based analysis
  - Reduce false positives using multi-condition validation
  - Compare traditional image processing with deep learning approaches
  - Provide a foundation for real-time fire monitoring systems

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 **Detection Approaches**

  1️⃣ Digital Image Processing (DIP-Based Detection)
  
  Fire Detection Techniques
    - HSV color thresholding
    - Contour area filtering
    - Brightness variance analysis (fire flicker property)
    - Morphological noise reduction
    
  Smoke Detection Techniques
    - Low saturation detection
    - Brightness filtering
    - Edge density analysis (soft texture characteristic)
    - Morphological region refinement

  2️⃣ Deep Learning Detection (YOLOv8)
    - Pre-trained YOLOv8 nano model
    - Automatic object localization with bounding boxes
    - Confidence-based detection
    - Saved annotated output images

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
🛠 **Technologies Used**
    - Python 3.x
    - OpenCV
    - NumPy
    - Ultralytics YOLOv8
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - HSV Color Space Segmentation

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
📂 **Project Structure**
    fire-smoke-detection-system/
│
├── data/                        # Input test images
├── outputs/                     # Saved detection results
├── models/                      # YOLO model files (optional)
│
├── dip_fire_pipeline.py         # Basic fire detection
├── dip_smoke_pipeline.py        # Basic smoke detection
├── dip_pipeline.py              # Advanced fire & smoke detection
├── yolo_pipeline.py             # YOLOv8-based detection
│
├── requirements.txt
└── README.md

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ **Installation**

  1️⃣ Clone the Repository
    git clone https://github.com/yourusername/fire-smoke-detection-system.git
    cd fire-smoke-detection-system
    
  2️⃣ Create Virtual Environment (Recommended)
    python -m venv venv
    venv\Scripts\activate     # Windows
    
  3️⃣ Install Dependencies
    pip install -r requirements.txt

If YOLO is not installed:
    pip install ultralytics

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 **How to Run**

  🔥 Advanced Fire & Smoke Detection (Recommended)
    python dip_pipeline.p
    
  🔥 Basic Fire Detection
    python dip_fire_pipeline.py
    
  🌫 Basic Smoke Detection
    python dip_smoke_pipeline.py
    
  🤖 YOLOv8 Detection
    python yolo_pipeline.py

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
📊** Output**

The system:

  - Displays detection results in a window
  - Draws bounding boxes around detected regions
  - Labels detected area as Fire / Smoke / Normal
  - Saves annotated images in the outputs/ directory

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
📈 **Future Improvements**

  - Real-time CCTV video stream integration
  - Alarm notification system
  - Web dashboard for monitoring
  - Accuracy evaluation metrics
  - Model training on custom dataset

------------------------------------------------------------------------------------------------------------------------------------------------------------------
📜 **License**

This project is developed for academic purposes.

## Contributors
-Hasara Wijayarthna
