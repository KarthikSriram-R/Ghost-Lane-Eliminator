# Ghost-Lane-Eliminator
🚦 Ghost Lane Eliminator v2.0 AI-Driven Dynamic Traffic Management &amp; Emergency Vehicle Preemption  Developed for AI4Dev'26 @ PSG Tech by Sai Karthik.P , Denzil Aaron.S , Karthik Sriram R


## 📖 Project Overview

The **Ghost Lane Eliminator** is an intelligent traffic control system designed to solve the inefficiency of "Ghost Lanes"—lanes where traffic signals remain green despite no vehicle presence, or stay red despite heavy congestion.

By utilizing **YOLOv8** (Real-time Object Detection) and custom tracking algorithms, the system calculates the "Traffic Load" of each lane in real-time and dynamically adjusts green light durations. It further prioritizes emergency vehicles (ambulances) using a high-priority preemption trigger.

---

## ✨ Key Features

* **Dynamic Green Scaling:** Instead of fixed timers, the system calculates green time based on vehicle weightage (e.g., Bus = 5.0, Car = 2.0, Bike = 1.0).
* **Emergency Preemption:** Automatically detects ambulances and immediately switches the signal to green for that specific lane.
* **Dual-Mode Detection:** * **YOLO Mode:** High-accuracy detection using `yolov8n.pt`.
* **Contour Fallback:** A LAB-color space-based fallback for environments without GPU/high-end CPU support.


* **Real-time Analytics:** * **Congestion Index:** A visual bar indicating the overall pressure on the junction.
* **Heatmapping:** Visualizes traffic density over time to identify bottleneck areas.


* **Flexible Input:** Supports live **Webcam** feeds or **Screen Capture** for processing pre-recorded traffic simulation videos.

---

## 🛠️ Technical Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **Deep Learning:** Ultralytics YOLOv8
* **Optimization:** MSS/Pillow for high-speed screen grabbing
* **Tracking:** Custom IOU-based (Intersection over Union) centroid tracker with speed estimation.

---

## 🚀 Controls & Usage

The system features an interactive dashboard with the following keyboard hotkeys:
| Key | Action |
| :--- | :--- |
| `Space` | Pause/Resume Processing |
| `H` | Toggle Traffic Heatmap Overlay |
| `O` | Switch Lane Orientation (Horizontal/Vertical) |
| `1-4` | Manually Force Active Lane |
| `S` | Take a Screenshot of the Analytics Panel |
| `R` | Reset Session Statistics |
| `Q` | Quit Application |

---

## 📊 Logic Flow

1. **Capture:** Frame is grabbed via Webcam or Screen Region.
2. **Detect:** Vehicles are identified and assigned a "Weight" based on class.
3. **Track:** Entities are tracked across frames to calculate speed and persistence.
4. **Allocate:** The `LaneManager` calculates the optimal $Time_{green}$ using the formula:

$$T_{green} = \min(Weighted\_Load \times 2 + T_{min}, T_{max})$$


5. **Visualize:** Data is pushed to a real-time side panel for monitoring.

---


