# AI-Powered-Library-Seat-Finder-System
Built a real-time occupancy detection system using YOLOv8 computer vision model and Streamlit to identify available library seats, enhancing space utilization and student satisfaction by providing live occupancy dashboards and achieving high detection accuracy

## Data Acquisition / Preparation
Input Data: Online photos and room images from Hayden and Noble Library on ASU Campus.
Detection model Used: Pre-trained YOLO v8 for people detection.
System built using Streamlit, OpenCV, and Ultralytics.
## CV Model
🧩 Where CV Fits 
- Model building, evaluation, deployment
- Detects seat occupancy from images

🚀 Why CV
- Fast, automated – avoids manual errors
- Alternatives (pressure sensors, manual validation) are costly and impractical

🛠️ Model Used
-Pretrained YOLOv8 via Ultralytics
- Fine-tuned on Hayden + Noble library photos

📊 Validation Plan
- Use random snapshots post-deployment
- Evaluate with Precision, Recall, mAP

To monitor and update the model post-deployment, we will continuously update and retrain the model with new images, to make sure the accuracy of the model stays high
<img width="1256" height="482" alt="image" src="https://github.com/user-attachments/assets/f289aea7-1c16-45b4-8903-c8987e78be0a" />
