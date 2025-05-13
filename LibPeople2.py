import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Initialize session state to store room data
if "room_data" not in st.session_state:
    st.session_state.room_data = []

st.set_page_config(page_title="Library Seat Finder", layout="wide")
st.title("📚 AI-Powered Occupancy Finder")

# View selection
view = st.sidebar.selectbox("Select View", ["Admin", "Student"])

# Detect people in image
def detect_people(image_path):
    results = model(image_path)
    person_detections = [box for box in results[0].boxes.cls if int(box) == 0]
    count = len(person_detections)
    return count, results[0]

# Detect people in video (average over frames)
def detect_people_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(frame_path, frame)
        count, _ = detect_people(frame_path)
        total_count += count
        frame_count += 1
        if frame_count >= 10:
            break

    cap.release()
    avg_count = total_count // frame_count if frame_count > 0 else 0
    return avg_count

# Admin View
if view == "Admin":
    st.sidebar.header("➕ Add New Room")
    facility_name = st.sidebar.text_input("Facility Name")
    room_name = st.sidebar.text_input("Room Name")
    capacity = st.sidebar.number_input("Room Capacity", min_value=1, value=30)
    image_file = st.sidebar.file_uploader("Upload Room Image", type=["jpg", "jpeg", "png"])
    video_file = st.sidebar.file_uploader("Or Upload Room Video", type=["mp4", "avi", "mov"])

    if st.sidebar.button("Add Room"):
        if (image_file or video_file) and room_name and facility_name:
            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                    temp.write(image_file.read())
                    temp_path = temp.name
                count, _ = detect_people(temp_path)
            elif video_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                    temp.write(video_file.read())
                    temp_path = temp.name
                count = detect_people_in_video(temp_path)

            occupancy = min(count, capacity)
            available = max(0, capacity - occupancy)

            st.session_state.room_data.append({
                "Facility": facility_name,
                "Room": room_name,
                "Capacity": capacity,
                "Occupancy": occupancy,
                "Available": available,
                "ImagePath": temp_path
            })

            st.sidebar.success(f"Room '{room_name}' added successfully!")
        else:
            st.sidebar.error("Please provide facility name, room name, and an image or video.")

# Main Dashboard (Visible to Both Admin and Students)
st.subheader("🏢 Current Room Occupancy Dashboard")

if st.session_state.room_data:
    try:
        df = pd.DataFrame(st.session_state.room_data)
        if not df.empty and all(col in df.columns for col in ["Facility", "Room", "Capacity", "Occupancy", "Available"]):
            if view == "Student":
                selected_facility = st.selectbox("🏛️ Select Facility", df["Facility"].dropna().unique())
                filtered_df = df[df["Facility"] == selected_facility]
                st.dataframe(filtered_df["Room Capacity Occupancy Available".split()])
            else:
                st.dataframe(df["Facility Room Capacity Occupancy Available".split()])

            # Charts Section
            st.subheader("📊 Room Occupancy Analysis")
            chart_df = df if view == "Admin" else filtered_df

            # Static bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(chart_df["Room"], chart_df["Occupancy"], label="Occupied Seats", color='tomato')
            ax.barh(chart_df["Room"], chart_df["Available"], left=chart_df["Occupancy"], label="Available Seats", color='lightgreen')
            ax.set_xlabel("Number of Seats")
            ax.set_title("Seat Occupancy vs Availability")
            ax.legend()
            st.pyplot(fig)

            # Pie chart
            fig2, ax2 = plt.subplots()
            ax2.pie(
                [chart_df["Occupancy"].sum(), chart_df["Available"].sum()],
                labels=["Occupied", "Available"],
                colors=["tomato", "lightgreen"],
                autopct='%1.1f%%',
                startangle=90
            )
            ax2.axis('equal')
            st.pyplot(fig2)

            # View individual room image/video
            selected_room = st.selectbox("🔍 View Room Media", chart_df["Room"].dropna().tolist())
            room_info = next((r for r in st.session_state.room_data if r.get("Room") == selected_room), None)

            if room_info and room_info.get("ImagePath"):
                if room_info["ImagePath"].lower().endswith(".jpg"):
                    st.image(room_info["ImagePath"], caption=f"{selected_room}: {room_info.get('Occupancy', '?')} / {room_info.get('Capacity', '?')} occupied", use_column_width=True)
                else:
                    st.video(room_info["ImagePath"], format="video/mp4")

            # Download CSV Option (Admin only)
            if view == "Admin":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Occupancy Data as CSV",
                    data=csv,
                    file_name='room_occupancy_data.csv',
                    mime='text/csv'
                )
        else:
            st.warning("Room data is incomplete or missing required columns.")
    except Exception as e:
        st.error(f"Error displaying dashboard: {e}")
else:
    st.info("No rooms added yet. Use the sidebar to add rooms.")
