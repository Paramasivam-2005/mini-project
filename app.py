import streamlit as st
import pandas as pd
import numpy as np
import cv2
import time  
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ✅ Streamlit config must be the first command
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# ✅ Custom CSS Styles for Better UI
st.markdown("""
    <style>
        /* Page Background */
        body {
            background-color: #f8f9fa;
            font-family: 'Arial', sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1e3d59;
            color: white;
        }

        /* Buttons */
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #ff3b2f;
            transform: scale(1.05);
        }

        /* Title Styling */
        h1 {
            text-align: center;
            color: #1e3d59;
        }

        /* Image Styling */
        .stImage img {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar Text */
        .stSidebar .stTitle {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load the pre-trained YOLO model
model = YOLO(r"D:\sign_language\best (8).pt")

# ✅ Initialize session state for detected signs
if 'detected_word' not in st.session_state:
    st.session_state.detected_word = ''
if 'letter_buffer' not in st.session_state:
    st.session_state.letter_buffer = ''
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
if 'reset_flag' not in st.session_state:
    st.session_state.reset_flag = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detected_signs_list' not in st.session_state:
    st.session_state.detected_signs_list = []

# ✅ Detection delay
detection_delay = 1.0

# ✅ Home Page
def show_home():
    st.title("🤟 Sign Language Recognition")
    st.write("""Experience real-time sign language recognition with cutting-edge AI.
    Enhance communication accessibility and explore gesture-based controls with ease.""")
    st.image("WhatsApp Image 2024-12-04 at 21.21.18_86e301aa.jpg", caption="Gesture Recognition in Action", use_container_width=True)

# ✅ Video Processor Class for Streamlit WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return frame

# ✅ Application Page
def show_application():
    st.title("📷 Real-Time Gesture Recognition")
    st.write("Enable your camera to demonstrate hand gestures. The app will classify gestures in real-time.")

    # Sidebar
    st.sidebar.title('🎥 Camera Settings')
    confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)
    reset_word = st.sidebar.button("🔄 Reset Detected Word")

    # Reset functionality
    if reset_word:
        st.session_state.reset_flag = True
    if st.session_state.reset_flag:
        st.session_state.detected_word = ''
        st.session_state.detected_signs_list = []
        st.session_state.reset_flag = False
        st.sidebar.success("✅ Detected word has been reset!")

    # Camera options
    use_webcam = st.checkbox('📹 Use Webcam')
    use_ipcam = st.checkbox('🌐 Use IP Camera')
    use_external_camera = st.checkbox('🎥 Use External Camera')

    cap = None
    stframe = st.empty()
    detected_signs_display = st.sidebar.empty()

    # ✅ Handling video input sources
    if use_ipcam:
        st.write("🔗 IP Camera Stream:")
        cap = cv2.VideoCapture("https://192.168.176.180:8080/video")
    elif use_webcam:
        st.write("📷 Webcam Detection:")
        cap = cv2.VideoCapture(0)
    elif use_external_camera:
        st.write("🔌 External Camera Detection:")
        cap = cv2.VideoCapture(2)  # Change device index if necessary
    else:
        st.write("📤 Upload an image for detection:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            st.session_state.uploaded_image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
            st.image(st.session_state.uploaded_image, channels="RGB", use_container_width=True)

    # ✅ Process video frames
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (320, 240))  # Reduce frame size before YOLO processing
                results = model.predict(frame_resized, stream=True)  # Enable streaming mode for efficiency
                current_time = time.time()

                for result in results:
                    for box in result.boxes:
                        if box.conf > confidence_threshold:
                            class_id = int(box.cls[0].item())
                            detected_sign = result.names[class_id]

                            if detected_sign != st.session_state.letter_buffer and (current_time - st.session_state.last_detection_time) > detection_delay:
                                st.session_state.letter_buffer = detected_sign
                                st.session_state.current_letter = detected_sign

                                if len(st.session_state.detected_word) < 50:
                                    st.session_state.detected_word += st.session_state.current_letter

                                st.session_state.last_detection_time = current_time

                detected_signs_display.text("✍ Detected Word: " + st.session_state.detected_word)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels='RGB', use_container_width=True)
            else:
                break
        cap.release()
    else:
        if st.session_state.uploaded_image is not None:
            uploaded_frame = cv2.resize(st.session_state.uploaded_image, (320, 240))  # Resize before YOLO
            results = model.predict(uploaded_frame, stream=True)
            st.session_state.detected_signs_list = []

            for result in results:
                for box in result.boxes:
                    if box.conf > confidence_threshold:
                        class_id = int(box.cls[0].item())
                        detected_sign = result.names[class_id]
                        uploaded_frame = result.plot()

                        if detected_sign not in st.session_state.detected_signs_list:
                            st.session_state.detected_signs_list.append(detected_sign)

            detected_signs_display.text("🤟 Detected Signs: " + ', '.join(st.session_state.detected_signs_list))
            uploaded_frame = cv2.cvtColor(uploaded_frame, cv2.COLOR_BGR2RGB)
            stframe.image(uploaded_frame, channels='RGB', use_container_width=True)
        else:
            st.write("⚠ No video source available.")

# ✅ Contact Page
def show_contact():
    st.title("📞 About")
    st.write("Making communication more accessible by recognizing hand gestures in real-time.")
    st.write("📧 **Email:** kavinkarthick.cs23@bitsathy.ac.in, paramasivam.cs23@bitsathy.ac.in, yasiryouhana.cs23@bitsathy.ac.in, jagan.ct23@bitsathy.ac.in")
    st.write("📞 **Phone:** 8807253798")

# ✅ Main Function
def main():
    st.sidebar.title("📌 Navigation")
    option = st.sidebar.selectbox("Go to", ["Home", "Application", "About"])

    if option == "Home":
        show_home()
    elif option == "Application":
        show_application()
    elif option == "About":
        show_contact()

if __name__ == "__main__":
    main()