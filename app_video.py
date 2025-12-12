import cv2
from ultralytics import YOLO
import time
import os
import subprocess

# ==========================================
# CONFIGURATION
# ==========================================
# OpenVINO INT8 model (fastest - 28.6 FPS)
POSSIBLE_MODEL_PATHS = [
    "models/best_int8_openvino_model",
    "models/best_openvino_model", 
    "models/best.pt"  # Fallback to PyTorch
]
VIDEO_SOURCE = "test_video.mp4"
CONFIDENCE_THRESHOLD = 0.5     
COOLDOWN_SECONDS = 5  # Increased to prevent rapid detections

# ==========================================
# SETUP
# ==========================================
# 1. Find Model
model_path = next((p for p in POSSIBLE_MODEL_PATHS if os.path.exists(p)), None)
if not model_path:
    print("❌ ERROR: Could not find 'best.pt'!")
    exit()

# 2. Check Video
if not os.path.exists(VIDEO_SOURCE):
    print(f"❌ ERROR: Could not find video file: {VIDEO_SOURCE}")
    print("Make sure 'test_video.mp4' is in this folder.")
    exit()

print(f"🚀 Loading Model: {model_path}")
print(f"🎬 Loading Video: {VIDEO_SOURCE}")

# 3. Setup Narrator (single speech at a time)
current_speech = None

def speak(text):
    """Non-blocking macOS TTS - suppresses error messages."""
    global current_speech
    if current_speech and current_speech.poll() is None:
        return  # Previous speech still going, skip
    current_speech = subprocess.Popen(
        ["say", text], 
        stderr=subprocess.DEVNULL
    )

# 4. Load Model (OpenVINO or PyTorch)
if 'openvino' in model_path:
    model = YOLO(model_path, task='detect')
    print("✅ Using OpenVINO INT8 (fastest)")
else:
    model = YOLO(model_path)
    try:
        model.to('mps')
        print("✅ Using Mac GPU (MPS)")
    except:
        print("⚠️ Using Standard CPU.")

# ==========================================
# MAIN LOOP (VIDEO FILE)
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
last_spoken_time = {}

print("▶️ Playing Video... Press 'Q' to Exit, 'F' to skip 5 seconds.")

while cap.isOpened():
    start_time = time.perf_counter()
    
    success, frame = cap.read()
    if not success: 
        print("End of video.")
        break

    # Inference
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated_frame = results[0].plot()
    
    # Logic - only speak ONE sign per frame
    current_time = time.time()
    spoken_this_frame = False
    
    for box in results[0].boxes:
        if spoken_this_frame:
            break
            
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        last_time = last_spoken_time.get(class_name, 0)
        if current_time - last_time > COOLDOWN_SECONDS:
            alert = f"Detected {class_name}"
            print(f"🗣️ {alert}")
            speak(alert)
            last_spoken_time[class_name] = current_time
            spoken_this_frame = True

    # FPS Counter
    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Driving Narrator (Video)", annotated_frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Fast forward 5 seconds
        print("⏩ Skipping 5 seconds...")
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (video_fps * 5))

cap.release()
cv2.destroyAllWindows()
