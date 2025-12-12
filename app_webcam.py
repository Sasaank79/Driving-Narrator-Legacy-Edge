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
CONFIDENCE_THRESHOLD = 0.5     
COOLDOWN_SECONDS = 5  # Increased to prevent rapid detections

# ==========================================
# SETUP
# ==========================================
# 1. Find Model
model_path = next((p for p in POSSIBLE_MODEL_PATHS if os.path.exists(p)), None)
if not model_path:
    print("❌ ERROR: Could not find 'best.pt'! Check your folders.")
    exit()

print(f"🚀 Loading Model: {model_path}")

# 2. Setup Narrator (single speech at a time)
current_speech = None

def speak(text):
    """Non-blocking macOS TTS - suppresses error messages."""
    global current_speech
    if current_speech and current_speech.poll() is None:
        return  # Previous speech still going, skip
    # Redirect stderr to suppress "Open speech channel failed" messages
    current_speech = subprocess.Popen(
        ["say", text], 
        stderr=subprocess.DEVNULL
    )

# 3. Load Model (OpenVINO or PyTorch)
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
# MAIN LOOP (WEBCAM)
# ==========================================
print("🎥 Starting Webcam... Press 'Q' to Exit.")
cap = cv2.VideoCapture(0)

last_spoken_time = {}

while cap.isOpened():
    start_time = time.perf_counter()
    
    success, frame = cap.read()
    if not success: break

    # Inference
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated_frame = results[0].plot()
    
    # Logic - only speak ONE sign per frame (highest confidence)
    current_time = time.time()
    spoken_this_frame = False
    
    for box in results[0].boxes:
        if spoken_this_frame:
            break  # Already spoke this frame
            
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Cooldown check
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

    cv2.imshow("Driving Narrator (Webcam)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
