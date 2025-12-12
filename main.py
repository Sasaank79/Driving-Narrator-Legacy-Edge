import cv2
import time
import os
import argparse
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "models/best_int8_openvino_model/" 
CONF_THRESHOLD = 0.60  # Was 0.55 -> Raised to 0.60 to kill weak detections
TTS_COOLDOWN = 2.0     # Reduced to 2.0s for more frequent narration

# --- CLASS MAPPER (Corrected Order) ---
NARRATOR_MAP = {
    # --- PRIORITY ---
    'stop': "Stop Sign",
    'Stop_Sign': "Stop Sign",
    'Stop 1': "Stop Sign",
    'Give Way': "Yield",
    'give_way': "Yield",
    'Priotiry Road': "Priority Road",
    'priority_road': "Priority Road",
    
    # --- SPEED LIMITS (Specifics First!) ---
    '30km': "Speed Limit 30",
    'limit -30': "Speed Limit 30",
    '50km': "Speed Limit 50",
    '50 mph': "Speed Limit 50",
    'limit -50': "Speed Limit 50",
    '60km': "Speed Limit 60",
    'limit -60': "Speed Limit 60",
    '70km': "Speed Limit 70",
    '80km': "Speed Limit 80",
    'limit -80': "Speed Limit 80",
    '100km': "Speed Limit 100",
    'Speed Limit': "Speed Limit", # Generic fallback
    
    # --- WARNINGS ---
    'Pedestrian': "Caution, Pedestrians",
    'Beware of children': "School Zone",
    'children': "School Zone",
    'Slippery': "Slippery Road",
    'construction': "Road Work",
    'danger': "Danger Ahead",
    'Road narrows': "Road Narrows",
    'Uneven Road': "Bumpy Road",
    'Hump': "Speed Bump",
    
    # --- TURNS ---
    'Turn left': "Left Turn Ahead",
    'Turn right': "Right Turn Ahead",
    'Curve': "Curve Ahead",
    'Round-About': "Roundabout",
    
    # --- PROHIBITORY ---
    'No Entry': "Do Not Enter",
    'no_entry': "Do Not Enter",
    'No_Over_Taking': "No Overtaking",
    'no overtaking': "No Overtaking",
    
    # --- IGNORE LIST ---
    'Keep-Left': None,
    'Keep-Right': None,
    'Traffic_signal': None,
    'undefined': None
}

def get_narration(class_name):
    name = class_name.lower()
    for key, text in NARRATOR_MAP.items():
        if key.lower() in name:
            return text
    return None

def speak(text):
    """Mac-native non-blocking TTS."""
    os.system(f"say '{text}' &")

def run_inference(source, save_output=False):
    print(f"🚀 Loading Narrator Engine: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Handle Source (Webcam index or File Path)
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open source '{source}'")
        return

    # Video Writer Setup (Optional)
    out = None
    if save_output:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle invalid FPS from webcams
        if fps == 0: fps = 15.0 
        
        save_path = "inference_demo.mp4"
        print(f"🎥 Recording enabled. Saving to: {save_path}")
        # mp4v is compatible with macOS QuickTime
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    last_spoken_time = {} 
    print(f"✅ Processing: {source}")
    print("Press 'q' to quit early.")
    
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret: 
            print("End of video stream.")
            break
        
        # INFERENCE
        results = model(frame, imgsz=416, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if hasattr(model, 'names'):
                    raw_name = model.names[cls_id]
                else:
                    raw_name = str(cls_id)

                narration = get_narration(raw_name)
                if narration:
                    current_time = time.time()
                    if (current_time - last_spoken_time.get(narration, 0)) > TTS_COOLDOWN:
                        print(f"🗣️ Narrating: {narration}")
                        speak(narration)
                        last_spoken_time[narration] = current_time

        # FPS Calc
        end_time = time.perf_counter()
        fps_curr = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show & Save
        cv2.imshow("Driving Narrator", annotated_frame)
        if out:
            out.write(annotated_frame)
        
        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            break
            
        # Fast Forward 5 Seconds (Press 'f')
        elif key == ord('f'):
            print("⏩ Skipping 5 seconds...")
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            fps_cap = cap.get(cv2.CAP_PROP_FPS)
            # Jump forward 5 seconds * FPS
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (fps_cap * 5))
            
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Path to video file or webcam index')
    parser.add_argument('--save', action='store_true', help='Save the processed video to file')
    args = parser.parse_args()
    
    run_inference(args.source, args.save)
