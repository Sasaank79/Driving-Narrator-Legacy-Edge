import cv2
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "test_video.mp4"
MODEL_PATH = "models/best_int8_openvino_model/"
# ---------------------

def run_inference():
    print(f"Loading OpenVINO model from: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"ERROR: Could not load model.\n{e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}")
        return

    prev_time = 0
    print("Starting Inference... Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        start_time = time.time()

        # Run Inference
        results = model(frame, imgsz=416, conf=0.50, verbose=False)

        # Plot results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Overlay FPS on screen
        cv2.putText(annotated_frame, f"FPS: {int(fps)} (OpenVINO INT8)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Driving Narrator - Edge Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
