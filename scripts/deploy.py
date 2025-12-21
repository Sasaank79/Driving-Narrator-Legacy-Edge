"""
Multithreaded Deploy - Real-time traffic sign detection
Uses INT8 OpenVINO model for CPU inference
"""
import cv2
import time
from ultralytics import YOLO
from threading import Thread
import queue
import argparse

# Shared queue for frames
frame_queue = queue.Queue(maxsize=5)
stop_flag = False

def video_reader(cap):
    """Background thread: continuously reads frames"""
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass

def run_inference(video_path, model_path, conf=0.50):
    global stop_flag
    
    print(f"Loading model from: {model_path}...")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"ERROR: Could not load model.\n{e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return

    # Warmup
    print("Warming up model...")
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            model(cv2.resize(frame, (416, 416)), verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Start background reader thread
    reader_thread = Thread(target=video_reader, args=(cap,), daemon=True)
    reader_thread.start()
    
    prev_time = time.time()
    fps_history = []
    frame_count = 0
    
    print("Starting inference... Press 'q' to exit.")

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        resized = cv2.resize(frame, (416, 416))
        results = model(resized, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        fps_history.append(fps)
        frame_count += 1
        
        if frame_count % 30 == 0:
            avg_fps = sum(fps_history[-30:]) / min(30, len(fps_history))
            print(f"  Avg FPS (last 30): {avg_fps:.1f}")
        
        cv2.putText(annotated_frame, f"FPS: {int(fps)} | INT8", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Driving Narrator", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
    
    if fps_history:
        print(f"\n=== FINAL STATS ===")
        print(f"Frames processed: {len(fps_history)}")
        print(f"Average FPS: {sum(fps_history) / len(fps_history):.1f}")
        print(f"Max FPS: {max(fps_history):.1f}")
        print(f"Min FPS: {min(fps_history):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time traffic sign detection')
    parser.add_argument('--video', default='test_video.mp4', help='Video file path')
    parser.add_argument('--model', default='models/best_int8_openvino_model/', help='Model path')
    parser.add_argument('--conf', type=float, default=0.50, help='Confidence threshold')
    args = parser.parse_args()
    
    run_inference(args.video, args.model, args.conf)
