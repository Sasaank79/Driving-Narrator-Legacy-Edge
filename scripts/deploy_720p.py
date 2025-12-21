"""
720p Deploy with Configurable Inference Size
- Display: Original resolution (720p or source)
- Inference: 416 or 640 (configurable)
- Higher confidence threshold to reduce false positives
"""
import cv2
import time
from ultralytics import YOLO
from threading import Thread
import queue
import argparse

frame_queue = queue.Queue(maxsize=5)
stop_flag = False

def video_reader(cap):
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

def run_deploy(video_path, model_path, imgsz=416, conf=0.65):
    global stop_flag
    
    print(f"Loading model: {model_path}")
    print(f"Inference size: {imgsz}x{imgsz}")
    print(f"Confidence threshold: {conf}")
    
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"ERROR: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Display resolution: {video_width}x{video_height}")

    # Warmup
    print("Warming up...")
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            model(cv2.resize(frame, (imgsz, imgsz)), verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Start reader thread
    reader_thread = Thread(target=video_reader, args=(cap,), daemon=True)
    reader_thread.start()
    
    prev_time = time.time()
    fps_history = []
    
    print(f"\nPress 'q' to quit, '+'/'-' to adjust confidence\n")

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        # Inference on resized frame
        resized = cv2.resize(frame, (imgsz, imgsz))
        results = model(resized, conf=conf, verbose=False)
        
        # Get detections and scale back to original size
        boxes = results[0].boxes
        
        # Draw on original frame
        scale_x = video_width / imgsz
        scale_y = video_height / imgsz
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_val:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        
        # Print FPS every 30 frames
        if len(fps_history) % 30 == 0:
            avg = sum(fps_history[-30:]) / 30
            print(f"  Avg FPS (last 30): {avg:.1f}")
        
        # Overlay info
        info_text = f"FPS: {int(fps)} | Infer: {imgsz}x{imgsz} | Conf: {conf:.2f}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Display: {video_width}x{video_height}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Driving Narrator V3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf = min(0.95, conf + 0.05)
            print(f"Confidence: {conf:.2f}")
        elif key == ord('-'):
            conf = max(0.10, conf - 0.05)
            print(f"Confidence: {conf:.2f}")

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
    
    if fps_history:
        print(f"\n=== STATS ===")
        print(f"Frames: {len(fps_history)}")
        print(f"Avg FPS: {sum(fps_history)/len(fps_history):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='test_video.mp4')
    parser.add_argument('--model', default='models/best_int8_openvino_model/')
    parser.add_argument('--imgsz', type=int, default=416, help='416 or 640')
    parser.add_argument('--conf', type=float, default=0.65, help='Confidence threshold')
    args = parser.parse_args()
    
    run_deploy(args.video, args.model, args.imgsz, args.conf)
