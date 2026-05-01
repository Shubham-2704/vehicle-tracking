"""
Vehicle Tracking & Counting System
Counts vehicles crossing a virtual line with IN/OUT direction
"""

import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os
import sys

def main():
    # Configuration
    VIDEO_PATH = "tracking.mp4"
    CONFIDENCE_THRESHOLD = 0.25
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    
    # Check video
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found!")
        sys.exit(1)
    
    print("Loading model and video...")
    model = YOLO("yolov8n.pt", verbose=False)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Counting line at 50% (middle of frame)
    line_y = int(height * 0.50)
    
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=fps
    )
    
    line_counter = sv.LineZone(
        start=sv.Point(0, line_y), 
        end=sv.Point(width, line_y)
    )
    
    out = cv2.VideoWriter("output_tracked.mp4", 
                          cv2.VideoWriter_fourcc(*"mp4v"), 
                          fps, (width, height))
    
    vehicle_positions = defaultdict(list)
    frame_count = 0
    
    print(f"Processing {total_frames} frames...")
    print(f"Line position: {line_y}px (50% from top)")
    print("-" * 40)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress update every 1000 frames
        if frame_count % 1000 == 0:
            print(f"Frame: {frame_count}/{total_frames} | IN: {line_counter.in_count} | OUT: {line_counter.out_count}")
        
        # Detection
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter vehicles
        vehicle_mask = np.isin(detections.class_id, VEHICLE_CLASSES)
        detections = detections[vehicle_mask]
        
        # Update tracker
        if len(detections) > 0:
            detections = tracker.update_with_detections(detections)
        
        # Store positions for direction
        if len(detections) > 0:
            for xyxy, _, _, tracker_id in zip(detections.xyxy, detections.confidence, 
                                                detections.class_id, detections.tracker_id):
                if tracker_id and tracker_id != -1:
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    vehicle_positions[tracker_id].append(center_y)
                    if len(vehicle_positions[tracker_id]) > 50:
                        vehicle_positions[tracker_id].pop(0)
        
        # Check line crossings
        if len(detections) > 0:
            line_counter.trigger(detections)
        
        # Annotate frame
        annotated = frame.copy()
        
        if len(detections) > 0:
            for xyxy, _, class_id, tracker_id in zip(detections.xyxy, detections.confidence,
                                                       detections.class_id, detections.tracker_id):
                if not tracker_id or tracker_id == -1:
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy)
                positions = vehicle_positions.get(tracker_id, [])
                
                # Determine direction (only if vehicle crossed the line)
                if len(positions) >= 10:
                    first_avg = np.mean(positions[:5])
                    last_avg = np.mean(positions[-5:])
                    direction = "↓ IN" if last_avg > first_avg else "↑ OUT"
                else:
                    direction = "●"
                
                # Color coding
                if class_id == 2:
                    color = (0, 255, 0)  # Car - Green
                elif class_id == 5:
                    color = (255, 165, 0)  # Bus - Orange
                elif class_id == 7:
                    color = (0, 165, 255)  # Truck - Blue
                else:
                    color = (255, 255, 0)  # Other - Yellow
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"#{tracker_id} {direction} {CLASS_NAMES.get(class_id, 'veh')}"
                cv2.putText(annotated, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trajectory trail
                if len(positions) > 1:
                    center_x = (x1 + x2) // 2
                    for j in range(1, len(positions)):
                        cv2.line(annotated, (center_x, int(positions[j-1])), 
                                (center_x, int(positions[j])), (0, 255, 255), 2)
        
        # Draw counting line
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(annotated, "COUNTING LINE", (width//2 - 70, line_y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Statistics panel
        cv2.rectangle(annotated, (5, 5), (220, 95), (0, 0, 0), -1)
        cv2.putText(annotated, f"TOTAL: {line_counter.out_count + line_counter.in_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated, f"IN: {line_counter.in_count}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"OUT: {line_counter.out_count}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        out.write(annotated)
        
        # Display
        cv2.imshow("Vehicle Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final results
    print("-" * 40)
    print(f"\nPROCESSING COMPLETE!")
    print(f"Total Frames: {frame_count}")
    print(f"IN: {line_counter.in_count} | OUT: {line_counter.out_count} | TOTAL: {line_counter.out_count + line_counter.in_count}")
    print(f"Output saved: output_tracked.mp4")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()