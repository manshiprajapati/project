import cv2
import mediapipe as mp
import numpy as np
import time
import math

EXERCISES = [
    {"name": "Look RIGHT", "instruction": "Move eyes to the RIGHT", "target": "right", "duration": 5},
    {"name": "Look LEFT", "instruction": "Move eyes to the LEFT", "target": "left", "duration": 5},
    {"name": "Look UP", "instruction": "Roll eyes upward", "target": "up", "duration": 5},
    {"name": "Look DOWN", "instruction": "Roll eyes downward", "target": "down", "duration": 5},
    {"name": "Roll Eyes Clockwise", "instruction": "Slowly roll your eyes in a circle", "target": "roll", "duration": 8},
    {"name": "Rest", "instruction": "Close your eyes and relax", "target": "rest", "duration": 8},
]


mp_face_mesh = mp.solutions.face_mesh  # Setup mediapipe for face detection
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS = [474, 475, 476, 477, 478]
RIGHT_IRIS = [469, 470, 471, 472, 473]   # Landmark points for eyes and iris
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373]

def get_iris_position(landmarks, iris_indices, eye_indices, w, h):
   
    try:
        iris_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices]
        iris_center = np.mean(iris_points, axis=0)
        
        eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        eye_left = min(p[0] for p in eye_points)
        eye_right = max(p[0] for p in eye_points)
        eye_top = min(p[1] for p in eye_points)
        eye_bottom = max(p[1] for p in eye_points)
        
        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top
        
        if eye_width > 0 and eye_height > 0:
            h_ratio = (iris_center[0] - eye_left) / eye_width
            v_ratio = (iris_center[1] - eye_top) / eye_height
            return h_ratio, v_ratio
    except:
        pass
    
    return 0.5, 0.5

def determine_gaze_direction(h_ratio, v_ratio):
    direction = "center"  # figures out which direction you're looking based on iris position
    if h_ratio < 0.35:
        direction = "right"
    elif h_ratio > 0.65:
        direction = "left"
    elif v_ratio < 0.35:
        direction = "up"
    elif v_ratio > 0.65:
        direction = "down"
    
    return direction

def draw_rounded_box(img, x1, y1, x2, y2, color, alpha=0.7):  # Drawing stuff on screen
    overlay = img.copy() # draws a box with transparency
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_text(img, text, x, y, scale=1.0, color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def draw_progress_bar(img, x, y, w, h, progress, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    filled_width = int(w * progress)
    cv2.rectangle(img, (x, y), (x + filled_width, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)

def main():
    print("\nEye Exercise Program")
    print("Controls:")
    print("  SPACE - Skip current exercise")
    print("  Q/ESC - Quit\n")
    
    cap = cv2.VideoCapture(0) # Open camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    phase = "intro" 
    exercise_idx = 0
    exercise_start_time = None
    countdown_start_time = None
    score = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # mirroring the camera
        current_time = time.time()
    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# process face detection with mediapipe
        results = face_mesh.process(rgb_frame)
        
        gaze_direction = "center"  # figuring out where the user is looking
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_h, left_v = get_iris_position(landmarks, LEFT_IRIS, LEFT_EYE, W, H)
            right_h, right_v = get_iris_position(landmarks, RIGHT_IRIS, RIGHT_EYE, W, H)
            
            avg_h = (left_h + right_h) / 2
            avg_v = (left_v + right_v) / 2
            
            gaze_direction = determine_gaze_direction(avg_h, avg_v)
        
        if phase == "intro":     # screen intro
            draw_rounded_box(frame, W//2 - 300, H//2 - 150, W//2 + 300, H//2 + 150, (20, 20, 60))
            draw_text(frame, "Eye Exercise", W//2 - 250, H//2 - 80, 1.5, (0, 255, 255), 3)
            draw_text(frame, f"{len(EXERCISES)} exercises to strengthen your eyes", W//2 - 280, H//2 - 20, 0.7, (200, 200, 200), 2)
            draw_text(frame, "Press SPACE to start", W//2 - 180, H//2 + 50, 1.0, (0, 255, 150), 2)
            draw_text(frame, "Press Q to quit", W//2 - 120, H//2 + 100, 0.6, (150, 150, 150), 1)
        
        elif phase == "countdown":  # COUNTDOWN before each exercise
            exercise = EXERCISES[exercise_idx]
            remaining = 3 - (current_time - countdown_start_time)
            
            if remaining <= 0:
                phase = "exercise"
                exercise_start_time = current_time
            else:
                count = math.ceil(remaining)
                draw_text(frame, exercise["name"], W//2 - 150, H//2 - 100, 1.2, (100, 200, 255), 2)
                draw_text(frame, exercise["instruction"], W//2 - 200, H//2 - 50, 0.8, (200, 200, 200), 2)
                draw_text(frame, str(count), W//2 - 30, H//2 + 80, 5.0, (0, 255, 255), 5)
        
        
        elif phase == "exercise":
            exercise = EXERCISES[exercise_idx]
            elapsed = current_time - exercise_start_time
            remaining = max(0, exercise["duration"] - elapsed)
            progress = min(1.0, elapsed / exercise["duration"])
            
            # check if time is up
            if elapsed >= exercise["duration"]:
                # give points if they did it right
                if exercise["target"] == "rest" or exercise["target"] == "roll" or gaze_direction == exercise["target"]:
                    score += 10
                
                exercise_idx += 1
                
                if exercise_idx >= len(EXERCISES):
                    phase = "complete"
                else:
                    phase = "countdown"
                    countdown_start_time = current_time
                continue
            
            # show current exercise info
            draw_rounded_box(frame, 20, 20, 500, 180, (30, 30, 30))
            draw_text(frame, f"Exercise {exercise_idx + 1}/{len(EXERCISES)}", 40, 60, 0.8, (150, 150, 150), 2)
            draw_text(frame, exercise["name"], 40, 100, 1.2, (100, 200, 255), 2)
            draw_text(frame, exercise["instruction"], 40, 140, 0.7, (200, 200, 200), 2)
            
            # progress bar
            draw_progress_bar(frame, 20, 190, 480, 20, progress, (0, 255, 100))
            
            # show remaining time
            draw_text(frame, f"Time: {remaining:.1f}s", 40, 250, 1.0, (255, 255, 0), 2)
            
            # score display
            draw_text(frame, f"Score: {score}", W - 250, 60, 1.0, (255, 200, 0), 2)
            
            # show what the user is doing
            if not face_detected:
                status_text = "No face detected"
                status_color = (0, 0, 255)
            elif exercise["target"] == "rest":
                status_text = "Relax your eyes"
                status_color = (255, 255, 255)
            elif exercise["target"] == "roll":
                status_text = "Keep rolling your eyes slowly"
                status_color = (255, 200, 100)
            elif gaze_direction == exercise["target"]:
                status_text = f"Correct! ({gaze_direction.upper()})"
                status_color = (0, 255, 0)
            else:
                status_text = f"Current: {gaze_direction.upper()}"
                status_color = (255, 150, 0)
            
            draw_rounded_box(frame, 20, H - 100, 500, H - 30, (30, 30, 30))
            draw_text(frame, status_text, 40, H - 55, 1.0, status_color, 2)
        
        # COMPLETE SCREEN
        elif phase == "complete":
            draw_rounded_box(frame, W//2 - 300, H//2 - 150, W//2 + 300, H//2 + 150, (20, 60, 20))
            draw_text(frame, "SESSION COMPLETE!", W//2 - 220, H//2 - 80, 1.5, (0, 255, 100), 3)
            draw_text(frame, "Great job! Your eyes thank you!", W//2 - 250, H//2 - 20, 0.9, (200, 200, 200), 2)
            draw_text(frame, f"Final Score: {score}", W//2 - 150, H//2 + 40, 1.2, (255, 255, 0), 2)
            draw_text(frame, "SPACE = Restart  |  Q = Quit", W//2 - 200, H//2 + 100, 0.7, (150, 150, 150), 2)
        
        # show the frame
        cv2.imshow("Eye Exercise", frame)
        
        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        elif key == ord(' '):  
            if phase == "intro":
                phase = "countdown"
                countdown_start_time = current_time
            elif phase == "exercise":
                exercise_idx += 1
                if exercise_idx >= len(EXERCISES):
                    phase = "complete"
                else:
                    phase = "countdown"
                    countdown_start_time = current_time
            elif phase == "complete":
                phase = "intro"
                exercise_idx = 0
                score = 0
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    
    print(f"\nSession ended. Final score: {score}")
    print("Thanks for using EyeZen!\n")

if __name__ == "__main__":
    main()