import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not working")
    exit()

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

pyautogui.FAILSAFE = False

blink_frames = 0
frames_since_click = 40

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        left_top = landmarks[159]
        left_bottom = landmarks[145]
        
        eye_height = abs(left_top.y - left_bottom.y)
        
        print(f"Eye height: {eye_height}")

        if eye_height < 0.015:
            blink_frames += 1
            if blink_frames >= 2 and frames_since_click > 40:
                pyautogui.click()
                print("CLICKED!")
                frames_since_click = 0
        else:
            blink_frames = 0

        frames_since_click += 1

    cv2.imshow('Eye Controlled Mouse', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows() 