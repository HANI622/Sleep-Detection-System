import cv2
import mediapipe as mp
import pygame
import math
import time
from datetime import datetime

from shared import dashboard_data

# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# ALARM
# ==========================================

pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# ==========================================
# MEDIAPIPE FACE MESH
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# FACE DETECTOR
# ==========================================

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ==========================================
# DRAWING
# ==========================================

mp_draw = mp.solutions.drawing_utils

drawing_spec = mp_draw.DrawingSpec(
    thickness=1,
    circle_radius=1
)

# ==========================================
# LEFT EYE LANDMARKS
# ==========================================

LEFT_EYE = [
    33, 7, 163, 144,
    145, 153, 154, 155,
    133, 173,
    157, 158, 159, 160,
    161, 246
]

# ==========================================
# RIGHT EYE LANDMARKS
# ==========================================

RIGHT_EYE = [
    362, 382, 381, 380,
    374, 373, 390, 249,
    263, 466,
    388, 387, 386,
    385, 384, 398
]

# ==========================================
# HEAD LANDMARKS
# ==========================================

NOSE = 1
CHIN = 152

# ==========================================
# SETTINGS
# ==========================================

EAR_THRESHOLD = 0.22
HEAD_THRESHOLD = 0.08
SLEEP_SECONDS = 3

# ==========================================
# VARIABLES
# ==========================================

blink_count = 0

eye_closed = False

sleep_start = None

alarm_playing = False

# ==========================================
# DISTANCE
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

# ==========================================
# EAR
# ==========================================

def eye_ratio(landmarks, eye):

    horizontal = distance(
        landmarks[eye[0]],
        landmarks[eye[8]]
    )

    vertical1 = distance(
        landmarks[eye[12]],
        landmarks[eye[4]]
    )

    vertical2 = distance(
        landmarks[eye[11]],
        landmarks[eye[5]]
    )

    ear = (
        vertical1 +
        vertical2
    ) / (2 * horizontal)

    return ear

# ==========================================
# FRAME GENERATOR
# ==========================================

def generate_frames():

    global blink_count
    global eye_closed
    global sleep_start
    global alarm_playing

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        current_time = datetime.now().strftime("%I:%M:%S %p")

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Face Mesh
        mesh_results = face_mesh.process(rgb)

        # Face Detection
        face_results = face_detector.process(rgb)

                # ==========================================
        # DASHBOARD
        # ==========================================

        

        #cv2.putText(
         #   frame,
          #  "AI SLEEP DETECTOR",
          #  (20, 40),
           # cv2.FONT_HERSHEY_DUPLEX,
           # 0.8,
           # (0, 255, 255),
            #2
        #)

        # ==========================================
        # FACE COUNT
        # ==========================================

        face_count = 0

        if face_results.detections:

            face_count = len(face_results.detections)

            for detection in face_results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "FACE",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Default Values

        status = "NO FACE"
        color = (0, 0, 255)
        ear = 0.0
        sleep_time = 0

                # ==========================================
        # FACE MESH
        # ==========================================

        if mesh_results.multi_face_landmarks:

            face_landmarks = mesh_results.multi_face_landmarks[0]

            landmarks = face_landmarks.landmark

            # --------------------------
            # Draw Left Eye
            # --------------------------

            for idx in LEFT_EYE:

                point = landmarks[idx]

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 255),
                    -1
                )

            # --------------------------
            # Draw Right Eye
            # --------------------------

            for idx in RIGHT_EYE:

                point = landmarks[idx]

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 255),
                    -1
                )

            # =====================================
            # EAR
            # =====================================

            left_ear = eye_ratio(
                landmarks,
                LEFT_EYE
            )

            right_ear = eye_ratio(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear +
                right_ear
            ) / 2

            # =====================================
            # HEAD DOWN
            # =====================================

            nose = landmarks[NOSE]
            chin = landmarks[CHIN]

            head_distance = chin.y - nose.y

            head_down = False

            if head_distance < HEAD_THRESHOLD:

                head_down = True

            # =====================================
            # SLEEP DETECTION
            # =====================================

            if ear < EAR_THRESHOLD:

                if not eye_closed:

                    eye_closed = True

                if sleep_start is None:

                    sleep_start = time.time()

                elapsed = time.time() - sleep_start

                sleep_time = elapsed

                status = "SLEEPING"

                color = (0,0,255)

                if elapsed >= SLEEP_SECONDS:

                    status = "WAKE UP!"

                    if not alarm_playing:

                        pygame.mixer.music.play(-1)

                        alarm_playing = True

            else:

                if eye_closed:

                    blink_count += 1

                    eye_closed = False

                sleep_start = None

                sleep_time = 0

                status = "AWAKE"

                color = (0,255,0)

                if alarm_playing:

                    pygame.mixer.music.stop()

                    alarm_playing = False

            # =====================================
            # HEAD STATUS
            # =====================================

            if head_down:

                cv2.putText(
                    frame,
                    "HEAD DOWN",
                    (420,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )


                        # ==========================================
        # DASHBOARD CARDS
        # ==========================================

        # STATUS CARD
        
        # MULTIPLE FACE WARNING
       
                    # ==========================================
        # FPS COUNTER
        # ==========================================

        fps = int(camera.get(cv2.CAP_PROP_FPS))

        if fps <= 0:
            fps = 30

        

        # ==========================================
        # CAMERA LABEL
        # ==========================================

        

        # ==========================================
        # ENCODE FRAME
        # ==========================================

        # =====================================
        # UPDATE DASHBOARD
        # =====================================

        dashboard_data["status"] = status
        dashboard_data["ear"] = round(ear, 2)
        dashboard_data["blinks"] = blink_count
        dashboard_data["sleep"] = round(sleep_time, 1)
        dashboard_data["faces"] = face_count

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )