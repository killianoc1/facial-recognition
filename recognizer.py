import cv2
import mediapipe as mp
import numpy as np
import random

class FaceRecognizer:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        landmarks = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            landmark_list = []
            for point in face_landmarks.landmark:
                landmark_list.extend([point.x, point.y, point.z])

            landmarks = landmark_list

        return frame, landmarks

    def estimate_head_pose(self, landmarks):
        if not landmarks:
            return None

        full = np.array(landmarks).reshape(-1, 3)
        image_points = []
        for idx in [1, 152, 33, 263, 61, 291]:
            if idx >= len(full):
                return None
            lm = full[idx]
            image_points.append([lm[0] * 640, lm[1] * 480])

        image_points = np.array(image_points, dtype="double")

        frame_w, frame_h = 640, 480
        focal = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            np.array([
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
            ], dtype="double"),
            image_points, camera_matrix, dist_coeffs
        )

        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rot_mat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj)[6]

        pitch, yaw, roll = [np.degrees(a) for a in euler.flatten()]
        return yaw, pitch, roll

    CHALLENGES = [
        ("Turn head LEFT",       lambda y, p: y < -8),
        ("Turn head RIGHT",      lambda y, p: y > 8),
        ("Nod YES (up-down)",    lambda y, p: p < -6),
        ("Shake head NO",        lambda y, p: abs(y) > 14),
        ("Look UP",              lambda y, p: p < -8),
        ("Look DOWN",            lambda y, p: p > 8),
    ]

    def run_liveness_challenges(self, cap, frame, window_name="Recognize Face"):
        num_challenges = random.randint(2, 3)
        selected = random.sample(self.CHALLENGES, num_challenges)

        passed_count = 0
        MAX_FRAMES_PER_CHALLENGE = 150

        for prompt_text, condition in selected:
            print(f"Challenge: {prompt_text}")
            challenge_passed = False
            frame_count = 0

            while frame_count < MAX_FRAMES_PER_CHALLENGE:
                ret, frame = cap.read()
                if not ret:
                    return False

                frame, landmarks = self.process_frame(frame)

                pose = self.estimate_head_pose(landmarks) if landmarks else None

                text = prompt_text if pose is None else f"{prompt_text} | Yaw: {pose[0]:.1f}° Pitch: {pose[1]:.1f}°"
                cv2.putText(frame, text, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

                if pose:
                    yaw, pitch, _ = pose
                    if condition(yaw, pitch):
                        challenge_passed = True
                        break

                frame_count += 1

            if challenge_passed:
                passed_count += 1
                print(f"Passed: {prompt_text}")
            else:
                print(f"Failed: {prompt_text}")
                return False

        print(f"Liveness passed ({passed_count}/{num_challenges} challenges)")
        return True