import cv2
import itertools
import time
import numpy as np
from deepface import DeepFace
from recognizer import FaceRecognizer
import database

EMBEDDING_THRESHOLD = 0.80  # Adjust as needed after testing


def show_spinner(frame, text="Processing", window_name="Face Scan", duration=0.8):
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    end_time = time.time() + duration
    while time.time() < end_time:
        spin_char = next(spinner)
        display_text = f"{text} {spin_char}"
        cv2.putText(frame, display_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(100)


def register_face(recognizer):
    name = input("Enter name to register: ").strip()
    if not name:
        print("Registration cancelled.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    embeddings = []

    print(f"You will scan 5 times for '{name}' to improve accuracy.")
    print("Press 's' to capture each scan. Press 'q' to quit early.")

    num_scans = 0
    while num_scans < 5:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        frame, landmarks = recognizer.process_frame(frame)

        status = f"Scan {num_scans+1}/5 - Face Detected - Press 's'" if landmarks else "No Face Detected"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and landmarks:
            show_spinner(frame, f"Scanning {num_scans+1}/5", "Register Face", duration=1.0)

            embedding = None
            try:
                # Adaptive crop: scale based on inter-eye distance
                left_eye = np.array(landmarks).reshape(-1, 3)[33]
                right_eye = np.array(landmarks).reshape(-1, 3)[263]
                eye_dist_norm = np.linalg.norm(left_eye - right_eye)
                face_width_px = int(eye_dist_norm * frame.shape[1] * 4.0)  # 4× inter-eye ≈ full face

                # Nose as center
                nose_lm = np.array(landmarks).reshape(-1, 3)[1]
                nose_x = int(nose_lm[0] * frame.shape[1])
                nose_y = int(nose_lm[1] * frame.shape[0])

                x_min = max(0, nose_x - face_width_px // 2)
                x_max = min(frame.shape[1], nose_x + face_width_px // 2)
                y_min = max(0, nose_y - face_width_px // 2)
                y_max = min(frame.shape[0], nose_y + face_width_px // 2)

                face_crop = frame[y_min:y_max, x_min:x_max]

                embedding = DeepFace.represent(face_crop, model_name="ArcFace",
                                               detector_backend="skip",
                                               enforce_detection=True)[0]["embedding"]
                print(f"Scan {num_scans+1}/5 complete. Embedding length: {len(embedding)} (adaptive ~{face_width_px}px)")
            except Exception as crop_error:
                print(f"Adaptive crop failed: {crop_error} - falling back to fixed 500px")
                try:
                    nose_lm = np.array(landmarks).reshape(-1, 3)[1]
                    nose_x = int(nose_lm[0] * frame.shape[1])
                    nose_y = int(nose_lm[1] * frame.shape[0])

                    crop_size = 500
                    x_min = max(0, nose_x - crop_size // 2)
                    x_max = min(frame.shape[1], nose_x + crop_size // 2)
                    y_min = max(0, nose_y - crop_size // 2)
                    y_max = min(frame.shape[0], nose_y + crop_size // 2)

                    face_crop = frame[y_min:y_max, x_min:x_max]

                    embedding = DeepFace.represent(face_crop, model_name="ArcFace",
                                                   detector_backend="skip",
                                                   enforce_detection=True)[0]["embedding"]
                    print(f"Scan {num_scans+1}/5 complete. Embedding length: {len(embedding)} (fallback fixed 500px)")
                except Exception as fallback_e:
                    print(f"Fallback failed: {fallback_e}")
                    continue

            if embedding:
                embeddings.append(embedding)
                num_scans += 1
                time.sleep(1.5)  # pause before next scan

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) > 0:
        database.insert_user(name, embeddings)
        print(f"Successfully registered {len(embeddings)} scans for {name}")
    else:
        print("Registration failed - no valid scans.")


def recognize_face(recognizer):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 's' to scan face. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = recognizer.process_frame(frame)

        if landmarks:
            status = "Face Detected"
            color = (0, 255, 0)
        else:
            status = "No Face Detected"
            color = (0, 0, 255)

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Recognize Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s') and landmarks:
            show_spinner(frame, "Scanning", "Recognize Face", duration=1.0)

            # Anti-spoofing
            try:
                spoof_result = DeepFace.extract_faces(frame, anti_spoofing=True)
                if spoof_result and len(spoof_result) > 0 and not spoof_result[0].get('is_real', True):
                    result_text = "Spoof Detected"
                    print(result_text)
                    cv2.putText(frame, result_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow("Recognize Face", frame)
                    cv2.waitKey(3000)
                    continue
                print("Anti-spoofing passed")
            except Exception as e:
                print(f"Anti-spoof error (continuing): {e}")

            # ==============================================
            # PROBE EMBEDDING CAPTURED HERE — BEFORE CHALLENGES
            # Face is still in relatively neutral position
            # ==============================================
            probe_emb = None
            try:
                # Adaptive crop: scale based on inter-eye distance
                left_eye = np.array(landmarks).reshape(-1, 3)[33]
                right_eye = np.array(landmarks).reshape(-1, 3)[263]
                eye_dist_norm = np.linalg.norm(left_eye - right_eye)
                face_width_px = int(eye_dist_norm * frame.shape[1] * 4.0)

                # Nose as center
                nose_lm = np.array(landmarks).reshape(-1, 3)[1]
                nose_x = int(nose_lm[0] * frame.shape[1])
                nose_y = int(nose_lm[1] * frame.shape[0])

                x_min = max(0, nose_x - face_width_px // 2)
                x_max = min(frame.shape[1], nose_x + face_width_px // 2)
                y_min = max(0, nose_y - face_width_px // 2)
                y_max = min(frame.shape[0], nose_y + face_width_px // 2)

                face_crop = frame[y_min:y_max, x_min:x_max]

                probe_emb = DeepFace.represent(face_crop, model_name="ArcFace",
                                               detector_backend="skip",
                                               enforce_detection=True)[0]["embedding"]
                print(f"Probe embedding length: {len(probe_emb)} (pre-challenge neutral pose)")
            except Exception as e:
                print(f"Embedding failed before challenges: {e}")
                # Optional: try fallback here if you want
                continue

            # Liveness challenges now run AFTER embedding capture
            if not recognizer.run_liveness_challenges(cap, frame, "Recognize Face"):
                result_text = "Liveness Failed"
                print(result_text)
                cv2.putText(frame, result_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Recognize Face", frame)
                cv2.waitKey(3000)
                continue

            # Now compare the neutral probe_emb to stored samples
            users = database.get_all_users()
            if not users:
                result_text = "No registered faces in database"
                print(result_text)
                continue

            best_match = None
            lowest_dist = float('inf')

            print("\nArcFace distances:")
            for name, stored_embs in users:
                for i, stored_emb in enumerate(stored_embs):
                    dist = np.linalg.norm(np.array(probe_emb) - np.array(stored_emb))
                    print(f"  {name} (sample {i+1}): {dist:.5f}")
                    if dist < lowest_dist:
                        lowest_dist = dist
                        best_match = name

            if lowest_dist < EMBEDDING_THRESHOLD:
                confidence = max(0, 1 - lowest_dist)
                result_text = f"Match: {best_match} (dist: {lowest_dist:.4f}, conf: {confidence:.2f})"
                color = (0, 255, 0)
            else:
                result_text = f"No match (best dist: {lowest_dist:.4f})"
                color = (0, 0, 255)

            print(result_text)
            cv2.putText(frame, result_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Recognize Face", frame)
            cv2.waitKey(5000)

    cap.release()
    cv2.destroyAllWindows()