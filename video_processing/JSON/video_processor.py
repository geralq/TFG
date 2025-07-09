import cv2
import json
import os
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_signer_pose(video_path, signer_id, output_dir=None):
    if output_dir is None:
        from config import OUTPUT_PATH
        output_dir = OUTPUT_PATH

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    pose_hand_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)

            frame_data = {
                "frame": frame_counter,
                "pose_landmarks": [],
                "hand_landmarks": []
            }

            if pose_results.pose_landmarks:
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    frame_data["pose_landmarks"].append({
                        "label": mp_pose.PoseLandmark(idx).name,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

            detected_hands = []

            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                    hand_data = {
                        "hand": f"Hand_{hand_idx}",
                        "landmarks": []
                    }
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        hand_data["landmarks"].append({
                            "label": mp_hands.HandLandmark(idx).name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    detected_hands.append(hand_data)

            if len(detected_hands) == 1:
                missing_hand_idx = 1 if detected_hands[0]["hand"] == "Hand_0" else 0
                missing_hand_data = {
                    "hand": f"Hand_{missing_hand_idx}",
                    "landmarks": [
                        {**lm, "x": -1, "y": -1, "z": -1}
                        for lm in detected_hands[0]["landmarks"]
                    ]
                }
                detected_hands.append(missing_hand_data)

            frame_data["hand_landmarks"] = detected_hands
            pose_hand_data.append(frame_data)
            frame_counter += 1

    cap.release()

    output_filename = os.path.basename(video_path).replace('.mp4', f'_{signer_id}.json')
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as json_file:
        json.dump(pose_hand_data, json_file, indent=4)

    print(f"Datos guardados en {output_path}")
