import pandas as pd
import json
import os

def is_hand_missing(landmarks):
    for lm in landmarks:
        if not (lm['x'] == -1 and lm['y'] == -1 and lm['z'] == -1):
            return False
    return True

def process_json_video(file):
    pose_landmarks = []
    hand_landmarks = []
    frames = []

    for frame in file:
        if not frame['hand_landmarks']:
            continue
        else:
            frames.append(frame['frame'])
            pose_landmarks.append(frame['pose_landmarks'])
            hand_landmarks.append(frame['hand_landmarks'])

    left_labels = ['LEFT_ANKLE', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_THUMB', 'LEFT_WRIST']
    right_labels = ['RIGHT_ANKLE', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_THUMB', 'RIGHT_WRIST']
    face = ['LEFT_EYE_INNER', 'RIGHT_EYE_INNER','LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER','RIGHT_EYE', 'LEFT_EYE','LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']

    excluded_labels = {
        "INDEX_FINGER_DIP", "INDEX_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_PIP",
        "PINKY_DIP", "PINKY_PIP", "RING_FINGER_DIP", "RING_FINGER_PIP",
        "THUMB_CMC", "THUMB_IP"
    }

    rows = []

    for i in range(len(pose_landmarks)):
        row = {}
        row['frame'] = frames[i]
        for j in range(len(pose_landmarks[i])):
            if not isinstance(pose_landmarks[i][j], dict):
                continue
            label = pose_landmarks[i][j]['label']
            if label not in left_labels and label not in right_labels and label not in face:
                row[f'pose.{label}_x'] = pose_landmarks[i][j]['x']
                row[f'pose.{label}_y'] = pose_landmarks[i][j]['y']
                row[f'pose.{label}_z'] = pose_landmarks[i][j]['z']

        for j in range(len(hand_landmarks[i])):
            hand_name = hand_landmarks[i][j]['hand']
            for landmark in hand_landmarks[i][j]['landmarks']:
                label = landmark['label']
                if label in excluded_labels:
                    continue
                if is_hand_missing(hand_landmarks[i][j]['landmarks']):
                    row[f'{hand_name}.{label}_x'] = -1
                    row[f'{hand_name}.{label}_y'] = -1
                    row[f'{hand_name}.{label}_z'] = -1
                else:
                    row[f'{hand_name}.{label}_x'] = landmark['x']
                    row[f'{hand_name}.{label}_y'] = landmark['y']
                    row[f'{hand_name}.{label}_z'] = landmark['z']

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by='frame', inplace=True)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(-1, inplace=True)

    fieldnames = list(df.columns)
    fieldnames.remove('frame')
    fieldnames = ['frame'] + sorted(fieldnames)
    df = df[fieldnames]

    return df

folder = '/home/gerardo/LSE_DATABASE/LSE_HEALTH'

output_folder = '/home/gerardo/LSE_SPOT_DATASET'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(folder):
    if file.endswith('.json'):
        json_file = json.load(open(os.path.join(folder, file)))
        df = process_json_video(json_file)
        df.to_csv(os.path.join(output_folder, file.replace('.json', '.csv')), index=False)
    else:
        print(f'Skipped {file}, not a JSON file.')
