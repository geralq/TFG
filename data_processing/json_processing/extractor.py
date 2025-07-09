def is_hand_missing(landmarks):
    for lm in landmarks:
        if not (lm['x'] == -1 and lm['y'] == -1 and lm['z'] == -1):
            return False
    return True

def extract_pose_landmarks(pose_landmarks, row, prefix='pose'):
    left_labels = ['LEFT_ANKLE', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_THUMB', 'LEFT_WRIST']
    right_labels = ['RIGHT_ANKLE', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_THUMB', 'RIGHT_WRIST']
    face = ['LEFT_EYE_INNER', 'RIGHT_EYE_INNER','LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER','RIGHT_EYE', 'LEFT_EYE','LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']

    for lm in pose_landmarks:
        if lm['label'] not in left_labels and lm['label'] not in right_labels and lm['label'] not in face:
            label = lm['label']
            row[f'{prefix}.{label}_x'] = lm['x']
            row[f'{prefix}.{label}_y'] = lm['y']
            row[f'{prefix}.{label}_z'] = lm['z']
    return row

def extract_hand_landmarks(hand_data, row, hand_name='Hand_0'):
    excluded_labels = {
        "INDEX_FINGER_DIP", "INDEX_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_PIP",
        "PINKY_DIP", "PINKY_PIP", "RING_FINGER_DIP", "RING_FINGER_PIP",
        "THUMB_CMC", "THUMB_IP"
    }

    for lm in hand_data['landmarks']:
        label = lm['label']
        if label in excluded_labels:
            continue
        if is_hand_missing(hand_data['landmarks']):
            row[f'{hand_name}.{label}_x'] = -1
            row[f'{hand_name}.{label}_y'] = -1
            row[f'{hand_name}.{label}_z'] = -1
        else:
            row[f'{hand_name}.{label}_x'] = lm['x']
            row[f'{hand_name}.{label}_y'] = lm['y']
            row[f'{hand_name}.{label}_z'] = lm['z']
    return row
