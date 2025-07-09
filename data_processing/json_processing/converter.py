import json
import pandas as pd
from .extractor import extract_pose_landmarks, extract_hand_landmarks

def json_to_csv(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict) and 'frames' in data:
        frames = data['frames']
    else:
        print("Formato JSON no reconocido.")
        return

    if not frames:
        print("No se encontraron fotogramas en el JSON.")
        return

    rows = []
    for frame_info in frames:
        row = {'frame': frame_info.get('frame', -1)}
        row = extract_pose_landmarks(frame_info.get('pose_landmarks', []), row)
        for hand in frame_info.get('hand_landmarks', []):
            row = extract_hand_landmarks(hand, row, hand_name=hand['hand'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by='frame', inplace=True)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(-1, inplace=True)

    fieldnames = list(df.columns)
    fieldnames.remove('frame')
    fieldnames = ['frame'] + sorted(fieldnames)
    df = df[fieldnames]

    output_csv = json_file.replace('.json', '.csv')
    df.to_csv(output_csv, index=False)
    print(f"CSV generado con interpolación en: {output_csv}")
