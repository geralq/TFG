import json
import pandas as pd
import cv2
import os
from sklearn import preprocessing
import re


signer_bow = pd.read_excel('/home/gerardo/LSE_HEALTH/LSE_TFG/video_processing/signer_bow.xlsx')
words = signer_bow.columns[1:]

dfs = pd.read_excel('/home/gerardo/LSE_HEALTH/LSE-Health-UVigo.xlsx', sheet_name=None)
GlossesContent = dfs['GlossesContent']

label_encoder = preprocessing.LabelEncoder()
labeled_words = label_encoder.fit_transform(words)
df = pd.DataFrame({'Word': words, 'Encoded_Label': labeled_words})
csv_file_path = 'labeled_words.csv'
df.to_csv(csv_file_path, index=False)
os.makedirs('/home/gerardo/WORD_POSE_DATASET', exist_ok=True)
for label in labeled_words:
    os.makedirs(f'/home/gerardo/WORD_POSE_DATASET/{label}', exist_ok=True)

def get_nonzero_words_for_signer(signer_number, data_frame):
    signer_row = data_frame[data_frame['Signer'] == signer_number]
    if signer_row.empty:
        return None
    words = []
    for col in signer_row.columns[1:]:
        if signer_row[col].values[0] > 0:
            words.append(col)
    return words

def extract_words_from_json(json_file):
    basename = os.path.basename(json_file)
    signer = re.search(r'_(\d+)(?=\.json)', basename).group(1)
    video = re.sub(r'_[0-9]+\.json$', '', basename)
    
    words = get_nonzero_words_for_signer(int(signer), signer_bow)
    if words is None:
        print(f"No data found for signer: {signer}. Skipping file: {json_file}")
        return
    
    gloss = GlossesContent.loc[GlossesContent['Gloss'].isin(words)]
    elan_json = gloss.loc[gloss['File'] == video]
    
    video_path = os.path.join('/home/gerardo/LSE_HEALTH/Videos-LSE-Health-UVigo/Videos-LSE-Health-UVigo', f'{video}.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    with open(json_file, 'r') as json_file_handle:
        data  = json.load(json_file_handle)
    
    for _, row in elan_json.iterrows():
        gloss_value = row['Gloss']
        start = row['Start(ms)']
        end = row['End(ms)']
        
        start_frame = int(start / 1000 * fps)
        end_frame = int(end / 1000 * fps)

        filtered_frames = [frame for frame in data if start_frame <= frame['frame'] <= end_frame]

        word_data = {
            'signer': signer,
            'video': video,
            'gloss': gloss_value,
            'start': start,
            'end': end,
            'frames': filtered_frames
        }

        label = df.loc[df['Word'] == gloss_value]['Encoded_Label'].values[0]
        output_name = f"{label}_{basename.split('.')[0]}_{start}_{end}.json"
        output_path = os.path.join(f'/home/gerardo/WORD_POSE_DATASET/{label}', output_name)
        with open(output_path, 'w') as output_file:
            json.dump(word_data, output_file)
        print(f"Datos guardados en {output_path}")


base_path = '/home/gerardo/LSE_DATABASE/LSE_HEALTH'
for json_file in os.listdir(base_path):
    full_path = os.path.join(base_path, json_file)
    extract_words_from_json(full_path)
