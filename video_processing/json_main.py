from JSON.config import VIDEO_BASE_PATH
from JSON.video_processor import extract_signer_pose
from JSON.data_loader import load_signer_video
import os

def process_all_videos():
    signer_video_df = load_signer_video()

    for _, row in signer_video_df.iterrows():
        signer = row['Signer']
        videos = row['ELAN file']
        for video in videos:
            video_path = os.path.join(VIDEO_BASE_PATH, f"{video}.mp4")
            extract_signer_pose(video_path, signer_id=signer)

def main():
        process_all_videos()

if __name__ == "__main__":
    main()
