from data_processing.pose_features.processor import process_directory
from data_processing.data_split.split_feature_data import split_data

features = '/home/gerardo/FEATURES_POSE_DATASET'
word_pose = '/home/gerardo/WORD_POSE_DATASET'


process_directory(word_pose, features, clean_output_dir=True)
split_data(features)