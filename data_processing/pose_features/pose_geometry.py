from itertools import combinations
from data_processing.pose_features.utils import euclidean_distance

def pose_distances(dict_coords):
    distances = {}

    distances['Nose-Left_Shoulder'] = euclidean_distance(dict_coords['pose.NOSE'], dict_coords['pose.LEFT_SHOULDER'])
    distances['Nose-Right_Shoulder'] = euclidean_distance(dict_coords['pose.NOSE'], dict_coords['pose.RIGHT_SHOULDER'])

    distances['Nose-Left_WRIST'] = euclidean_distance(dict_coords['pose.NOSE'], dict_coords['Hand_0.WRIST'])
    distances['Nose-Right_WRIST'] = euclidean_distance(dict_coords['pose.NOSE'], dict_coords['Hand_1.WRIST'])

    distances['Left_Shoulder-Left_WRIST'] = euclidean_distance(dict_coords['pose.LEFT_SHOULDER'], dict_coords['Hand_0.WRIST'])
    distances['Right_Shoulder-Right_WRIST'] = euclidean_distance(dict_coords['pose.RIGHT_SHOULDER'], dict_coords['Hand_1.WRIST'])
    distances['Left_WRIST-Right_WRIST'] = euclidean_distance(dict_coords['Hand_0.WRIST'], dict_coords['Hand_1.WRIST'])

    fingers = ['THUMB', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']
    for f in fingers:
        distances[f'Hand_0.{f}_Tip-Mcp'] = euclidean_distance(dict_coords[f'Hand_0.{f}_TIP'], dict_coords[f'Hand_0.{f}_MCP'])
        distances[f'Hand_1.{f}_Tip-Mcp'] = euclidean_distance(dict_coords[f'Hand_1.{f}_TIP'], dict_coords[f'Hand_1.{f}_MCP'])

    for f1, f2 in combinations(fingers, 2):
        distances[f'Hand_0.{f1}_Tip-Hand_0.{f2}_Tip'] = euclidean_distance(dict_coords[f'Hand_0.{f1}_TIP'], dict_coords[f'Hand_0.{f2}_TIP'])
        distances[f'Hand_1.{f1}_Tip-Hand_1.{f2}_Tip'] = euclidean_distance(dict_coords[f'Hand_1.{f1}_TIP'], dict_coords[f'Hand_1.{f2}_TIP'])

    return distances
