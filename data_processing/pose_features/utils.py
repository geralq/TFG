import numpy as np

def euclidean_distance(p1, p2):
    if p1[0] == -1 and p1[1] == -1:
        return -1
    if p2[0] == -1 and p2[1] == -1:
        return -1
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def angle_between_joints(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))