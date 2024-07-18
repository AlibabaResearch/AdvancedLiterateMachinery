def box_contains(box1, box2, threshold=10):
    """Check if box1 contains box2 within a given threshold."""
    return box1[0] <= box2[0] + threshold and box1[1] <= box2[1] + threshold and \
           box1[2] + threshold >= box2[2] and box1[3] + threshold >= box2[3]

def find_closest_box(target_box, candidate_boxes):
    """Find the closest box to a target box based on Euclidean distance."""
    def get_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def euclidean_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    target_center = get_center(target_box)
    min_distance = float('inf')
    closest_index = -1
    for i, box in enumerate(candidate_boxes):
        candidate_center = get_center(box)
        distance = euclidean_distance(target_center, candidate_center)
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    return closest_index
