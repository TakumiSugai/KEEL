import numpy as np

def extract_vector_1by2(vector):
    return vector[:2]

def extract_vector_1by3(vector):
    return vector[:3]

def extend_vector_1by4(vector):
    if 2 == len(vector):
        return np.append(vector, [0, 1])
    elif 3 == len(vector):
        return np.append(vector, 1)
    return vector

def exterior_angle(vector_from, vector_to):
    if not isinstance(vector_from, np.ndarray) \
        or not isinstance(vector_to, np.ndarray):
        return None
    
    vector_from_xy = extract_vector_1by2(vector_from)
    vector_to_xy = extract_vector_1by2(vector_to)
    
    outer_product = np.cross(vector_from_xy, vector_to_xy)

    if 0 < outer_product:
        return unsigined_exterior_angle(vector_from_xy, vector_to_xy)
    elif outer_product < 0:
        return - unsigined_exterior_angle(vector_from_xy, vector_to_xy)
    else: #outer_product = 0
        inner_product = np.dot(vector_from_xy, vector_to_xy)
        if inner_product < 0:
            return np.pi
        else: #0 < inner_product
            return 0

def unsigined_exterior_angle(vector_from, vector_to):
    vector_from_xy = extract_vector_1by2(vector_from)
    vector_to_xy = extract_vector_1by2(vector_to)

    inner_product = np.dot(vector_from_xy, vector_to_xy)
    len_vector_from = np.linalg.norm(vector_from_xy, ord=2)
    len_vector_to = np.linalg.norm(vector_to_xy, ord=2)
    return np.arccos(inner_product/(len_vector_from * len_vector_to))

