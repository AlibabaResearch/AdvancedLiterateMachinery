import numpy as np
import cv2
from PIL import Image

def bezier_fit_cubic(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    if dt.sum() == 0:
        return None
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))

    M = np.array(
        [
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    t2 = t**2
    t3 = t**3
    c = np.ones_like(t)
    T = np.stack((t3, t2, t, c), axis=1)
    M_inverse = np.linalg.inv(M)
    control_points = np.matmul(M_inverse, np.linalg.inv(np.matmul(T.T, T)))
    control_points = np.matmul(control_points, T.T)
    control_points = np.matmul(control_points, data)
    return control_points

def bezier_fit_quad(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()
    t4 = t ** 4
    t3 = t ** 3
    t2 = t ** 2
    c = np.ones_like(t)
    T = np.stack((t4, t3, t2, t, c), axis=1)

    data = np.column_stack((x, y))

    M = np.array(
        [
            [1, -4, 6, -4, 1],
            [-4, 12, -12, 4, 0],
            [6, -12, 6, 0, 0],
            [-4, 4, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]
    )
    M_inv = np.linalg.pinv(M)
    control_points = np.matmul(np.matmul(np.matmul(M_inv, np.linalg.pinv(np.matmul(T.T, T))), T.T), data)
    return control_points

def insert_mid_points(x_data, y_data):
    if len(x_data) == 2 and len(y_data) == 2:
        ts = [0.33, 0.66]
        x_data_mid = [(x_data[0] * (1 - t) + x_data[1] * t) for t in ts]
        x_data = [x_data[0]] + x_data_mid + [x_data[1]]
        y_data_mid = [(y_data[0] * (1 - t) + y_data[1] * t) for t in ts]
        y_data = [y_data[0]] + y_data_mid + [y_data[1]]
    elif len(x_data) == 3 and len(y_data) == 3:
        x_data_mid_1 = [x_data[0] * 0.5 + x_data[1] * 0.5]
        x_data_mid_2 = [x_data[1] * 0.5 + x_data[2] * 0.5]
        x_data = [x_data[0]] + x_data_mid_1 + x_data_mid_2 + [x_data[2]]
        y_data_mid_1 = [y_data[0] * 0.5 + y_data[1] * 0.5]
        y_data_mid_2 = [y_data[1] * 0.5 + y_data[2] * 0.5]
        y_data = [y_data[0]] + y_data_mid_1 + y_data_mid_2 + [y_data[2]]
    return np.array(x_data), np.array(y_data)

def cpts_to_bezier_cpts_quad(word_cpts):
    top_bound_pts = word_cpts[: len(word_cpts) // 2]
    bot_bound_pts = word_cpts[len(word_cpts) // 2:]
    top_xs, top_ys = top_bound_pts[:, 0], top_bound_pts[:, 1]
    bot_xs, bot_ys = bot_bound_pts[:, 0], bot_bound_pts[:, 1]
    if len(top_bound_pts) < 4:
        top_xs, top_ys = insert_mid_points(top_xs, top_ys)
    top_bezier_pts = bezier_fit_quad(top_xs, top_ys)
    if len(bot_bound_pts) < 4:
        bot_xs, bot_ys = insert_mid_points(bot_xs, bot_ys)
    bot_bezier_pts = bezier_fit_quad(bot_xs, bot_ys)
    bezier_pts = np.concatenate((top_bezier_pts, bot_bezier_pts), axis=0).reshape(-1)
    return bezier_pts

def cpts_to_bezier_cpts_cubic(word_cpts):
    top_bound_pts = word_cpts[: len(word_cpts) // 2]
    bot_bound_pts = word_cpts[len(word_cpts) // 2:]
    top_xs, top_ys = top_bound_pts[:, 0], top_bound_pts[:, 1]
    bot_xs, bot_ys = bot_bound_pts[:, 0], bot_bound_pts[:, 1]
    if len(top_bound_pts) < 4:
        top_xs, top_ys = insert_mid_points(top_xs, top_ys)
    top_bezier_pts = bezier_fit_cubic(top_xs, top_ys)
    if len(bot_bound_pts) < 4:
        bot_xs, bot_ys = insert_mid_points(bot_xs, bot_ys)
    bot_bezier_pts = bezier_fit_cubic(bot_xs, bot_ys)
    if top_bezier_pts is None or bot_bezier_pts is None:
        return None
    bezier_pts = np.concatenate((top_bezier_pts, bot_bezier_pts), axis=0).reshape(-1)
    return bezier_pts

def cpts_to_bezier_cpts_cubic_edge(top_bound_pts):
    top_xs, top_ys = top_bound_pts[:, 0], top_bound_pts[:, 1]
    if len(top_bound_pts) < 4:
        top_xs, top_ys = insert_mid_points(top_xs, top_ys)
    top_bezier_pts = bezier_fit_cubic(top_xs, top_ys)
    return top_bezier_pts

def generate_bezier_cubic(control_points, t):
    P = np.array(control_points).reshape(-1, 2)
    M = np.array(
        [
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    T = np.array([[t**3, t**2, t, 1]])
    B = np.matmul(np.matmul(T, M), P)
    return B[0,0], B[0,1]

def draw_bezier(img, bz_points, color):
    #print(bz_points)
    img = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)  
    ts = np.linspace(0,1,20).tolist()
    up_pts = [generate_bezier_cubic(bz_points[:8], t) for t in ts]
    down_pts = [generate_bezier_cubic(bz_points[8:], t) for t in ts]
    #print(up_pts, down_pts)
    bound_pts = up_pts + down_pts
    bound_pts = [(int(pt[0]), int(pt[1])) for pt in bound_pts]
    cv2.polylines(img, [np.array(bound_pts).reshape((-1,1,2))], True, color)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    return image
