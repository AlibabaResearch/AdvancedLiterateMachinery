import os 
import bezier
import subprocess
import numpy as np

class2index_cord = {
    'menu.cnt': 1104,
    'menu.discountprice': 1105,
    'menu.etc': 1106,
    'menu.itemsubtotal': 1107,
    'menu.nm': 1108,
    'menu.num': 1109,
    'menu.price': 1110,
    'menu.sub.cnt': 1111,
    'menu.sub.nm': 1112,
    'menu.sub.price': 1113,
    'menu.sub.unitprice': 1114,
    'menu.unitprice': 1115,
    'menu.vatyn': 1116,
    'sub_total.discount_price': 1117,
    'sub_total.etc': 1118,
    'sub_total.othersvc_price': 1119,
    'sub_total.service_price': 1120,
    'sub_total.subtotal_price': 1121,
    'sub_total.tax_price': 1122,
    'total.cashprice': 1123,
    'total.changeprice': 1124,
    'total.creditcardprice': 1125,
    'total.emoneyprice': 1126,
    'total.menuqty_cnt': 1127,
    'total.menutype_cnt': 1128,
    'total.total_etc': 1129,
    'total.total_price': 1130,
    'void_menu.nm': 1131,
    'void_menu.price': 1132,
}

class2index_sroie = {
            'company': 1104,
            'address': 1105,
            'date': 1106,
            'total': 1107
        }

def recog_indices_to_str(recog_indices, chars):
    recog_str = []
    for idx in recog_indices:
        if idx < len(chars):
            recog_str.append(chars[idx])
        else:
            break 
    return ''.join(recog_str)

def sample_bezier_curve(bezier_pts, num_points=10, mid_point=False):
    curve = bezier.Curve.from_nodes(bezier_pts.transpose())
    if mid_point:
        x_vals = np.array([0.5])
    else:
        x_vals = np.linspace(0, 1, num_points)
    points = curve.evaluate_multi(x_vals).transpose()
    return points 

def bezier2bbox(bezier_pts):
    bezier_pts = bezier_pts.reshape(8, 2)
    points1 = sample_bezier_curve(bezier_pts[:4], 20)
    points2 = sample_bezier_curve(bezier_pts[4:], 20)
    points = np.concatenate((points1, points2))
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    return [xmin, ymin, xmax, ymax]

def bezier2polygon(bezier_pts):
    bezier_pts = bezier_pts.reshape(8, 2)
    points1 = sample_bezier_curve(bezier_pts[:4], 8)
    points2 = sample_bezier_curve(bezier_pts[4:], 8)
    points = np.concatenate((points1, points2))
    return points

def bezier_fit_quad(x, y):
    '''
        x: 文字行上or下边界控制点x坐标 (num_points, ) ndarray
        y: 文字行上or下边界控制点y坐标 (num_points, ) ndarray
    '''
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()
    t3 = t ** 3
    t2 = t ** 2
    c = np.ones_like(t)
    T = np.stack((t3, t2, t, c), axis=1)

    data = np.column_stack((x, y))

    M = np.array(
        [
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ]
    )

    M_inv = np.linalg.pinv(M)
    control_points = np.matmul(np.matmul(np.matmul(M_inv, np.linalg.pinv(np.matmul(T.T, T))), T.T), data)
    return control_points


def insert_mid_points(x_data, y_data):
    assert len(x_data) == 2 and len(y_data) == 2
    ts = [0.33, 0.66]
    x_data_mid = [(x_data[0] * (1 - t) + x_data[1] * t) for t in ts]
    x_data = [x_data[0]] + x_data_mid + [x_data[1]]
    y_data_mid = [(y_data[0] * (1 - t) + y_data[1] * t) for t in ts]
    y_data = [y_data[0]] + y_data_mid + [y_data[1]]
    return np.array(x_data), np.array(y_data)


def gen_bezier_ctrl_points(vertices):
    num_points = len(vertices)
    assert(num_points%2==0)
    
    curve_data_top = vertices[:num_points//2]
    curve_data_bottom = vertices[num_points//2:]

    x_data = curve_data_top[:, 0]
    y_data = curve_data_top[:, 1]
    if len(x_data) == 2 and len(y_data) == 2:
        x_data, y_data = insert_mid_points(x_data, y_data)

    control_points = bezier_fit_quad(x_data, y_data).astype(np.int32).flatten().tolist()

    x_data_b = curve_data_bottom[:, 0]
    y_data_b = curve_data_bottom[:, 1]
    if len(x_data_b) == 2 and len(y_data_b) == 2:
        x_data_b, y_data_b = insert_mid_points(x_data_b, y_data_b)

    control_points_b = bezier_fit_quad(x_data_b, y_data_b).astype(np.int32).flatten().tolist()

    return control_points + control_points_b


def decode_seq(seq, args, decode_type='pt', probs=None):  
    if decode_type == 'pt':
        seq = seq.reshape(-1, 2)
        decode_result = []
        for text_ins_seq in seq:
            point_x = text_ins_seq[0] / args.num_bins
            point_y = text_ins_seq[1] / args.num_bins
            decode_result.append({'point': (point_x.item(), point_y.item())})
    elif decode_type == 'poly':
        seq = seq.reshape(-1, 32)
        decode_result = []
        for text_ins_seq in seq:
            polygon = text_ins_seq / args.num_bins
            decode_result.append({'polygon': polygon})
    elif decode_type == 'rec':
        seq = seq.reshape(-1, args.rec_length)
        probs = probs.reshape(-1, args.rec_length)

        total_probs = []
        decode_result = []
        for i in range(len(seq)):
            recog = []
            tmp_probs = []
            for j in range(len(seq[i])):
                if seq[i][j] == args.recog_pad_index:
                    break 
                if seq[i][j] == args.rec_eos_index:
                    break
                if seq[i][j] == args.recog_pad_index - 1:
                    continue
                
                recog.append(args.chars[seq[i][j] - args.num_bins])
                tmp_probs.append(probs[i][j].item())

            total_probs.append(sum(tmp_probs) / (len(tmp_probs) + 1e-5))

            recog = ''.join(recog)
            decode_result.append({'rec': recog})
            
    if decode_type == 'rec':
        return decode_result, total_probs
    else:
        return decode_result


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message