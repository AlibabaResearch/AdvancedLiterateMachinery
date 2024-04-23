import cv2 
import torch
import numpy as np

from .misc import recog_indices_to_str, sample_bezier_curve, decode_seq

def visualize_decoded_result(image, results, polygons):
    if isinstance(image, torch.Tensor):
        image = image_tensor_to_opencv(image)
    
    for result in results[:]:
        point_x, point_y = result['pts'][0]
        polygon = torch.tensor(result['polys'])
        rec = result['rec']
        visualize_single_point_polygon_text(image, point_x, point_y, polygon, rec)
    
    image_h, image_w = image.shape[:2]

    if polygons != None:
        polygons = polygons * torch.tensor([image_w, image_h] * 16)
        image = visualize_polygons(image, polygons)

    return image

def visualize_seq(image, seq, seq_type, args):
    if isinstance(image, torch.Tensor):
        image = image_tensor_to_opencv(image)
    
    image_h, image_w = image.shape[:2]
    decode_result = decode_seq(seq, seq_type, args)
    for text_ins in decode_result:
        point_x, point_y = text_ins['point']
        point_x = int(point_x * image_w)
        point_y = int(point_y * image_h)
        visualize_single_point_text(image, point_x, point_y, text_ins['recog'])
    return image


def visualize_single_point_polygon_text(image, point_x, point_y, polygon, text):
    point_x = int(point_x)
    point_y = int(point_y)
    cv2.circle(image, (point_x, point_y), 5, (0, 0, 255), -1)

    polygon = polygon.numpy().astype(np.int32).reshape(-1,2)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(image, text, (point_x, point_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return image
    

def visualize_single_point_text(image, point_x, point_y, text):
    point_x = int(point_x)
    point_y = int(point_y)
    cv2.circle(image, (point_x, point_y), 6, (0, 255, 255), -1)
    return image

def image_tensor_to_opencv(image, image_size=None):

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # 将mean和std转化为与image相同的维度
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    # 反归一化
    image = image * std + mean

    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255 
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not image_size is None:
        image_h, image_w = image_size 
        image = image[:image_h, :image_w]

    return image 

def visualize_annotations(image, target, chars):
    if isinstance(image, torch.Tensor):
        image = image_tensor_to_opencv(image, target['size'])
    
    image_h, image_w = image.shape[:2]
    bboxes = target['bboxes']
    bboxes = bboxes * torch.tensor([image_w, image_h] * 2)
    bboxes = bboxes.type(torch.int32)
    image_bboxes = visualize_bboxes(image.copy(), bboxes)

    bezier_pts = target['bezier_pts']
    bezier_pts = bezier_pts * torch.tensor([image_w, image_h] * 8)
    image_bezier_curves = visualize_bezier_curves(image.copy(), bezier_pts)

    polygons = target['polygons']
    polygons = polygons * torch.tensor([image_w, image_h] * 16)
    image_polygons = visualize_polygons(image.copy(), polygons)

    center_pts = target['center_pts']/1000
    center_pts = center_pts * torch.tensor([image_w, image_h])
    center_pts = center_pts.type(torch.int32)

    recog_strs = []
    for recog_indices in target['recog']:
        recog_strs.append(recog_indices_to_str(recog_indices, chars))
    image_texts = visualize_texts(image.copy(), recog_strs, center_pts.type(torch.int32))
    
    return image_bboxes, image_bezier_curves, image_polygons, image_texts

def visualize_bboxes(image, bboxes):
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.numpy()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
    return image

def visualize_bezier_curves(image, bezier_pts):
    for bezier_pt in bezier_pts:
        bezier_pt = bezier_pt.numpy().reshape(8, 2)
        curve1 = sample_bezier_curve(bezier_pt[:4], 10)
        curve2 = sample_bezier_curve(bezier_pt[4:], 10)
        polygon = np.concatenate((curve1, curve2))
        polygon = polygon.astype(np.int32)
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        curve1 = sample_bezier_curve(bezier_pt[:4], 100)[30:70,:]
        curve2 = sample_bezier_curve(bezier_pt[4:], 100)[30:70,:][::-1, :]

        dist = curve2 - curve1
        curve1 = curve1 + 0.3*dist 
        curve2 = curve1 + 0.4*dist

        polygon = np.concatenate((curve1, curve2[::-1,:]))
        polygon = polygon.astype(np.int32)
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    return image

def visualize_polygons(image, polygons):
    for polygon in polygons:
        polygon = polygon.numpy().astype(np.int32).reshape(-1,2)
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    return image

def visualize_texts(image, texts, points):
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    for text, point in zip(texts, points):
        cv2.putText(image, text, (point[0], point[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(image, (point[0], point[1]), 6, (0, 255, 255), -1)
    return image