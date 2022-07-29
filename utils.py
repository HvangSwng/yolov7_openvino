
import numpy as np
import cv2


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dh = img.shape[0]
    dw = img.shape[1]
    theMax = max(dh, dw)
    addh = int((theMax - dh) // 2)
    addw = int((theMax - dw) // 2)
    img = cv2.copyMakeBorder(img, addh, addh, addw, addw, cv2.BORDER_CONSTANT, value=color)

    return img

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y

def nms(prediction, conf_thres, iou_thres):
    predictions = np.squeeze(prediction[0])
    # Filter out object confidence scores below threshold
    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf_thres]
    obj_conf = obj_conf[obj_conf > conf_thres]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, np.newaxis]

    # Get the scores
    scores = np.max(predictions[:, 5:], axis=1)

    # Filter out the objects with a low score
    valid_scores = scores > conf_thres
    predictions = predictions[valid_scores]
    scores = scores[valid_scores]
    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # Get bounding boxes for each object
    boxes = xywh2xyxy(predictions[:, :4])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres).flatten()

    return boxes[indices], scores[indices], class_ids[indices]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
