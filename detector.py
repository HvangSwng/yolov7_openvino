from openvino.runtime import Core
import numpy as np
from utils import letterbox,sigmoid,nms

class Yolov7(object):
    def __init__(self, model_path):

        ie = Core()
        self.model = ie.read_model(model_path)
        self.compiled_model = ie.compile_model(model=self.model, device_name="CPU")
        self.img_size = (640, 640)
        self.conf_thres = 0.3
        self.iou_thres = 0.4
        self.class_num = 80
        self.stride = [8, 16, 32]
        self.anchor_list = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]

    def input_preprocess(self,ori_img):
        img = letterbox(ori_img, self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def output_postprocess(self,pred,img_shape,src_size):
        # get the each feature map's output data
        anchor = np.array(self.anchor_list).astype(np.float).reshape(3, -1, 2)
        area = self.img_size[0] * self.img_size[1]
        size = [int(area / self.stride[0] ** 2), int(area / self.stride[1] ** 2), int(area / self.stride[2] ** 2)]
        feature = [[int(j / self.stride[i]) for j in self.img_size] for i in range(3)]

        output = []
        output.append(sigmoid(pred[self.compiled_model.output(0)].reshape(-1, size[0] * 3, 5 + self.class_num)))
        output.append(sigmoid(pred[self.compiled_model.output(1)].reshape(-1, size[1] * 3, 5 + self.class_num)))
        output.append(sigmoid(pred[self.compiled_model.output(2)].reshape(-1, size[2] * 3, 5 + self.class_num)))

        # Postprocessing
        grid = []
        for _, f in enumerate(feature):
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

        result = []
        for i in range(3):
            src = output[i]
            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(3):
                dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + grid[i]) * self.stride[i])
                dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
            src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
            src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
            result.append(src)

        results = np.concatenate(result, 1)
        boxes, scores, class_ids = nms(results, self.conf_thres, self.iou_thres)

        # scale bounding boxes
        gain = min(img_shape[0] / src_size[0],img_shape[1] / src_size[1])
        padding = (img_shape[1] - src_size[1] * gain) / 2, (img_shape[0] - src_size[0] * gain) / 2

        boxes[:, [0, 2]] -= padding[0]  # x padding
        boxes[:, [1, 3]] -= padding[1]  # y padding
        boxes[:, :4] /= gain

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])
        boxes[:, 1].clip(0, img_shape[0])
        boxes[:, 2].clip(0, img_shape[1])
        boxes[:, 3].clip(0, img_shape[0])

        return boxes, scores, class_ids



    def inference(self, ori_img):
        src_size = ori_img.shape[:2]
        # Preprocessing
        input_img=self.input_preprocess(ori_img)
        img_shape = input_img.shape[2:]

        # inference
        pred = self.compiled_model([input_img])

        # output_postprocess
        boxes, scores, class_ids=self.output_postprocess(pred, img_shape, src_size)

        # load the results
        results = []
        for xyxy, conf, cls in zip(boxes, scores, class_ids):
            if cls==0:
                results.append(xyxy)

        return results
