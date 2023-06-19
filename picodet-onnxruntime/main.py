import cv2
import numpy as np
import argparse
import onnxruntime as ort
import math
import time
import os
import json

class PicoDet():

    def __init__(self, model_pb_path, label_path, prob_threshold=0.4,
                 iou_threshold=0.3):
        self.classes = list(
            map(
                lambda x: x.strip(),
                open(label_path, 'r').readlines()
            )
        )
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array([103.53, 116.28, 123.675],
                             dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395],
                            dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3

        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (self.net.get_inputs()[0].shape[2],
                            self.net.get_inputs()[0].shape[3])

        self.num_outs = int(len(self.net.get_outputs()) * 0.5)
        self.reg_max = int(self.net.get_outputs()[self.num_outs].shape[-1] / 4) - 1
        self.project = np.arange(self.reg_max + 1)
        self.strides = [int(8 * (2**i)) for i in range(self.num_outs)]
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]),
                 math.ceil(self.input_shape[1] / self.strides[i])), self.strides[i])
            self.mlvl_anchors.append(anchors)

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        cx = xv + 0.5 * (stride-1)
        cy = yv + 0.5 * (stride - 1)
        return np.stack((cx, cy), axis=-1)

    def softmax(self,x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    ### c++: https://blog.csdn.net/wuqingshan2010/article/details/107727909
    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img, 0, 0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0
                )  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top,
                                         self.input_shape[0] - newh - top,
                                         0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)

        return img, newh, neww, top, left


    def post_process(self, preds):
        cls_scores, bbox_preds = preds[:self.num_outs], preds[self.num_outs:]
        det_bboxes, det_conf, det_classid = self.get_bboxes_single(
            cls_scores, bbox_preds, 1, rescale=False)
        return det_bboxes.astype(np.int32), det_conf, det_classid


    def get_bboxes_single(self, cls_scores, bbox_preds, scale_factor, rescale=False):

        mlvl_bboxes = []
        mlvl_scores = []

        for stride, cls_score, bbox_pred, anchors in zip(
                self.strides, cls_scores, bbox_preds, self.mlvl_anchors):
            if cls_score.ndim==3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim==3:
                bbox_pred = bbox_pred.squeeze(axis=0)
                bbox_pred = self.softmax(
                    bbox_pred.reshape(-1, self.reg_max + 1),
                    axis=1
                )

            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1,4)
            bbox_pred *= stride

            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(anchors, bbox_pred,
                                        max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)

        if rescale:
            mlvl_bboxes /= scale_factor

        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(),
                                   confidences.tolist(),
                                   self.prob_threshold,
                                   self.iou_threshold)

        if len(indices)>0:
            indices=indices.flatten()
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            #print('nothing detect')
            return np.array([]), np.array([]), np.array([])


    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


    def detect(self, srcimg):

        result = {}

        ###### Preprocess #######
        begin = time.time()

        img, newh, neww, top, left = self.resize_image(srcimg)
        img = self._normalize(img)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        end = time.time()
        elapsed = end - begin
        result["preprocess"] = elapsed * 1000


        ###### INFERENCE  ######
        begin = time.time()

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        end = time.time()

        elapsed = end - begin
        result["inference"] = elapsed * 1000


        ###### POSTPROCESS ####
        begin = time.time()
        boxes_xywhsl = []

        det_bboxes, det_conf, det_classid = self.post_process(outs)
        ratioh, ratiow = srcimg.shape[0]/newh,srcimg.shape[1]/neww
        for i in range(det_bboxes.shape[0]):
            xmin = max(int((det_bboxes[i,0] - left) * ratiow), 0)
            ymin = max(int((det_bboxes[i,1] - top) * ratioh), 0)
            xmax = min(int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1])
            ymax = min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
            score = det_conf[i]
            label = self.classes[det_classid[i]]
            box = [xmin, ymin, xmax - xmin, ymax - ymin, float(score), label]
            boxes_xywhsl.append(box)

        end = time.time()
        elapsed = end - begin

        result["postprocess"] = elapsed * 1000
        result["xywhsl"] = boxes_xywhsl
        return result


def measurePicoDetModel(model, imageDir):
    """
    get dictionary with inference results
    for all images in the imageDir
    """
    # measure for all
    # existing images
    result = {}
    directory = os.fsencode(imageDir)
    for f in os.listdir(directory):
        imageName = os.fsdecode(f)
        if imageName.endswith(".jpg"):

            # load the actual image
            # for inference
            fileName = imageDir + imageName

            image = cv2.imread(fileName)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # do the inference
            begin = time.time()
            pred_result = model.detect(image)
            end = time.time()
            elapsed = end - begin
            pred_result["total_seperate"] = elapsed
            result[fileName] = pred_result

    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir", type=str,
        default="/home/pi/rp4objectdetection/datasets/coco-2017/validation/data/",
        help="image path"
    )
    parser.add_argument(
        "--modelpath", type=str,
        default="coco/picodet_s_320_coco.onnx",
        choices=["coco/picodet_m_320_coco.onnx",
                 "coco/picodet_m_416_coco.onnx",
                 "coco/picodet_s_320_coco.onnx",
                 "coco/picodet_s_416_coco.onnx"],
        help="onnx filepath"
    )
    parser.add_argument("--classfile", type=str, default="coco/coco.names", help="classname filepath")
    parser.add_argument("--confThreshold", default=0.01, type=float, help="class confidence")
    parser.add_argument("--nmsThreshold", default=0.6, type=float, help="nms iou thresh")

    args = parser.parse_args()
    model = PicoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.confThreshold,
        iou_threshold=args.nmsThreshold
    )
    measurements = measurePicoDetModel(model, args.imgdir)
    with open(f"measurement_{os.path.basename(os.path.normpath(args.modelpath))}.json", 'w') as fp:
        json.dump(measurements, fp)
