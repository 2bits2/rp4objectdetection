# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch

from loguru import logger
from PIL import Image

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis, postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList

import json
import time


def myprocessing(boxes, scores, cls_ids, conf=0.01, class_names=None):
    res_boxes = []
    res_scores = []
    res_labels = []

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        res_labels.append(class_names[cls_id])
        res_boxes.append([x0, y0, x1 - x0, y1 - y0])
        res_scores.append(score.item())

    return (res_boxes, res_scores, res_labels)

class Infer():
    def __init__(self, config, infer_size=[640,640], device='cpu', output_dir='./', ckpt=None, end2end=False):
        self.speed = {}
        self.ckpt_path = ckpt
        suffix = ckpt.split('.')[-1]
        if suffix != 'onnx':
            NotImplementedError(f'expected onnx model')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if torch.cuda.is_available() and device=='cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        self.model = self._build_engine(self.config)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h<=target_size[0] and w<=target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]
        return ImageList(pad_imgs, img_sizes, pad_sizes)

    def _build_engine(self, config):
        print(f'Inference with onnx engine!')
        model, self.input_name, self.infer_size, _, _ = self.build_onnx_engine(self.ckpt_path)
        return model

    def build_onnx_engine(self, onnx_path):
        import onnxruntime
        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        out_names = []
        out_shapes = []
        for idx in range(len(session.get_outputs())):
            out_names.append(session.get_outputs()[idx].name)
            out_shapes.append(session.get_outputs()[idx].shape)
        return session, input_name, input_shape[2:], out_names, out_shapes


    def preprocess(self, origin_img):
        #img = np.asarray(origin_img.convert('RGB'))
        img = transform_img(origin_img, 0,
                            **self.config.test.augment.transform,
                            infer_size=self.infer_size)
        # img is a image_list
        oh, ow, _  = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)
        img = img.to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, image, origin_shape=None, conf=0.01):
        scores = torch.Tensor(preds[0])
        bboxes = torch.Tensor(preds[1])
        output = postprocess(scores, bboxes,
                             self.config.model.head.num_classes,
                             self.config.model.head.nms_conf_thre,
                             self.config.model.head.nms_iou_thre,
                             image)
        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        bboxes, scores, labels = myprocessing(bboxes,
                                              scores,
                                              cls_inds,
                                              conf,
                                              self.class_names)
        return bboxes,  scores, labels

    def forward(self, origin_image):

        begin = time.time()
        image, origin_shape = self.preprocess(origin_image)
        end = time.time()
        elapsed = end - begin
        self.speed['preprocess'] = elapsed * 1000

        begin = time.time()
        image_np = np.asarray(image.tensors.cpu())
        output = self.model.run(None, {self.input_name: image_np})
        end = time.time()
        elapsed = end - begin
        self.speed['inference'] = elapsed * 1000

        begin = time.time()
        bboxes, scores, labels = self.postprocess(output, image, origin_shape=origin_shape)
        end = time.time()
        elapsed = end - begin
        self.speed['postprocess'] = elapsed * 1000

        return bboxes, scores, labels


    # def visualize(self, image, bboxes, scores, cls_inds, conf, save_name='vis.jpg', save_result=True):
    #     if save_result:
    #         save_path = os.path.join(self.output_dir, save_name)
    #         #print(f"save visualization results at {save_path}")
    #         cv2.imwrite(save_path, vis_img[:, :, ::-1])
    #     return vis_img




def measureDamoYoloModel(infer_engine, imageDir):

    # all the data we want
    # to capture for
    # mean average precision
    # boxes = []
    # scores = []
    # labels = []

    # # and speed measurement
    # speeds_preprocess = []
    # speeds_inference = []
    # speeds_postprocess = []
    # speeds_total_seperate = []

    # filenames = []

    result = {}

    directory = os.fsencode(imageDir)
    for f in os.listdir(directory):
        imageName = os.fsdecode(f)
        if imageName.endswith(".jpg"):

            # load the actual image
            # for inference
            fullImageName = imageDir + imageName
            image = cv2.imread(fullImageName)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # extract info
            bboxes, res_scores, labels = infer_engine.forward(image)
            pred_result = {}
            pred_result['preprocess'] = infer_engine.speed['preprocess']
            pred_result['inference'] = infer_engine.speed['inference']
            pred_result['postprocess'] = infer_engine.speed['postprocess']
            pred_result['xywhsl'] = [[box[0],box[1],box[2],box[3],score, label] for box, score, label in zip(bboxes, res_scores, labels)]

            # save the result
            result[fullImageName] = pred_result


            #vis_res = infer_engine.visualize(origin_image, bboxes, scores, cls_inds, conf=0.8, save_name=f"demo/{img['file_name']}", save_result=True)

    # # now present everything
    # # in a nice overview
    # result = {
    #     'preprocess': speeds_preprocess,
    #     'inference': speeds_inference,
    #     'postprocess': speeds_postprocess,
    #     'boxes': boxes,
    #     'scores': scores,
    #     'labels': labels,
    #     'files': filenames
    # }

    return result

def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')
    parser.add_argument('-f',
                        '--config_file',
                        type=str,
                        help='pls input your config file',)
    parser.add_argument('--engine',
                        default=None,
                        type=str,
                        help='engine for inference')
    parser.add_argument('--infer_size',
                        nargs='+',
                        type=int,
                        help='test img size')
    parser.add_argument('-i',
                        '--image_dir',
                        type=str,
                        default='/home/pi/rp4objectdetection/datasets/coco-2017/validation/data/',
                        help="image directory for inference"
                        )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    infer_engine = Infer(config,
                         infer_size=args.infer_size,
                         ckpt=args.engine,
                         )
    measurements = measureDamoYoloModel(infer_engine, args.image_dir)
    with open('measurement_damoyolo.json', 'w') as fp:
        json.dump(measurements, fp)

if __name__ == '__main__':
    main()
