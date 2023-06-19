
import os
import os.path
from ultralytics import YOLO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
import ultralytics.yolo.utils.ops as yoloOp
from pprint import pprint
import time
import json

def measureYoloModel(modelname, imageDir):
    """
    get dictionary with inference results
    for all images in the imageDir
    """

    # load the object detection model
    model = YOLO(modelname)

    # both models use the same
    # categories but
    # somehow the model classes and the coco classes
    # are at different positions
    # so there should be a translation via the model names
    modelCategoryNames = model.model.names

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # do the inference
            # but verbose=False
            # because I don't want to measure
            # the logging time
            begin = time.time()
            results = model.predict(
                source=image,
                verbose=False,
                conf=0.01
            )
            end = time.time()
            elapsed = end - begin

            pred_result = {}

            pred_result['preprocess'] = results[0].speed['preprocess']
            pred_result['inference'] = results[0].speed['inference']
            pred_result['postprocess'] = results[0].speed['postprocess']
            pred_result['total_seperate'] = elapsed

            boxes = yoloOp.xywh2ltwh(results[0].boxes.xywh).tolist()
            confs = results[0].boxes.conf.tolist()
            labels = list(
                map(
                    lambda x: modelCategoryNames[x.item()],
                    results[0].boxes.cls
                )
            )
            pred_result['xywhsl'] = [[box[0],box[1],box[2],box[3],conf,lab] for box, conf, lab in zip(boxes, confs, labels)]

            result[fileName] = pred_result

    return result

if __name__ == '__main__':
    # I want to validate the object detection models
    # with the coco images
    imageDir = '/home/pi/project/datasets/coco-2017/validation/data/'
    modelname = "yolov8n.pt"
    measurements = measureYoloModel(modelname, imageDir)

    with open(f'measurement_{modelname}.json', 'w') as fp:
        json.dump(measurements, fp)

