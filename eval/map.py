import os
import json
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mean_average_precision import MetricBuilder

from pprint import pprint

# from os import listdir
# get the path/directory

projectDir='/home/pi/rp4objectdetection/'
dataDir=f'{projectDir}datasets/coco-2017/'
dataType='val2017'
annFile='{}/raw/instances_{}.json'.format(dataDir,dataType)
imgDir=f'{dataDir}validation/data/'


# initialize Coco
coco = COCO(annFile)
imageInfos = coco.loadImgs(coco.getImgIds())

coco_label_to_class = {}
cats = coco.loadCats(coco.getCatIds())
for cat in cats:
    coco_label_to_class[cat['name']] = cat['id']

# now load all the measurement results of
# all the models
measurements = {}
for dirpath, _, filenames in os.walk("/home/pi/project"):
    for filename in filenames:
        if filename.startswith('measurement') and filename.endswith('.json'):
            fullname = os.path.join(dirpath, filename)
            with open(fullname, "r") as f:
                measurements[filename] = json.load(f)


# get target and mpredictions
# for each image
image_detections = {}
for imgInfo in imageInfos:

    # currently not
    # every file is there
    image_path = imgDir + imgInfo['file_name']
    if os.path.isfile(image_path):

        # get the ground truth
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=imgInfo['id']))
        target_boxes = []
        target_labels = []
        for ann in annotations:
            xywh = ann['bbox']
            cls = ann['category_id']
            target_boxes.append([
                xywh[0],
                xywh[1],
                xywh[0] + xywh[2],
                xywh[1] + xywh[3],
            ])
            target_labels.append(cls)

        ground_truth = dict(
            boxes=target_boxes,
            labels=target_labels
        )

        # get the predictions
        model_predictions = {}
        for model_name, model_data in measurements.items():
            pred_boxes = []
            pred_scores = []
            pred_labels = []

            xywhsl_boxes = model_data[image_path]['xywhsl']
            for xywhsl in xywhsl_boxes:
                if not xywhsl[5] == 'background':
                    pred_boxes.append([
                        xywhsl[0],
                        xywhsl[1],
                        xywhsl[0] + xywhsl[2],
                        xywhsl[1] + xywhsl[3]
                    ])
                    pred_scores.append(xywhsl[4])
                    pred_labels.append(coco_label_to_class[xywhsl[5]])

            model_predictions[model_name] = dict(
                boxes=pred_boxes,
                scores=pred_scores,
                labels=pred_labels
            )

        image_detections[image_path] = {
            'predictions': model_predictions,
            'groundtruth': ground_truth
        }

# calculate the mapMetrics
mApResults = {}
for model_name, model_data in measurements.items():
    metric = MeanAveragePrecision()
    for image_path, detection in image_detections.items():
        metric.update(
            [dict(
                boxes=torch.tensor(image_detections[image_path]['predictions'][model_name]['boxes']),
                labels=torch.tensor(image_detections[image_path]['predictions'][model_name]['labels']),
                scores=torch.tensor(image_detections[image_path]['predictions'][model_name]['scores']),
            )],
            [
                dict(
                    boxes=torch.tensor(image_detections[image_path]['groundtruth']['boxes']),
                    labels=torch.tensor(image_detections[image_path]['groundtruth']['labels']),
                )
            ]
        )

    metric_result = metric.compute()
    print(model_name)
    print(metric_result)
    for key, val in metric_result.items():
        metric_result[key] = val.item()
    mApResults[model_name] = metric_result

with open("mapResult.json", "w") as fp:
    json.dump(mApResults, fp)

print(mApResults)







# mApResults = {}
# for model_name in measurements.keys():
#     metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=80)
#     for image_path, detection in image_detections.items():
#         gt = image_detections[image_path]['groundtruth']
#         gt = np.array(
#             [ [box[0], box[1], box[2], box[3], label, 0, 0]  for box, label in zip(gt['boxes'], gt['labels'])]
#         )
#         preds = image_detections[image_path]['predictions']
#         preds = np.array(
#             [[box[0],box[1],box[2],box[3],label,score] for box,label,score in zip(preds[model_name]['boxes'],preds[model_name]['labels'],preds[model_name]['scores'])]
#         )
#         metric_fn.add(preds, gt)
#     cocometric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
#     mApResults[model_name] = cocometric
# print(mApResults)



#         xywhsl_boxes = measurementsmodel_name][image_path]['xywhsl']
# print(targets)
