#include "model.h"
#include <iostream>
#include <stdio.h>
#include <vector>


int main(int argc, char** argv)
{
  // specify the directory that
  // contains all the images
  const char* dirname = "/home/pi/rp4objectdetection/datasets/coco-2017/validation/data/";

  // these are the models we
  // want to test
  YoloV8 yolov8("yolov8n");
  NanoDet nanoDet("nanodet");
  YoloV2 yoloFastestV2("yoloFastestV2");

  // for all the images we do object detections
  // and save them as a json file
  saveImageDirectoryInferencesAsJson(yoloFastestV2, dirname,
                                     "../eval/measurements/measurement_ncnn_cpp_yoloFastestV2.json");
  saveImageDirectoryInferencesAsJson(yolov8, dirname,
                                     "../eval/measurements/measurement_ncnn_cpp_yolov8n.json");
  saveImageDirectoryInferencesAsJson(nanoDet, dirname,
                                     "../eval/measurements/measurement_ncnn_cpp_nanodet.json");
  // done
  return 0;
}
