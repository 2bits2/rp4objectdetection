
#ifndef DETECT_H
#define DETECT_H

#include <opencv2/core/core.hpp>
#include <net.h>


struct Detection {
  cv::Rect_<float> box;
  float score;
  int category;
  const char *name;
};

struct InferenceResult {
  float preprocess;
  float inference;
  float postprocess;
  std::vector<Detection> detections;
};



















#endif
