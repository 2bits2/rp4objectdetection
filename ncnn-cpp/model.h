#ifndef YOLOV8_H
#define YOLOV8_H

#include <opencv2/core/core.hpp>
#include <net.h>


struct Object
{
  cv::Rect_<float> rect;
  const char *name;
  float prob;
  int cate;
  float area() {return rect.width * rect.height;}
};

// all the data
// i want to measure
struct InferenceResult {
  double preprocess;
  double inference;
  double postprocess;
  std::vector<Object> objects;
};

struct Model {
  virtual int detect(const cv::Mat& rgb, InferenceResult &result) = 0;
  virtual std::string getName() = 0;
};


struct FileInferenceResult {
  int file_id;
  std::string filename;
  InferenceResult result;
};

void saveImageDirectoryInferencesAsJson(
                                        Model &model,
                                        const char *dirname,
                                        std::string outputfilename
                                        );

struct YoloV8 : public Model {

  YoloV8(std::string name);
  std::string getName();
  int detect(const cv::Mat& rgb, InferenceResult &result);

  ncnn::Net yolo;
  int target_size;
  float mean_vals[3];
  float norm_vals[3];
  std::string name;
  ~YoloV8(){}
};


struct NanoDet : public Model {

  NanoDet(std::string name);
  std::string getName();
  int detect(const cv::Mat& rgb, InferenceResult &result);

  ncnn::Net nanodet;
  int target_size;
  float prob_threshold;
  float nms_threshold;
  float mean_vals[3];
  float norm_vals[3];
  std::string name;
  ~NanoDet(){}
};


struct YoloV2 : public Model {
  YoloV2(std::string name);
  std::string getName();

  ncnn::Net net;
  std::vector<float> anchor;
  int numAnchor;
  int numOutput;
  int numThreads;
  int numCategory;
  int inputWidth, inputHeight;
  float nmsThresh;
  float prob_threshold;
  std::string name;


  int detect(const cv::Mat& rgb, InferenceResult &result);
  int nmsHandle(std::vector<Object> &tmpBoxes, std::vector<Object> &dstBoxes);
  int getCategory(const float *values, int index, int &category, float &score);
  int predHandle(const ncnn::Mat *out, std::vector<Object> &dstBoxes,
                 const float scaleW, const float scaleH, const float thresh);

  ~YoloV2(){}
};










#endif // YOLOV8_H
