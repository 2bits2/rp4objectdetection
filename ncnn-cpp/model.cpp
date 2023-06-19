// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// modified 1-14-2023 Q-engineering
// modified May 2023 Tobi

#include "model.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <dirent.h>
#include <string>

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};


const char* yolov8_class_names[] = { "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };


const char* nanodet_class_names[] = {
  "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const char* yolov2_class_names[] = {
  "person", "bicycle",
  "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "backpack", "umbrella", "handbag", "tie", "suitcase",
  "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork",
  "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors",
  "teddy bear", "hair drier", "toothbrush"
};




static void nanodet_generate_proposals(const ncnn::Mat& cls_pred,
                                       const ncnn::Mat& dis_pred,
                                       int stride,
                                       const ncnn::Mat& in_pad,
                                       float prob_threshold,
                                       std::vector<Object>& objects)
{
    const int num_grid = cls_pred.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = cls_pred.w;
    const int reg_max_1 = dis_pred.w / 4;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const int idx = i * num_grid_x + j;

            const float* scores = cls_pred.row(idx);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                if (scores[k] > score)
                {
                    label = k;
                    score = scores[k];
                }
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*)dis_pred.row(idx));
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.cate = label;
                obj.name = nanodet_class_names[label];
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}




static float fast_exp(float x)
{
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

static float sigmoid(float x)
{
  return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object& a, const Object& b)
{
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j)
    {
      while (faceobjects[i].prob > p)
        i++;

      while (faceobjects[j].prob < p)
        j--;

      if (i <= j)
        {
          std::swap(faceobjects[i], faceobjects[j]);
          i++;
          j--;
        }
    }
  #pragma omp parallel sections
  {

    #pragma omp section
    {
      if (left < j) qsort_descent_inplace(faceobjects, left, j);
    }
    #pragma omp section
    {
      if (i < right) qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
  if (faceobjects.empty())
    return;

  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<int>& picked,
                              float nms_threshold)
{
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++)
    {
      areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

  for (int i = 0; i < n; i++)
    {
      const Object& a = faceobjects[i];

      int keep = 1;
      for (int j = 0; j < (int)picked.size(); j++)
        {
          const Object& b = faceobjects[picked[j]];

          // intersection over union
          float inter_area = intersection_area(a, b);
          float union_area = areas[i] + areas[picked[j]] - inter_area;
          // float IoU = inter_area / union_area
          if (inter_area / union_area > nms_threshold)
            keep = 0;
        }

      if (keep)
        picked.push_back(i);
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
  for (int i = 0; i < (int)strides.size(); i++)
    {
      int stride = strides[i];
      int num_grid_w = target_w / stride;
      int num_grid_h = target_h / stride;
      for (int g1 = 0; g1 < num_grid_h; g1++)
        {
          for (int g0 = 0; g0 < num_grid_w; g0++)
            {
              GridAndStride gs;
              gs.grid0 = g0;
              gs.grid1 = g1;
              gs.stride = stride;
              grid_strides.push_back(gs);
            }
        }
    }
}
static void yolov8_generate_proposals(std::vector<GridAndStride> grid_strides,
                               const ncnn::Mat& pred,
                               float prob_threshold,
                               std::vector<Object>& objects)
{
  const int num_points = grid_strides.size();
  const int num_class = 80;
  const int reg_max_1 = 16;

  for (int i = 0; i < num_points; i++)
    {
      const float* scores = pred.row(i) + 4 * reg_max_1;

      // find label with max score
      int label = -1;
      float score = -FLT_MAX;
      for (int k = 0; k < num_class; k++)
        {
          float confidence = scores[k];
          if (confidence > score)
            {
              label = k;
              score = confidence;
            }
        }
      float box_prob = sigmoid(score);
      if (box_prob >= prob_threshold)
        {
          ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
          {
            ncnn::Layer* softmax = ncnn::create_layer("Softmax");

            ncnn::ParamDict pd;
            pd.set(0, 1); // axis
            pd.set(1, 1);
            softmax->load_param(pd);

            ncnn::Option opt;
            opt.num_threads = 1;
            opt.use_packing_layout = false;

            softmax->create_pipeline(opt);

            softmax->forward_inplace(bbox_pred, opt);

            softmax->destroy_pipeline(opt);

            delete softmax;
          }

          float pred_ltrb[4];
          for (int k = 0; k < 4; k++)
            {
              float dis = 0.f;
              const float* dis_after_sm = bbox_pred.row(k);
              for (int l = 0; l < reg_max_1; l++)
                {
                  dis += l * dis_after_sm[l];
                }

              pred_ltrb[k] = dis * grid_strides[i].stride;
            }

          float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
          float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

          float x0 = pb_cx - pred_ltrb[0];
          float y0 = pb_cy - pred_ltrb[1];
          float x1 = pb_cx + pred_ltrb[2];
          float y1 = pb_cy + pred_ltrb[3];

          Object obj;
          obj.rect.x = x0;
          obj.rect.y = y0;
          obj.rect.width = x1 - x0;
          obj.rect.height = y1 - y0;
          obj.cate = label;
          obj.name = yolov8_class_names[label];
          obj.prob = box_prob;

          objects.push_back(obj);
        }
    }
}



YoloV8::YoloV8(std::string name_input)
{
  // the target size must
  // be divisible by 32.
  target_size = 640;
  name = name_input;
  yolo.clear();
  yolo.opt = ncnn::Option();
  yolo.opt.num_threads = 4;
  yolo.load_param("./data/yolov8n.param");
  yolo.load_model("./data/yolov8n.bin");

  // target_size = _target_size;
  mean_vals[0] = 103.53f;
  mean_vals[1] = 116.28f;
  mean_vals[2] = 123.675f;
  norm_vals[0] = 1.0 / 255.0f;
  norm_vals[1] = 1.0 / 255.0f;
  norm_vals[2] = 1.0 / 255.0f;
}

NanoDet::NanoDet(std::string name_input)
{
  name = name_input;
  target_size = 320; //640; //320;
  prob_threshold = 0.01f;
  nms_threshold = 0.5f;
  mean_vals[0] = 103.53f;
  mean_vals[1] = 116.28f;
  mean_vals[2] = 123.675f;

  norm_vals[0] = 0.017429f;
  norm_vals[1] = 0.017507f;
  norm_vals[2] = 0.017125f;

  nanodet.load_param("./data/nanodet_m.param");
  nanodet.load_model("./data/nanodet_m.bin");
  nanodet.opt.num_threads=4;
}

YoloV2::YoloV2(std::string name_input)
{
  name = name_input;
  numOutput = 2;
  numThreads = 4;
  numAnchor = 3;
  numCategory = 80;
  nmsThresh = 0.25;
  prob_threshold = 0.01;
  inputWidth = 352;
  inputHeight = 352;
  //anchor box w h
  std::vector<float> bias {12.64, 19.39, 37.88,51.48, 55.71, 138.31,
    126.91, 78.23, 131.57, 214.55, 279.92, 258.87};
  anchor.assign(bias.begin(), bias.end());

  net.opt.use_winograd_convolution = true;
  net.opt.use_sgemm_convolution = true;
  net.opt.use_int8_inference = true;
  net.opt.use_vulkan_compute = false;
  net.opt.use_fp16_packed = true;
  net.opt.use_fp16_storage = true;
  net.opt.use_fp16_arithmetic = true;
  net.opt.use_int8_storage = true;
  net.opt.use_int8_arithmetic = true;
  net.opt.use_packing_layout = true;
  net.opt.use_shader_pack8 = false;
  net.opt.use_image_storage = false;
  net.load_param("./data/yolo-fastestv2-opt.param");
  net.load_model("./data/yolo-fastestv2-opt.bin");
}


std::string YoloV2::getName(){
  return name;
}

std::string YoloV8::getName(){
  return name;
}

std::string NanoDet::getName(){
  return name;
}


int YoloV2::predHandle(const ncnn::Mat *out,
                       std::vector<Object> &dstObjects,
                       const float scaleW, const float scaleH,
                       const float thresh)
{
  for (int i = 0; i < numOutput; i++) {
    int stride;
    int outW, outH, outC;

    outH = out[i].c;
    outW = out[i].h;
    outC = out[i].w;

    assert(inputHeight / outH == inputWidth / outW);
    stride = inputHeight / outH;

    for (int h = 0; h < outH; h++) {
      const float* values = out[i].channel(h);

      for (int w = 0; w < outW; w++) {
        for (int b = 0; b < numAnchor; b++) {
          //float objScore = values[4 * numAnchor + b];
          // TargetBox tmpBox;
          Object tmpBox;
          int category = -1;
          float score = -1;

          getCategory(values, b, category, score);

          if (score > thresh) {
            float bcx, bcy, bw, bh;
            float x1, y1, x2, y2;

            bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
            bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
            bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
            bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

            x1 = (bcx - 0.5 * bw) * scaleW;
            y1 = (bcy - 0.5 * bh) * scaleH;
            x2 = (bcx + 0.5 * bw) * scaleW;
            y2 = (bcy + 0.5 * bh) * scaleH;

            tmpBox.rect.x = x1;
            tmpBox.rect.y = y1;
            tmpBox.rect.width = x2 - x1;
            tmpBox.rect.height = y2 - y1;
            tmpBox.prob = score;
            tmpBox.cate = category;
            tmpBox.name = yolov2_class_names[category];
            dstObjects.push_back(tmpBox);
          }
        }
        values += outC;
      }
    }
  }
  return 0;
}




int NanoDet::detect(const cv::Mat& bgr, InferenceResult &result)
{
  std::chrono::steady_clock::time_point start, end;

  // PREPROCESS //////////
  start = std::chrono::steady_clock::now();

  int width = bgr.cols;
  int height = bgr.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data,
                                                 ncnn::Mat::PIXEL_BGR,
                                                 width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                           wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    end = std::chrono::steady_clock::now();
    result.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count();


    ncnn::Extractor ex = nanodet.create_extractor();
    std::vector<Object> proposals;
    ncnn::Mat cls_pred_792;
    ncnn::Mat dis_pred_795;
    std::vector<Object> objects8;

    ncnn::Mat cls_pred_814;
    ncnn::Mat dis_pred_817;
    std::vector<Object> objects16;

    ncnn::Mat cls_pred_836;
    ncnn::Mat dis_pred_839;
    std::vector<Object> objects32;

    // INFERENCE /////////////
    start = std::chrono::steady_clock::now();
    ex.input("input.1", in_pad);

    // stride 8
    ex.extract("792", cls_pred_792);
    ex.extract("795", dis_pred_795);

    // stride 16
    ex.extract("814", cls_pred_814);
    ex.extract("817", dis_pred_817);

    // stride 32
    ex.extract("836", cls_pred_836);
    ex.extract("839", dis_pred_839);

    end = std::chrono::steady_clock::now();
    result.inference = std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count();


    // POSTPROCESS ///////
    // generate proposal
    start = std::chrono::steady_clock::now();

    nanodet_generate_proposals(cls_pred_792, dis_pred_795, 8, in_pad, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    nanodet_generate_proposals(cls_pred_814, dis_pred_817, 16, in_pad, prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());

    nanodet_generate_proposals(cls_pred_836, dis_pred_839, 32, in_pad, prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());

    // sort all proposals by score from
    // highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    result.objects.resize(count);
    for (int i = 0; i < count; i++)
      {
        result.objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (result.objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (result.objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (result.objects[i].rect.x + result.objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (result.objects[i].rect.y + result.objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        result.objects[i].rect.x = x0;
        result.objects[i].rect.y = y0;
        result.objects[i].rect.width = x1 - x0;
        result.objects[i].rect.height = y1 - y0;
      }

    end = std::chrono::steady_clock::now();

    result.postprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count();
    return 0;
}






int YoloV8::detect(const cv::Mat& rgb, InferenceResult &result){
  float prob_threshold = 0.01f;
  float nms_threshold = 0.5f;

  // we want to measure some execution times
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> end;
  std::chrono::duration<double> elapsed;

  // PREPROCESS ///////////////////////////////
  start = std::chrono::steady_clock::now();

  int width = rgb.cols;
  int height = rgb.rows;
  // pad to multiple of 32
  int w = width;
  int h = height;
  float scale = 1.f;
  if (w > h)
    {
      scale = (float)target_size / w;
      w = target_size;
      h = h * scale;
    }
  else
    {
      scale = (float)target_size / h;
      h = target_size;
      w = w * scale;
    }
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                                               rgb.data,
                                               ncnn::Mat::PIXEL_RGB2BGR,
                                               width, height, w, h);

  // pad to target_size rectangle
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(
                         in, in_pad, hpad / 2, hpad - hpad / 2,
                         wpad / 2, wpad - wpad / 2,
                         ncnn::BORDER_CONSTANT, 0.f
                         );

  in_pad.substract_mean_normalize(0, norm_vals);

  std::vector<Object> proposals;
  ncnn::Mat out;
  ncnn::Extractor ex = yolo.create_extractor();

  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  result.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();


  // INFERENCE /////////////////////////////////
  start = std::chrono::steady_clock::now();
  ex.input("images", in_pad);
  ex.extract("output", out);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  result.inference = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();


  // POSTPROCESS /////////////////////////////
  start = std::chrono::steady_clock::now();

  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  std::vector<GridAndStride> grid_strides;
  generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
  yolov8_generate_proposals(grid_strides, out, prob_threshold, proposals);

  // sort all proposals by score from highest to lowest
  qsort_descent_inplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, nms_threshold);

  int count = picked.size();

  result.objects.resize(count);
  for (int i = 0; i < count; i++)
    {
      result.objects[i] = proposals[picked[i]];

      // adjust offset to original unpadded
      float x0 = (result.objects[i].rect.x - (wpad / 2)) / scale;
      float y0 = (result.objects[i].rect.y - (hpad / 2)) / scale;
      float x1 = (result.objects[i].rect.x + result.objects[i].rect.width - (wpad / 2)) / scale;
      float y1 = (result.objects[i].rect.y + result.objects[i].rect.height - (hpad / 2)) / scale;

      // clip
      x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
      y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
      x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
      y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

      result.objects[i].rect.x = x0;
      result.objects[i].rect.y = y0;
      result.objects[i].rect.width = x1 - x0;
      result.objects[i].rect.height = y1 - y0;
    }

  // sort objects by area
  struct
  {
    bool operator()(const Object& a, const Object& b) const
    {
      return a.rect.area() > b.rect.area();
    }
  } objects_area_greater;
  std::sort(result.objects.begin(), result.objects.end(), objects_area_greater);

  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  result.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

  return 0;
}



int YoloV2::detect(const cv::Mat& rgb, InferenceResult &result){
  float thresh = prob_threshold;
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> end;

  // PREPROCESSING
  start = std::chrono::steady_clock::now();
  // dstBoxes.clear();
  float scaleW = (float)rgb.cols / (float)inputWidth;
  float scaleH = (float)rgb.rows / (float)inputHeight;

  //resize of input image data
  ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(
                                                     rgb.data,
                                                     ncnn::Mat::PIXEL_BGR, \
                                                     rgb.cols,
                                                     rgb.rows,
                                                     inputWidth,
                                                     inputHeight
                                                     );

  //Normalization of input image data
  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};

  inputImg.substract_mean_normalize(mean_vals, norm_vals);

  end = std::chrono::steady_clock::now();
  result.preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // INFERENCE
  start = std::chrono::steady_clock::now();
  ncnn::Extractor ex = net.create_extractor();
  ncnn::Mat out[2];
  ex.set_num_threads(numThreads);
  ex.input("input.1", inputImg);
  ex.extract("794", out[0]); //22x22
  ex.extract("796", out[1]); //11x11
  end = std::chrono::steady_clock::now();
  result.inference = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // POSTPROCESSING
  start = std::chrono::steady_clock::now();
  std::vector<Object> tmpObjects;
  predHandle(out, tmpObjects, scaleW, scaleH, thresh);
  nmsHandle(tmpObjects, result.objects);
  end = std::chrono::steady_clock::now();
  result.postprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  return 0;
}


int YoloV2::getCategory(const float *values, int index, int &category, float &score)
{
  float tmp = 0;
  float objScore  = values[4 * numAnchor + index];
  for (int i = 0; i < numCategory; i++) {
    float clsScore = values[4 * numAnchor + numAnchor + i];
    clsScore *= objScore;

    if(clsScore > tmp) {
      score = clsScore;
      category = i;
      tmp = clsScore;
    }
  }
  return 0;
}


bool scoreSort(Object a, Object b){
  return a.prob > b.prob;
}

int YoloV2::nmsHandle(std::vector<Object> &tmpBoxes,
                      std::vector<Object> &dstBoxes)
{
  std::vector<int> picked;

  sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

  for(size_t i = 0; i < tmpBoxes.size(); i++) {
    int keep = 1;
    for(size_t j = 0; j < picked.size(); j++) {
      float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
      float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
      float IoU = inter_area / union_area;
      if(IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
        keep = 0;
        break;
      }
    }
    if (keep) {
      picked.push_back(i);
    }
  }

  for(size_t i = 0; i < picked.size(); i++) {
    dstBoxes.push_back(tmpBoxes[picked[i]]);
  }
  return 0;
}


int drawObjects(cv::Mat& rgb, const std::vector<Object>& objects)
{
  for (size_t i = 0; i < objects.size(); i++)
    {
      const Object& obj = objects[i];
      cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));

      char text[256];
      sprintf(text, "%s %.1f%%", obj.name, obj.prob * 100);

      int baseLine = 0;
      cv::Size label_size = cv::getTextSize(text,
                                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                            1, &baseLine
                                            );

      int x = obj.rect.x;
      int y = obj.rect.y - label_size.height - baseLine;
      if (y < 0)
        y = 0;
      if (x + label_size.width > rgb.cols)
        x = rgb.cols - label_size.width;

      cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                  cv::Size(label_size.width,
                                           label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

      cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
  return 0;
}


void printFileInferenceMeasurementsAsJson(
                                        FILE *file,
                                        std::vector<FileInferenceResult> &measures
                                        ){

  fprintf(file, "{");
  for(unsigned long m = 0; m<measures.size(); m++){
    auto &measure = measures[m];
    fprintf(file, "\"%s\":{", measure.filename.c_str());

    fprintf(file, "\"preprocess\": %f, \"inference\": %f, \"postprocess\": %f,",
            measure.result.preprocess,
            measure.result.inference,
            measure.result.postprocess
            );

    fprintf(file, "\"xywhsl\":[");
    for(unsigned long i=0; i<measure.result.objects.size(); i++){
      fprintf(file, "[%f, %f, %f, %f, %f, \"%s\"]",
              measure.result.objects[i].rect.x,
              measure.result.objects[i].rect.y,
              measure.result.objects[i].rect.width,
              measure.result.objects[i].rect.height,
              measure.result.objects[i].prob,
              measure.result.objects[i].name
              );
      if(i != measure.result.objects.size()-1){
        fprintf(file, ",");
      }
    }
    fprintf(file, "]");
    fprintf(file, "}");

    if(m != measures.size() - 1){
      fprintf(file, ",");
    }
  }
  fprintf(file, "}");
}



void printFileInferenceMeasurementsAsJson2(
                                        FILE *file,
                                        std::vector<FileInferenceResult> &measures
                                        ){

  fprintf(file, "[");
  for(unsigned long m = 0; m<measures.size(); m++){
    auto &measure = measures[m];
    for(unsigned long i=0; i<measure.result.objects.size(); i++){
      fprintf(file,
              "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\": %f}",
              measure.file_id,
              measure.result.objects[i].cate + 1 ,
              measure.result.objects[i].rect.x,
              measure.result.objects[i].rect.y,
              measure.result.objects[i].rect.x + measure.result.objects[i].rect.width,
              measure.result.objects[i].rect.y + measure.result.objects[i].rect.height,
              measure.result.objects[i].prob
              );
      // fprintf(file, "[%f, %f, %f, %f, %f, \"%s\"]",
      //         measure.result.objects[i].rect.x,
      //         measure.result.objects[i].rect.y,
      //         measure.result.objects[i].rect.width,
      //         measure.result.objects[i].rect.height,
      //         measure.result.objects[i].prob,
      //         measure.result.objects[i].name
      //         );
      if(i != measure.result.objects.size()-1){
        fprintf(file, ",");
      }

    }
    if(m != measures.size() - 1){
      fprintf(file, ",");
    }
  }
  fprintf(file, "]");
}


void saveImageDirectoryInferencesAsJson(Model &model, const char *dirname,
                                        std::string outputfilename){

  std::vector<FileInferenceResult> measurements;

  // foreach image in the folder
  // we will do our
  // model measurements
  DIR *dr;
  struct dirent *en;
  dr = opendir(dirname);

  std::string suffix = ".jpg";

  if (dr) {
    while ((en = readdir(dr)) != NULL) {
      if(en->d_name[0] != '.') {

        FileInferenceResult measure;
        std::string file_name = std::string(en->d_name);
        file_name.erase(0, file_name.find_first_not_of('0'));
        file_name.erase(
                               file_name.length() - suffix.length(),
                               suffix.length()
                               );

        measure.file_id = std::stoi(file_name, nullptr);

        std::string filepath = std::string(dirname) + std::string(en->d_name);
        measure.filename = filepath;
        cv::Mat image = cv::imread(filepath.c_str(), 1);
        if (image.empty()){
          fprintf(stderr, "cv::imread %d failed\n", measure.file_id);
          continue;
        }
        model.detect(image, measure.result);
        measurements.push_back(measure);
      }
    }
    closedir(dr);
  } else {
    fprintf(stderr, "expected an imagefolder \n");
  }


  std::sort(measurements.begin(), measurements.end(),
       [](const FileInferenceResult& d1, const FileInferenceResult& d2) {
         return d1.file_id < d2.file_id;
       });

  // Output the results
  // to a json file
  // const char *result_file_name = "measurement_yolov8nNcnn.json";
  FILE *file = fopen(outputfilename.c_str(), "w");
  if(!file) {
    fprintf(stderr, "Couldn't create file to write\n");
  } else {
    printFileInferenceMeasurementsAsJson(file, measurements);
    fclose(file);
  }
}





