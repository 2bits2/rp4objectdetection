import json
import cv2
import os

yolov2Results = {}
with open("/home/pi/project/ncnncpp/yoloFastestV2.json", "r") as f:
    yolov2Results = json.load(f)



for image_path, detections in yolov2Results.items():
    # print(image_path)
    image = cv2.imread(image_path)

    for x, y, w, h, s, l in detections['xywhsl']:
        cv2.rectangle(image,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      color=(255, 31, 0),
                      thickness=2
                      )
        cv2.putText(image, l, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    new_image_name = "./inference_images/yolofastestv2_{}".format(
        os.path.basename(os.path.normpath(image_path))
    )
    # save result
    cv2.imwrite(new_image_name, image)

