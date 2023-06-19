import os
import json
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np


# helper function
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


# set the path/directory
projectDir='/home/pi/rp4objectdetection/'
dataDir=f'{projectDir}datasets/coco-2017/'
dataType='val2017'
annFile='{}/raw/instances_{}.json'.format(dataDir,dataType)
imgDir=f'{dataDir}validation/data/'


# initialize Coco
coco = COCO(annFile)
imageInfos = coco.loadImgs(coco.getImgIds())

# for each existing image
# get the annotations
anns = {}
for imgInfo in imageInfos:
    # currently not
    # every file is there
    image_path = imgDir + imgInfo['file_name']
    if os.path.isfile(image_path):
        ann = coco.loadAnns(
            coco.getAnnIds(
                imgIds=imgInfo['id'])
        )
        anns[image_path] = ann

# now load all the measurement results of
# all the models
measurements = {}
print("load measurement results")
for dirpath, _, filenames in os.walk("/home/pi/project"):
    for filename in filenames:
        if filename.startswith('measurement') and filename.endswith('.json'):
            fullname = os.path.join(dirpath, filename)
            with open(fullname, "r") as f:
                measurements[filename] = json.load(f)

# get the speeds
model_speeds_inference = {}
model_speeds_preprocess = {}
model_speeds_postprocess = {}
model_speeds_total = {}

print("extract speed data")
for model_name, model_data in measurements.items():
    model_speeds_inference[model_name] = []
    model_speeds_preprocess[model_name] = []
    model_speeds_postprocess[model_name] = []
    model_speeds_total[model_name] = []

    for image_path in model_data:
        inf = int(model_data[image_path]['inference'])
        pre = int(model_data[image_path]['preprocess'])
        post = int(model_data[image_path]['postprocess'])

        model_speeds_inference[model_name].append(inf)
        model_speeds_preprocess[model_name].append(pre)
        model_speeds_postprocess[model_name].append(post)
        model_speeds_total[model_name].append(inf + pre + post)

        # image = cv2.imread(image_path)
        # for x,y,w,h,score,label in model_data[image_path]['xywhsl']:
        #     cv2.rectangle(image,
        #                   (int(x), int(y)), (int(x + w), int(y + h)), color=(255, 31, 0), thickness=2)
        # # save result
        # cv2.imwrite(f"./inference_images/{model_name}_{os.path.basename(os.path.normpath(image_path))}", image)



# calculate speed statistics
print("calculate statistics")
model_statistics = {}
for model_name, model_data in measurements.items():
    model_statistics[model_name] = {}
    print(model_name)

    preprocess_data = model_speeds_preprocess[model_name]
    postprocess_data = model_speeds_postprocess[model_name]
    inference_data = model_speeds_inference[model_name]
    total_data = model_speeds_total[model_name]

    model_statistics[model_name]['total_max'] = np.max(total_data)
    model_statistics[model_name]['total_avg'] = np.mean(total_data)
    model_statistics[model_name]['total_med'] = np.median(total_data)
    model_statistics[model_name]['total_min'] = np.min(total_data)
    model_statistics[model_name]['total_std'] = np.std(total_data)

    model_statistics[model_name]['pre_max'] = np.max(preprocess_data)
    model_statistics[model_name]['pre_avg'] = np.mean(preprocess_data)
    model_statistics[model_name]['pre_med'] = np.median(preprocess_data)
    model_statistics[model_name]['pre_min'] = np.min(preprocess_data)
    model_statistics[model_name]['pre_std'] = np.std(preprocess_data)

    model_statistics[model_name]['pre_rel'] = model_statistics[model_name]['pre_avg'] / model_statistics[model_name]['total_avg']

    model_statistics[model_name]['inf_max'] = np.max(inference_data)
    model_statistics[model_name]['inf_avg'] = np.mean(inference_data)
    model_statistics[model_name]['inf_med'] = np.median(inference_data)
    model_statistics[model_name]['inf_min'] = np.min(inference_data)
    model_statistics[model_name]['inf_std'] = np.std(inference_data)

    model_statistics[model_name]['inf_rel'] = model_statistics[model_name]['inf_avg'] / model_statistics[model_name]['total_avg']

    model_statistics[model_name]['post_max'] = np.max(postprocess_data)
    model_statistics[model_name]['post_avg'] = np.mean(postprocess_data)
    model_statistics[model_name]['post_med'] = np.median(postprocess_data)
    model_statistics[model_name]['post_min'] = np.min(postprocess_data)
    model_statistics[model_name]['post_std'] = np.std(postprocess_data)
    model_statistics[model_name]['post_rel'] = model_statistics[model_name]['post_avg'] / model_statistics[model_name]['total_avg']


print("write mAp Speed diagrams")
mApFileName="mapResult.json"
if os.path.isfile("./mapResult.json"):

    mApData = {}
    with open(mApFileName) as fp:
        mApData = json.load(fp)

    for metric in list(mApData.values())[0].keys():
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.suptitle(f"ms / {metric} graph")

        do_plot = True
        for model_name, model_stat in model_statistics.items():
            y = mApData[model_name][metric] * 100
            x = round(model_statistics[model_name]['total_med'])

            labelname = os.path.splitext(model_name)[0]
            labelname = remove_prefix(labelname, 'measurement_')

            color = [[
                (hash(model_name) % 7) / 7,
                (hash(model_name) % 23) / 23,
                (hash(model_name) % 43) / 43,
            ]]

            if y > 0:
                ax.scatter(
                    x, y,
                    c=color,
                    label=f"{labelname}: ({x} ms, {round(y,2)} {metric})",
                )
                ax.set_xlabel("speed in ms")
                ax.set_ylabel(f"% {metric}")
                ax.text(x + 3, y, f"{labelname}", fontsize=12)
                # ax.text(x, -0.01,f"{labelname}", rotation=-45, rotation_mode='anchor')
            else:
                do_plot = False
                break


        if do_plot:
            plt.legend(
                # bbox_to_anchor=(1.15, 0.2),
                       # loc='lower center'
                loc='upper left',
                prop={"size":12}
                       )
            plt.tight_layout()
            plt.savefig(f"{metric}_result_scatter2.jpg")



    # x = model_statistics[model_name]['total_med'],
    #     y = mApData[model_name][measurement_key]
    #     if y > 0:
    #         fig, ax = plt.subplots(figsize=(8, 5))
    #         ax.scatter(
    #             x,
    #             y,
    #             c=np.random.rand(3,),
    #             label=f"{model_name} {measurement_key}"
    #         )

    # ax.legend()
    # plt.savefig(f"result_scatter.jpg")

# print("write speed statistic images")
# for model_name, model_statistic in model_statistics.items():
#     fig, axs = plt.subplots(4, 3, figsize=(10, 15), width_ratios=[1, 1, 3])
#     fig.subplots_adjust(hspace=0.6, wspace=0.2)
#     modeltitle = os.path.splitext(model_name)[0]

#     fig.suptitle(f"{modeltitle}")
#     axs[0, 0].set_title("total statistic")
#     total_text = axs[0, 0].text(
#         0.1, 0.90,
#         "max: {}\navg:{}\nmed:{}\nmin:{}\nstd:{}".format(
#             model_statistic['total_max'],
#             model_statistic['total_avg'],
#             model_statistic['total_med'],
#             model_statistic['total_min'],
#             round(model_statistic['total_std'], 2)
#         ),
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='top'
#     )
#     axs[0, 0].set_xticklabels([])
#     axs[0, 0].set_yticklabels([])
#     axs[0, 0].spines['top'].set_visible(False)
#     axs[0, 0].spines['bottom'].set_visible(False)
#     axs[0, 0].spines['left'].set_visible(False)
#     axs[0, 0].spines['right'].set_visible(False)
#     axs[0, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

#     axs[0, 1].boxplot(model_speeds_total[model_name])
#     axs[0, 1].set_ylabel("duration ms")
#     axs[0, 1].set_xticklabels([])

#     axs[0, 2].set_xlabel("picture")
#     axs[0, 2].plot(model_speeds_total[model_name])

#     axs[1, 0].set_title("Inference statistic")
#     total_text = axs[1, 0].text(
#         0.1, 0.90,
#         "relevance:{:.0%}\nmax:{}\navg:{}\nmed:{}\nmin:{}\nstd:{}".format(
#             model_statistic['inf_rel'],
#             model_statistic['inf_max'],
#             model_statistic['inf_avg'],
#             model_statistic['inf_med'],
#             model_statistic['inf_min'],
#             round(model_statistic['inf_std'], 2)
#         ),
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='top'
#     )
#     axs[1, 0].set_xticklabels([])
#     axs[1, 0].set_yticklabels([])
#     axs[1, 0].spines['top'].set_visible(False)
#     axs[1, 0].spines['bottom'].set_visible(False)
#     axs[1, 0].spines['left'].set_visible(False)
#     axs[1, 0].spines['right'].set_visible(False)
#     axs[1, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

#     axs[1, 1].boxplot(model_speeds_inference[model_name])
#     axs[1, 1].set_ylabel("duration ms")
#     axs[1, 1].set_xticklabels([])

#     axs[1, 2].set_xlabel("picture")
#     axs[1, 2].plot(model_speeds_inference[model_name])


#     axs[2, 0].set_title("Preprocess statistic")
#     total_text = axs[2, 0].text(
#         0.1, 0.90,
#         "relevance:{:.0%}\nmax:{}\navg:{}\nmed:{}\nmin:{}\nstd:{}".format(
#             model_statistic['pre_rel'],
#             model_statistic['pre_max'],
#             model_statistic['pre_avg'],
#             model_statistic['pre_med'],
#             model_statistic['pre_min'],
#             round(model_statistic['pre_std'], 2)
#         ),
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='top'
#     )
#     axs[2, 0].set_xticklabels([])
#     axs[2, 0].set_yticklabels([])
#     axs[2, 0].spines['top'].set_visible(False)
#     axs[2, 0].spines['bottom'].set_visible(False)
#     axs[2, 0].spines['left'].set_visible(False)
#     axs[2, 0].spines['right'].set_visible(False)
#     axs[2, 0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

#     axs[2, 1].boxplot(model_speeds_preprocess[model_name])
#     axs[2, 1].set_ylabel("duration ms")
#     axs[2, 1].set_xticklabels([])

#     axs[2, 2].set_xlabel("picture")
#     axs[2, 2].plot(model_speeds_preprocess[model_name])

#     axs[3, 0].set_title("Postprocess statistic")
#     total_text = axs[3, 0].text(
#         0.1, 0.90,
#         "relevance:{:.0%}\nmax:{}\navg:{}\nmed:{}\nmin:{}\nstd:{}".format(
#             model_statistic['post_rel'],
#             model_statistic['post_max'],
#             model_statistic['post_avg'],
#             model_statistic['post_med'],
#             model_statistic['post_min'],
#             round(model_statistic['post_std'], 2)
#         ),
#         fontsize=12,
#         horizontalalignment='left',
#         verticalalignment='top'
#     )
#     axs[3, 0].set_xticklabels([])
#     axs[3, 0].set_yticklabels([])
#     axs[3, 0].spines['top'].set_visible(False)
#     axs[3, 0].spines['bottom'].set_visible(False)
#     axs[3, 0].spines['left'].set_visible(False)
#     axs[3, 0].spines['right'].set_visible(False)
#     axs[3, 0].tick_params(
#         axis='both', which='both',
#         bottom=False, top=False,
#         left=False, right=False
#     )
#     axs[3, 1].boxplot(model_speeds_postprocess[model_name])
#     axs[3, 1].set_ylabel("duration ms")
#     axs[3, 1].set_xticklabels([])
#     axs[3, 2].set_xlabel("picture")
#     axs[3, 2].plot(model_speeds_postprocess[model_name])

#     plt.savefig(f"{modeltitle}_statistics.jpg")
