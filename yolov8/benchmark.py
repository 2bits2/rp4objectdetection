from ultralytics.yolo.utils.benchmarks import benchmark
import os, glob, shutil

half=False
imagesize=640
ptmodelnames = ['yolov8n']
int8=True

for ptmodelname in ptmodelnames:
    print(f'testing {ptmodelname}.pt')
    benchmark(model=ptmodelname, imgsz=imagesize, half=half, int8=int8)
    print('deleting model specific benchmark leftovers')
    for filename in glob.glob(f'./{ptmodelname}*'):
        if os.path.isfile(filename):
            os.remove(filename)
        elif os.path.isdir(filename):
            shutil.rmtree(filename)
