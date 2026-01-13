import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch as torch


if __name__ == '__main__':
  
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBRGB6C-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
# model.load(r'yolov8n.pt') # loading pretrain weights

    model.train(data=R'ultralytics/cfg/datasets/FLIR_aligned-rgbt.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=2,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/FLIR',
                name='FLIR-yolo11n-RGBRGB6C-midfusion',
                )