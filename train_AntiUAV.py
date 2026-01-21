# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR
from ultralytics import settings
settings.update({"mlflow": False})


if __name__ == '__main__':

    # model = YOLO("/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/hyper-yolo-RGBT/hyper-yolo-RGBT-midfusion-cmssm.yaml")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/200epoch-yolo12-RGBT-midfusion-affine-hp13/weights/best.pt")
    model = YOLO("/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/12-RGBT/yolo12-affine-hg.yaml")
    model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
                cache=False,   
                imgsz=640,
                epochs=200,
                batch=64,
                close_mosaic=0, ###关闭mosaic
                workers=8,
                device='0,1,6,7',
                # device='2,3',
                # device='4,5',
                # device='6,7',[]
                # device='0,1,2,3',
                # device='4,5,6,7',
                # device='0,1,2,3,4,5,6,7',
                # device='2,3,4,5',
                channels=6,  #
                project='runs/AntiUAV',
                # name='AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain001',
                # name='rtdetr-resnet50-RGBT-midfusion',
                name='200epoch-yolo12-affine2-localoffset-hg168',
                # name='ppyoloe-s-midfusion',
                # name='PicoDet-s-RGBT-midfusion',
                # name='PicoDet-s-RGBT-midfusion-cmssm',
                amp=False,
                optimizer='SGD',  # using SGD
                resume=True,
                # resume='/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion20/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['infrared','visible'],
                pairs_rgb_ir=['visible','infrared'],
                use_simotm="RGBRGB6C",
                )
                
