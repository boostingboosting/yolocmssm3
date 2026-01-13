# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR
from ultralytics import settings
settings.update({"mlflow": False})

def freeze_rgb_backbone(trainer):
    print("freeze_rgb_backbone..........................................................................")

    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("enc_rgb"):
            print("True..........................................................................")
            module.init_weights(
                pretrained="/home/zhangquan/clg/efficientvit_b1_r288.pt")
            module.eval()
            module.requires_grad = False



# if __name__ == '__main__':
  
#     # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBRGB6C-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
# # model.load(r'yolov8n.pt') # loading pretrain weights
#     model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolobackboneHead-cmssm15/weights/last.pt")
#     # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolobackboneHead-cmssm-8dir7/weights/last.pt")
#     # model.add_callback("on_train_start", freeze_rgb_backbone)
#     model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
#                 cache=False,   
#                 imgsz=640,
#                 epochs=150,
#                 batch=64,
#                 close_mosaic=0, ###关闭mosaic
#                 workers=8,
#                 device='0,1,2,3',
#                 # device='1',
#                 optimizer='SGD',  # using SGD
#                 resume=True,
#                 # resume='/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion20/weights/last.pt', # last.pt path
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 # pairs_rgb_ir=['infrared','visible'],
#                 pairs_rgb_ir=['visible','infrared'],
#                 use_simotm="RGBRGB6C",
#                 channels=6,  #
#                 project='runs/AntiUAV',
#                 # name='AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain001',
#                 name='150epoch-yolobackboneHead-cmssm',
#                 )

#         # Add a callback to put the frozen layers in eval mode to prevent BN values from changing



# if __name__ == '__main__':
  
#     model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBRGB6C-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
# # model.load(r'yolov8n.pt') # loading pretrain weights
#     # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi/weights/last.pt")
#     # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolobackboneHead-cmssm-8dir7/weights/last.pt")
#     # model.add_callback("on_train_start", freeze_rgb_backbone)
#     model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
#                 cache=False,   
#                 imgsz=640,
#                 epochs=150,
#                 batch=256,
#                 close_mosaic=0, ###关闭mosaic
#                 workers=8,
#                 device='4,5,6,7',
#                 # device='1',
#                 optimizer='SGD',  # using SGD
#                 resume=True,
#                 # resume='/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion20/weights/last.pt', # last.pt path
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 # pairs_rgb_ir=['infrared','visible'],
#                 pairs_rgb_ir=['visible','infrared'],
#                 use_simotm="RGBRGB6C",
#                 channels=6,  #
#                 project='runs/AntiUAV',
#                 # name='AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain001',
#                 name='150epoch-yolorgbt-adjuster-noLocal',
#                 )

      


if __name__ == '__main__':
  
    # model = RTDETR('/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/rt-detr-RGBT/rtdetr-resnet50-RGBT-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model = YOLO('/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/PicoDet-RGBT/PicoDet-s-RGBT-midfusion-cmssm.yaml')
    # model = YOLO('/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/hyper-yolo-RGBT/hyper-yolo-RGBT-midfusion-cmssm.yaml')
    # model = YOLO('/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/ppyoloe-RGBT/ppyoloe-s-midfusion-cmssm.yaml')
    
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/hyper-yolo-RGBT-midfusion-Affine-gain00110/weights/last.pt")
# model.load(r'yolov8n.pt') # loading pretrain weights
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi/weights/last.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolobackboneHead-cmssm-8dir7/weights/last.pt")
    # model.add_callback("on_train_start", freeze_rgb_backbone)

    # model = YOLO("/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/hyper-yolo-RGBT/hyper-yolo-RGBT-midfusion-cmssm.yaml")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/200epoch-yolo12-RGBT-midfusion-affine-hp13/weights/best.pt")
    model = YOLO("/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/models/12-RGBT/yolo12-RGBT-midfusion-cmssm.yaml")
    model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
                cache=False,   
                imgsz=640,
                epochs=200,
                batch=128,
                close_mosaic=0, ###关闭mosaic
                workers=32,
                # device='0,1',
                # device='2,3',
                # device='4,5',
                # device='6,7',
                # device='0,1,2,3',
                # device='4,5,6,7',
                device='0,1,2,3,4,5,6,7',
                # device='0,1,6,7',
                optimizer='SGD',  # using SGD
                resume=True,
                # resume='/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion20/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['infrared','visible'],
                pairs_rgb_ir=['visible','infrared'],
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/AntiUAV',
                # name='AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain001',
                # name='rtdetr-resnet50-RGBT-midfusion',
                name='200epoch-yolo12-RGBT-midfusion-affine05-hp',
                # name='ppyoloe-s-midfusion',
                # name='PicoDet-s-RGBT-midfusion',
                # name='PicoDet-s-RGBT-midfusion-cmssm',
                amp=False,
                )
                
