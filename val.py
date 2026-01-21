import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR

#测试集验证
if __name__ == '__main__':
    # model = YOLO(R'LLVIP/LLVIP-yolov8-RGBT-midfusion/weights/best.pt')
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolorgbt-adjuster2/weights/best.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/results/150epoch-yolorgbt-adjuster/weights/best.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/results/AntiUAV-yolo11n-RGBRGB6C-midfusion/weights/best.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/results/AntiUAV-yolo11n-RGBRGB6C-midfusion-P3/weights/best.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi-flow/weights/best.pt")
    # model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/150epoch-yolorgbt-adjuster-noLocal4/weights/best.pt")
    model = YOLO("/home/zhangquan/clg/yolocmssm2/results/200epoch-yolo12-affine2-localoffset-hg168/weights/best.pt")
    model.val(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
              split='test',#测试集
              imgsz=640,
              batch=32,
              device='4',
              use_simotm="RGBRGB6C",
              channels=6,
              pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
            #   name='yolobackboneHead-150epoch-cmssm2',)
              name='200epoch-yolo12-affine2-localoffset-hg168',)