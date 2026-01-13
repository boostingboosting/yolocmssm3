import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# if __name__ == '__main__':

#     '''
#         source 为图像的最终目录,需要和train/val目录一致，且需要包含visible字段，visible同级目录下存在infrared目录，原理是将visible替换为infrared，加载双光谱数据
        
#         "source" refers to the final directory for the images.
#         The source needs to be in the same directory as the train/val directories, and it must contain the "visible" field. 
#         There is an "infrared" directory at the same level as the "visible" directory. 
#         The principle is to replace "visible" with "infrared" and load the dual-spectrum data.
#     '''
#     model = YOLO(r"/home/zhangquan/clg/yolocmssm2/exp/AntiUAV-yolo11n-100epoch-RGBRGB6C-midfusion-cmssm/weights/best.pt") # select your model.pt path
#     model.predict(source=r'/home/zhangquan/clg/Anti-UAV-RGBT-640/visible/test',
#                   imgsz=640,
#                   project='runs/detect/cmssm',
#                   name='exp',
#                   show=False,
#                   save_frames=True,
#                   use_simotm="RGBRGB6C",
#                   channels=6,
#                   save=True,
#                   # conf=0.2,
#                   visualize=True # visualize model features maps
#                 )

from ultralytics import YOLO

# 1. 加载训练好的权重文件
model = YOLO("/home/zhangquan/clg/yolocmssm2/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion/weights/best.pt")  # 替换为你的权重路径

# 2. 执行测试集测试（关键：指定 split="test"）
results = model.val(
    data="/home/zhangquan/clg/yolocmssm2/ultralytics/cfg/datasets/AntiUAV-rgbt.yaml",  # 数据集配置文件
    split="test",           # 明确使用测试集
    save=True,              # 保存可视化结果（默认True）
    save_txt=True,          # 保存每个样本的预测结果到txt（可选）
    save_conf=True          # 保存预测置信度（可选）
)

# 3. 打印测试集核心指标（按需获取）
print("测试集 mAP50-95:", results.box.map)       # 目标检测：mAP50-95
print("测试集 mAP50:", results.box.map50)         # 目标检测：mAP50
print("测试集 mAP75:", results.box.map75)         # 目标检测：mAP75
print("各类别精度:", results.box.ap_classwise)    # 每个类别的AP指标
print("测试集图片数量:", results.dataset.n)       # 测试集样本总数