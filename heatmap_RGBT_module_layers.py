from ultralytics import YOLO
# 导入你的目标模块（替换为你要查询的模块类，核心修改点1）
from ultralytics.nn.modules.conv import SilenceList

# 1. 加载YOLO模型（替换为你的权重路径，核心修改点2）
model = YOLO("/home/zhangquan/clg/yolocmssm2/results/150epoch-yolorgbt-adjuster/weights/best.pt")
# 切换到评估模式（不影响层级遍历，可选）
model.eval()

# 2. 遍历所有模块，分配全局连续索引，定位目标模块
target_layer_class = SilenceList  # 目标模块类
global_layer_index = 0  # 全局层索引（从0开始，可改为1开始，与你的统计习惯一致）
target_layer_info = []  # yaml文件中的层级，热力图使用的是这个

# 关键：named_modules() 遍历所有嵌套模块（与框架统计layers的逻辑一致）
for layer_name, layer_module in model.model.named_modules():
    # 跳过模型本身（避免将整个model算作一层，保证与layers统计一致）
    if layer_module is model.model:
        continue
    
    # 3. 匹配目标模块
    if isinstance(layer_module, target_layer_class):
        # 记录目标模块的信息
        target_layer_info.append({
            "global_layer_index": global_layer_index,
            "layer_name": layer_name,
            "layer_module": layer_module
        })
        print(f"找到目标模块！")
        print(f"  全局层索引（对应layers统计）：{global_layer_index}")
        print(f"  网络层级名称：{layer_name}")
        print(f"  模块对象：{layer_module.__class__.__name__}")
    
    # 4. 全局索引自增（每遍历一个模块，层号+1，对应总layers数）
    global_layer_index += 1

# 5. 输出汇总信息
print("=" * 60)
print(f"模型总模块数（与输出的layers一致）：{global_layer_index}")
if len(target_layer_info) == 0:
    print(f"未找到 {target_layer_class.__name__} 模块，请检查模块类名是否正确")
else:
    print(f"共找到 {len(target_layer_info)} 个 {target_layer_class.__name__} 模块，信息如下：")
    for info in target_layer_info:
        print(f"  层索引：{info['global_layer_index']}，层级名称：{info['layer_name']}")