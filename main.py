import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from astrbot.core.message.components import Plain, Reply
from astrbot.core.star.filter.event_message_type import EventMessageType
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

from typing import List, Dict

def create_model(num_classes, model_type):
    """创建指定类型的模型，确保与训练时结构一致"""
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'resnet18':
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model

# 优化的模型加载函数（减少不必要的调试信息）
def load_trained_model(model_path, device):
    """加载模型，优先尝试ResNet18（根据成功案例）"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = len(checkpoint['classes'])

    # 优先尝试ResNet18（因为之前已成功加载）
    model_types = ['resnet18', 'mobilenet_v2']  # 调整顺序，优先尝试成功的模型
    model = None

    for model_type in model_types:
        try:
            test_model = create_model(num_classes, model_type)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            model = test_model
            print(f"✅ 成功加载 {model_type} 模型权重")  # 清晰的成功提示
            break
        except RuntimeError:
            # 只在最后一个模型尝试失败时才显示错误
            if model_type == model_types[-1]:
                print(f"❌ 尝试加载 {model_type} 模型失败")
            continue

    if model is None:
        # 最后的兼容加载方案
        print("⚠️ 尝试兼容模式加载权重...")
        model = create_model(num_classes, 'resnet18')
        model_dict = model.state_dict()
        checkpoint_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
        print(f"⚠️ 兼容模式加载完成，匹配 {len(checkpoint_dict)}/{len(model_dict)} 个权重层")

    model = model.to(device)
    model.eval()
    return model, checkpoint['classes'], checkpoint['target_size']

# 图片预处理
def get_transform(target_size):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 单张图片预测
def predict_image(model, image_path, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        image = transform(image).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return preds[0].item(), probabilities[0][preds[0]].item() * 100 ,

    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {str(e)}")
        return None, None,

#加载模型
model_path = './data/plugins/astrbot_plugin_wmc_detector/feature_classifier.pth'
if not os.path.exists(model_path):
    print(f"错误: 模型文件 {model_path} 不存在")
    exit(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"加载模型 {model_path}...")
try:
    model, classes, target_size = load_trained_model(model_path, device)
    print(f"模型加载成功，目标类别: {classes}，图片统一尺寸: {target_size}x{target_size}")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)
transform = get_transform(target_size)

@register("wmc_detector", "TKworld", "awmc", "1.0")
class MyPlugin(Star):

    def __init__(self, context: Context ,config: dict ):
        super().__init__(context)
        self.config = config

    @filter.event_message_type(EventMessageType.ALL)
    async def detect_wmc(self, event: AstrMessageEvent):
        
        message_chain = event.get_messages()
        if len(message_chain) == 1:
            if message_chain[0].type == "Image":
                image_filename = message_chain[0].file
                image_url = message_chain[0].url
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import \
                    AiocqhttpMessageEvent
                assert isinstance(event, AiocqhttpMessageEvent)
                client = event.bot
                payloads2 = {
                    "file_id": image_filename
                }
                response = await client.api.call_action('get_image', **payloads2)
                image_path = response['file']
                str_path = str(image_path)
                if str_path.endswith('.gif'):
                    return

                if os.path.exists(image_path):
                    try:
                        result, prob = predict_image(model, image_path, transform, device)
                        print(f"预测结果: {result}, 置信度: {prob}")  # 补充打印prob，便于调试，0表示为wmc，置信度为百分数制（省略百分号）

                        if result is not None:  # 明确判断None，避免空字符串等假值被误判
                            if prob > 80 and result == 0: #置信度大于80%进行回复，可调整
                                message_chain = [Reply(id=event.message_obj.message_id), Plain(self.config.REPLY_CONTENT)]
                                yield event.chain_result(message_chain)
                        else:
                            print("未获得有效预测结果")
                            return  # 无有效结果时终止
                    except Exception as e:  # 捕获具体异常并打印，便于排查
                        print(f"预测过程出错: {str(e)}")
                    else:
                        pass
                else:
                    pass
