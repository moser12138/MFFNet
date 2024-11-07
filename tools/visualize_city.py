import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.MFFNet_city import SemanticSegmentationNet
import torch.nn.functional as F

# 定义 Cityscapes 官方的颜色映射，每个类别一个 RGB 颜色
CITYSCAPES_COLORMAP = np.array([
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32],  # bicycle
])


# 载入图像的预处理函数
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((1024, 2048)),  # 根据模型输入大小调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 创建批次维度
    return input_batch, input_image


# 加载模型并执行预测
def predict(model, input_batch, original_size):
    with torch.no_grad():
        model.eval()
        output = model(input_batch)  # 获取模型输出
        if isinstance(output, tuple):
            output = output[0]  # 假设主输出在tuple中的第一个

        # 获取每个像素的类别索引
        output_predictions = output.argmax(1)  # shape: [batch_size, height, width]

        # 上采样到原始图像的分辨率
        output_predictions = F.interpolate(output_predictions.unsqueeze(1).float(),
                                           size=original_size,
                                           mode='bilinear',
                                           align_corners=False).squeeze(1)

    return output_predictions.cpu().numpy()[0]  # 返回numpy数组，用于可视化


# 将类别索引映射为 Cityscapes 颜色
def decode_segmentation(mask):
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)

    for label in range(len(CITYSCAPES_COLORMAP)):
        idx = mask == label
        r[idx] = CITYSCAPES_COLORMAP[label, 0]
        g[idx] = CITYSCAPES_COLORMAP[label, 1]
        b[idx] = CITYSCAPES_COLORMAP[label, 2]

    return np.stack([r, g, b], axis=2)


# 可视化并保存分割结果
def visualize_segmentation(original_image, segmentation_mask, output_path):
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image)
    # plt.title("Original Image")
    #
    # plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask)
    plt.axis('off')

    # plt.title("Segmentation Result")

    plt.savefig(output_path,bbox_inches='tight', pad_inches = 0)
    plt.show()

def main(image_path, model_path, output_path):
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = SemanticSegmentationNet(19)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.to(device)  # 将模型移动到设备

    # 预处理图像
    input_batch, original_image = preprocess_image(image_path)
    input_batch = input_batch.to(device)  # 将输入图像移动到设备

    # 进行预测
    output_predictions = predict(model, input_batch,[1024,2048])
    # output_predictions = predict(model, input_batch,[2048,4096])

    # 将预测的类别映射为颜色
    segmentation_result = decode_segmentation(output_predictions)

    # 可视化结果并保存
    visualize_segmentation(original_image, segmentation_result, output_path)

if __name__ == "__main__":
    # 示例使用
    model_path = "../pretrained_models/MFFNet_city_79_25.pth"

    # image_path = "./picture/city/ex1.png"
    # image_path = "./picture/city/ex2.png"
    # image_path = "./picture/city/ex3.png"
    # image_path = "./picture/city/ex4.png"
    image_path = "./picture/city/ex5.png"

    # output_path = "./picture/city/output/ex1.png"
    # output_path = "./picture/city/output/ex2.png"
    # output_path = "./picture/city/output/ex3.png"
    # output_path = "./picture/city/output/ex4.png"
    output_path = "./picture/city/output/ex5.png"

    main(image_path, model_path, output_path)
