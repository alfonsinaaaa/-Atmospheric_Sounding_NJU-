import cv2
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def remove_black_border(image, threshold=0.01):
    if image is None:
        return None
    mask = image > threshold
    if not np.any(mask):
        return image
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image[rmin:rmax+1, cmin:cmax+1]

def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Img = Img / 255.0
        Img = remove_black_border(Img)
        if Img is None:
            return None
        Img = cv2.resize(Img, target_size)
        return Img
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def calculate_cloud_coverage(image):
    if image is None:
        return 0.0
    img_uint8 = (image * 255).astype(np.uint8) # 使用自适应阈值分割，让分析更适合不同光照条件
    binary = cv2.adaptiveThreshold(img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # 去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


    white_pixels = np.sum(binary == 255) # 计算白色像素比例
    total_pixels = binary.size
    cloud_coverage = white_pixels / total_pixels
    return cloud_coverage

def classify_cloud_coverage(coverage):
    if coverage < 0.1:
        return 0  # 无云
    elif coverage < 0.3:
        return 1  # 少云
    elif coverage < 0.7:
        return 2  # 多云
    else:
        return 3  # 阴天


def extract_features(image):
    if image is None:
        return None
    coverage = calculate_cloud_coverage(image) # 计算云量
    # 计算图像的统计特征
    mean = np.mean(image)
    std = np.std(image)
    max_val = np.max(image)
    min_val = np.min(image)
    # 边缘检测
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size


    Img = (image * 255).astype(np.uint8) # 纹理特征
    try:
        # 计算灰度直方图
        hist = cv2.calcHist([Img], [0], None, [256], [0, 256])
        hist_mean = np.mean(hist)
        hist_std = np.std(hist)
    except:
        hist_mean = 0
        hist_std = 0

    # 特征向量，添加更多特征以区分无云和少云
    features = [coverage, mean, std, max_val, min_val, edge_density, hist_mean, hist_std]
    return features

def load_dataset(DATE): # 数据集
    images = []
    labels = []

    for filename in os.listdir(DATE):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(DATE, filename)
            image = preprocess_image(image_path)
            if image is not None:
                features = extract_features(image)
                if features is not None:
                    images.append(features)
                    # 提取云量标签（第一个数字）
                    try:
                        # 分割文件名
                        if '_' in filename:
                            parts = filename.split('_')
                        elif '-' in filename:
                            parts = filename.split('-')
                        else:
                            parts = []
                        if len(parts) >= 1:
                            cloud_label = int(parts[0])
                            # 根据云量标签映射到分类
                            if cloud_label == 1:
                                label = 0  # 无云
                            elif cloud_label == 2:
                                label = 1  # 少云
                            elif cloud_label == 3:
                                label = 2  # 多云
                            elif cloud_label == 4:
                                label = 3  # 阴天
                            else:
                                # 如果标签不在预期范围内，使用计算的云量
                                coverage = features[0]
                                label = classify_cloud_coverage(coverage)
                        else:
                            # 如果无法提取标签，使用计算的云量
                            coverage = features[0]
                            label = classify_cloud_coverage(coverage)
                    except:
                        # 如果提取失败，使用计算的云量
                        coverage = features[0]
                        label = classify_cloud_coverage(coverage)
                    labels.append(label)

    return np.array(images), np.array(labels)

def train_model(X_train, y_train):
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # MLPClassifier - 增加更多隐藏层
    model = MLPClassifier(hidden_layer_sizes=(100, 75, 50, 25), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# 开始计时
start_time = time.time()

data_dir = 'data_cloud_amount'

# 加载数据集
load_start = time.time()
x_data, y_data = load_dataset(data_dir)
load_end = time.time()
print(f"总数据集大小: {len(x_data)} 样本")
print(f"数据加载时间: {load_end - load_start:.2f} 秒")

# 分割数据集为80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print(f"训练集大小: {len(X_train)} 样本")
print(f"测试集大小: {len(X_test)} 样本")

# 训练模型
train_start = time.time()
model_obj, scaler_obj = train_model(X_train, y_train)
train_end = time.time()
print(f"模型训练时间: {train_end - train_start:.2f} 秒")

# 测试模型
test_start = time.time()
# 使用分割后的测试集进行测试
print("====================================")
print("测试集结果:")

# 统计实际和预测的类别数量
counts_a = [0, 0, 0, 0]  # 无云, 少云, 多云, 阴天
for label in y_test:
    counts_a[label] += 1

# 预测测试集
X_test_scaled = scaler_obj.transform(X_test)
predictions = model_obj.predict(X_test_scaled)

counts_p = [0, 0, 0, 0]  # 无云, 少云, 多云, 阴天
for label in predictions:
    counts_p[label] += 1

# 计算准确率
acc = accuracy_score(y_test, predictions)

# 输出结果
print("\n1. 实际的云量结果:")
print(f"无云: {counts_a[0]}张")
print(f"少云: {counts_a[1]}张")
print(f"多云: {counts_a[2]}张")
print(f"阴天: {counts_a[3]}张")

print("\n2. 测试的分类结果:")
print(f"无云: {counts_p[0]}张")
print(f"少云: {counts_p[1]}张")
print(f"多云: {counts_p[2]}张")
print(f"阴天: {counts_p[3]}张")

print(f"\n3. 测试准确率: {acc:.4f}")

test_end = time.time()
print(f"测试时间: {test_end - test_start:.2f} 秒")

# 总运行时间
end_time = time.time()
print(f"总运行时间: {end_time - start_time:.2f} 秒")






