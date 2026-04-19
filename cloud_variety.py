import cv2
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 6个标签分别对应 无云, 卷云, 层状云, 积状云, 积雨云, 混合云 from zzq

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

def preprocess_image(image_path, target_size=(256,256)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gr = gr / 255.0
        gr = remove_black_border(gr)
        if gr is None:
            return None
        gr = cv2.resize(gr, target_size)
        return gr
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def calculate_cloud_coverage(image):
    if image is None:
        return 0.0
    img_uint8 = (image * 255).astype(np.uint8)  # 使用自适应阈值分割，更适合不同光照条件
    binary = cv2.adaptiveThreshold(img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # 去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    white_pixels = np.sum(binary == 255) # 计算白色像素比例（云）
    total_pixels = binary.size

    cloud_coverage = white_pixels / total_pixels
    return cloud_coverage

def classify_cloud_type(cloud_type):
    if cloud_type == 1:
        return '无云'
    elif cloud_type == 2:
        return '卷云'
    elif cloud_type == 3:
        return '层状云'
    elif cloud_type == 4:
        return '积状云'
    elif cloud_type == 5:
        return '积雨云'
    elif cloud_type == 6:
        return '混合云'
    else:
        return '未知'

def extract_features(image):
    if image is None:
        return None
    coverage = calculate_cloud_coverage(image)# 计算云量
    mean = np.mean(image) 
    std = np.std(image)
    max_val = np.max(image)
    min_val = np.min(image)
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    gr = (image * 255).astype(np.uint8)
    try:
        hist = cv2.calcHist([gr], [0], None, [256], [0, 256])
        hist_mean = np.mean(hist)
        hist_std = np.std(hist)
    except:
        hist_mean = 0
        hist_std = 0
    # 特征向量，添加更多特征以区分无云和少云
    features = [coverage, mean, std, max_val, min_val, edge_density, hist_mean, hist_std]
    return features
def load_dataset(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(data_dir, filename)
            image = preprocess_image(image_path)
            if image is not None:
                features = extract_features(image)
                if features is not None:
                    images.append(features)
                    # 文件名格式：[云类型数字]_[其他信息].jpg 或 [云类型数字]-[其他信息].jpg
                    try:
                        # 分割文件名
                        if '_' in filename:
                            parts = filename.split('_')
                        elif '-' in filename:
                            parts = filename.split('-')
                        else:
                            parts = []

                        if len(parts) >= 1:
                            cloud_type = int(parts[0])
                            if cloud_type >= 1 and cloud_type <= 6:
                                label = cloud_type - 1  # 转换为0-5的标签
                            else:
                                label = 0
                        else:
                            label = 0
                    except:
                        # 如果提取失败，使用0
                        label = 0

                    labels.append(label)

    return np.array(images), np.array(labels)

def train_model(X_train, y_train):
    """训练模型"""
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 模型
    model = MLPClassifier(hidden_layer_sizes=(100, 75, 50, 25), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def process_test_folder(test_dir, model, scaler):
    if not os.path.exists(test_dir):
        print(f"测试文件夹不存在: {test_dir}")
        return
    pr = []
    ar = []
    fs = []

    ci0 = []  # 无云
    ci1 = []  # 卷云
    ci2 = []  # 层状云
    ci3 = []  # 积状云
    ci4 = []  # 积雨云
    ci5 = []  # 混合云

    print("====================================")
    print("测试集结果:")
    for filename in os.listdir(test_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, filename)
            test_image = preprocess_image(image_path)
            if test_image is not None:
                features = extract_features(test_image)
                if features is not None:
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)[0]
                    actual_label = -1
                    try:
                        if '_' in filename:
                            parts = filename.split('_')
                        elif '-' in filename:
                            parts = filename.split('-')
                        else:
                            parts = []

                        if len(parts) >= 1:
                            cloud_type = int(parts[0])
                            if cloud_type >= 1 and cloud_type <= 6:
                                actual_label = cloud_type - 1
                    except:
                        pass
                    # 记录结果
                    fs.append(filename)
                    pr.append(prediction)
                    ar.append(actual_label)
                    if prediction == 0:
                        ci0.append(filename)
                    elif prediction == 1:
                        ci1.append(filename)
                    elif prediction == 2:
                        ci2.append(filename)
                    elif prediction == 3:
                        ci3.append(filename)
                    elif prediction == 4:
                        ci4.append(filename)
                    else:
                        ci5.append(filename)

    # 统计实际云类型结果
    ac = [0, 0, 0, 0, 0, 0]  
    for label in ar:
        if label != -1:
            ac[label] += 1
    pc = [0, 0, 0, 0, 0, 0] 
    for label in pr:
        pc[label] += 1

    # 计算准确率
    c = 0
    t = 0
    for i in range(len(ar)):
        if ar[i] != -1:
            t += 1
            if pr[i] == ar[i]:
                c += 1
    acc = c / t if t > 0 else 0
    print("\n1. 实际的云类型结果:")
    print(f"无云: {ac[0]}张")
    print(f"卷云: {ac[1]}张")
    print(f"层状云: {ac[2]}张")
    print(f"积状云: {ac[3]}张")
    print(f"积雨云: {ac[4]}张")
    print(f"混合云: {ac[5]}张")

    print("\n2. 测试的分类结果:")
    print(f"无云: {pc[0]}张")
    print(f"卷云: {pc[1]}张")
    print(f"层状云: {pc[2]}张")
    print(f"积状云: {pc[3]}张")
    print(f"积雨云: {pc[4]}张")
    print(f"混合云: {pc[5]}张")

    print(f"\n3. 测试准确率: {acc:.4f}")
    print("\n是否输出不同云类型对应的文件编号？(Y/N): ")
    ui = input().strip().lower()
    if ui == 'y':
        print("\n无云文件:")
        for img in ci0:
            print(img)
        print("\n卷云文件:")
        for img in ci1:
            print(img)
        print("\n层状云文件:")
        for img in ci2:
            print(img)
        print("\n积状云文件:")
        for img in ci3:
            print(img)
        print("\n积雨云文件:")
        for img in ci4:
            print(img)
        print("\n混合云文件:")
        for img in ci5:
            print(img)

    return pr, ar

# 开始计时
start_time = time.time()

d = 'data_cloud_variety'

# 加载数据集
load_start = time.time()
x, y = load_dataset(d)
load_end = time.time()
print(f"总数据集大小: {len(x)} 样本")
print(f"数据加载时间: {load_end - load_start:.2f} 秒")

# 分割数据集为80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"训练集大小: {len(X_train)} 样本")
print(f"测试集大小: {len(X_test)} 样本")

# 训练模型
train_start = time.time()
m, s = train_model(X_train, y_train)
train_end = time.time()
print(f"模型训练时间: {train_end - train_start:.2f} 秒")

# 测试模型
test_start = time.time()
print("====================================")
print("测试集结果:")

# 统计实际云类型结果
ac = [0, 0, 0, 0, 0, 0]  
for label in y_test:
    ac[label] += 1

# 预测测试集
X_test_scaled = s.transform(X_test)
predictions = m.predict(X_test_scaled)

pc = [0, 0, 0, 0, 0, 0] 
for label in predictions:
    pc[label] += 1

# 计算准确率
acc = accuracy_score(y_test, predictions)

print("\n1. 实际的云类型结果:")
print(f"无云: {ac[0]}张")
print(f"卷云: {ac[1]}张")
print(f"层状云: {ac[2]}张")
print(f"积状云: {ac[3]}张")
print(f"积雨云: {ac[4]}张")
print(f"混合云: {ac[5]}张")

print("\n2. 测试的分类结果:")
print(f"无云: {pc[0]}张")
print(f"卷云: {pc[1]}张")
print(f"层状云: {pc[2]}张")
print(f"积状云: {pc[3]}张")
print(f"积雨云: {pc[4]}张")
print(f"混合云: {pc[5]}张")

print(f"\n3. 测试准确率: {acc:.4f}")

test_end = time.time()
print(f"测试时间: {test_end - test_start:.2f} 秒")

# 总运行时间
end_time = time.time()
print(f"总运行时间: {end_time - start_time:.2f} 秒")
