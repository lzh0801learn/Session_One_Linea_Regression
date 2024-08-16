import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

### 2. 加载数据
# 读取标准城市名
standard_cities = pd.read_csv('standard_cities.csv')
standard_cities_list = standard_cities['city_name'].tolist()

# 读取训练集和验证集
train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')

### 3. 数据预处理
#对城市名称进行预处理，例如去除空格、转换为小写等。

def preprocess_city_name(name):
    name = name.strip().lower()
    return name
train_data['city'] = train_data['city'].apply(preprocess_city_name)
validation_data['city'] = validation_data['city'].apply(preprocess_city_name)
### 4. 特征提取与标签编码
#使用TF-IDF向量化器将城市名转换为特征向量，并使用标签编码器将标准城市名转换为标签。
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

# 使用训练集的城市名和标准城市名来拟合向量化器
X_train = vectorizer.fit_transform(train_data['city'])
y_train = label_encoder.fit_transform(train_data['standard_city'])
# 验证集
X_val = vectorizer.transform(validation_data['city'])
y_val = label_encoder.transform(validation_data['standard_city'])
### 5. 训练线性回归模型

#选择 Ridge Regression 进行训练，因为它在处理多重共线性方面表现较好。

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

### 6. 模型验证
y_pred = model.predict(X_val)
y_pred_labels = np.round(y_pred).astype(int)
y_pred_labels = label_encoder.inverse_transform(y_pred_labels)

validation_data['predicted_city'] = y_pred_labels

# 计算准确率
accuracy = accuracy_score(validation_data['standard_city'], y_pred_labels)
print(f'Validation Accuracy: {accuracy}')

### 7. 数据清洗
def clean_city_names(data, model, vectorizer, label_encoder):
    data['city'] = data['city'].apply(preprocess_city_name)
    X = vectorizer.transform(data['city'])
    y_pred = model.predict(X)
    y_pred_labels = np.round(y_pred).astype(int)
    data['cleaned_city'] = label_encoder.inverse_transform(y_pred_labels)
    return data

# 假设你有一个新的待清洗的数据集
new_data = pd.read_csv('new_data.csv')
cleaned_data = clean_city_names(new_data, model, vectorizer, label_encoder)

# 保存清洗后的数据
cleaned_data.to_csv('cleaned_data.csv', index=False)

### 8. 半监督学习
# 将无标签数据和有标签数据结合
unlabeled_data = pd.read_csv('unlabeled_data.csv')
unlabeled_data['city'] = unlabeled_data['city'].apply(preprocess_city_name)

# 使用现有模型预测标签
unlabeled_X = vectorizer.transform(unlabeled_data['city'])
unlabeled_y_pred = model.predict(unlabeled_X)
unlabeled_y_pred_labels = np.round(unlabeled_y_pred).astype(int)
unlabeled_data['standard_city'] = label_encoder.inverse_transform(unlabeled_y_pred_labels)

# 结合新数据进行重新训练
combined_data = pd.concat([train_data, unlabeled_data], ignore_index=True)
X_combined = vectorizer.fit_transform(combined_data['city'])
y_combined = label_encoder.fit_transform(combined_data['standard_city'])

# 重新训练模型
model.fit(X_combined, y_combined)