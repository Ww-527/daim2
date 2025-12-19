import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. 读取书本对应的CSV数据（编码gbk，列名匹配书本逻辑）
df = pd.read_csv('insurance-chinese.csv', encoding='gbk')

# 2. 按书本代码的特征编码逻辑做独热编码（生成和书本变量名一致的列）
# 注意：列名要匹配书本代码中的变量名（sex_female、smoke_no、region_northeast等）
df_encoded = pd.get_dummies(
    df,
    columns=['性别', '是否吸烟', '区域'],
    # 编码后列名强制匹配书本代码的变量名（关键！）
    prefix=['sex', 'smoke', 'region'],
    prefix_sep='_',
    drop_first=False
)

# 手动调整列名（确保和书本代码的变量名100%匹配）
df_encoded.rename(columns={
    'sex_女性': 'sex_female',
    'sex_男性': 'sex_male',
    'smoke_否': 'smoke_no',
    'smoke_是': 'smoke_yes',
    'region_东北部': 'region_northeast',
    'region_东南部': 'region_southeast',
    'region_西北部': 'region_northwest',
    'region_西南部': 'region_southwest'
}, inplace=True)

# 3. 定义特征列（和书本代码的format_data顺序完全一致）
feature_cols = [
    '年龄', 'BMI', '子女数量',
    'sex_female', 'sex_male',
    'smoke_no', 'smoke_yes',
    'region_northeast', 'region_southeast', 'region_northwest', 'region_southwest'
]

# 4. 特征和目标变量（医疗费用是目标列）
X = df_encoded[feature_cols]
y = df_encoded['医疗费用']

# 5. 训练模型（随机森林，和书本描述一致）
rfr_model = RandomForestRegressor(random_state=42)
rfr_model.fit(X, y)

# 6. 保存模型（生成rfr_model.pkl，供书本代码调用）
with open('rfr_model.pkl', 'wb') as f:
    pickle.dump(rfr_model, f)

print("✅ 模型训练完成！生成rfr_model.pkl文件")
print("🔍 模型特征列：", rfr_model.feature_names_in_)  # 核对列名是否匹配
