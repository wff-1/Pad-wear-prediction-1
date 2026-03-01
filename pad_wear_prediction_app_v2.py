# pad_wear_prediction_app_v2.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# ====================== 新增：背景色+字体大小样式配置 ======================
def set_page_style():
    st.markdown(
        """
        <style>
        /* 网页背景色（浅天蓝色渐变） */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #e6f7ff, #f0f8ff);
        }

        /* 1. 一级标题（大标题）字体大小 */
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 32px;  /* 可调整，默认约28px */
            font-weight: bold;  /* 加粗（可选） */
        }

        /* 2. 二级标题（子标题）字体大小 */
        h2, .stSubheader {
            font-size: 24px;  /* 默认约22px */
            color: #34495e;
        }

        /* 3. 普通文本（说明、提示、按钮文字）字体大小 */
        .stMarkdown, .stText, .stButton>button, .stSelectbox, .stNumberInput {
            font-size: 16px;  /* 默认约14px，调大更易读 */
        }

        /* 4. 预测结果（metric指标）字体大小 */
        .stMetric label {
            font-size: 18px;  /* 指标标签大小 */
        }
        .stMetric value {
            font-size: 28px;  /* 指标数值大小（核心结果） */
        }
        .stMetric delta {
            font-size: 14px;  /* 误差提示大小 */
        }

        /* 5. 展开栏/备注文字大小 */
        .stExpander, .stCaption {
            font-size: 14px;  /* 次要文字稍小 */
        }

        /* 按钮样式优化（配合字体） */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ====================== 页面基础设置 ======================
st.set_page_config(
    page_title="衬垫磨损量预测工具（数字版）",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 调用样式函数（必须放在最前面）
set_page_style()

st.title("🔧 模塑型自润滑关节轴承衬垫磨损量预测模型")
st.markdown("### （该预测为在275 MPa载荷、室温环境、自润滑关节轴承摆动25000次时衬垫的磨损量）")
st.divider()

# ====================== 1. 模型初始化（适配数字填料含量） ======================
@st.cache_resource  # 缓存模型，避免重复训练
def init_model():
    """训练并缓存预测模型，适配数字型润滑填料含量"""
    # 1. 构建实验数据集（数字型填料含量）
    data = {
        "润滑填料含量(%)": [40, 40, 40, 40, 35, 35, 35, 35],
        "结构尺寸(cm)": [25.4, 47, 25.4, 25.4, 25.4, 47, 25.4, 25.4],
        "成型时间(h)": [12, 12, 4, 12, 12, 12, 4, 12],
        "工况频率(Hz)": [0.5, 0.5, 0.5, 0.17, 0.5, 0.5, 0.5, 0.17],
        "磨损量(um)": [78, 99, 103, 55, 87, 118, 96, 67]
    }
    df = pd.DataFrame(data)

    # 2. 训练模型（直接用数字特征，无需编码）
    features = ["润滑填料含量(%)", "结构尺寸(cm)", "成型时间(h)", "工况频率(Hz)"]
    X = df[features]
    y = df["磨损量(um)"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, df

# 初始化模型和数据集
model, df = init_model()

# ====================== 2. 网页输入界面（适配数字填料含量） ======================
st.subheader("请输入预测参数")
col1, col2 = st.columns(2)

with col1:
    # 下拉选择：润滑填料含量（数字型，限定35/40）
    lubricant_content = st.selectbox(
        "材料制备——润滑填料含量 (%)",
        options=[35, 40],  # 直接显示数字
        help="选择衬垫使用的润滑填料含量（仅支持35%/40%）"
    )
    # 数字输入：结构尺寸
    structure_size = st.number_input(
        "结构设计——轴承外圈直径尺寸 (cm)",
        min_value=0.0,
        value=25.4,
        step=0.1,
        help="衬垫的核心结构尺寸，示例值：25.4、47"
    )

with col2:
    # 数字输入：成型工艺——固化时间
    molding_time = st.number_input(
        "成型工艺——固化时间 (h)",
        min_value=0.0,
        value=12.0,
        step=0.1,
        help="衬垫成型工艺的时间，示例值：4、12"
    )
    # 数字输入：工况频率
    working_frequency = st.number_input(
        "工况——测试频率 (Hz)",
        min_value=0.0,
        value=0.5,
        step=0.01,
        help="使用工况的频率，示例值：0.17、0.5"
    )

# ====================== 3. 预测逻辑与结果展示 ======================
st.divider()
if st.button("🚀 点击预测磨损量", type="primary"):
    # 构造输入特征（直接用数字，无需编码）
    input_features = np.array([[lubricant_content, structure_size, molding_time, working_frequency]])
    # 执行预测
    predicted_wear = model.predict(input_features)[0]
    predicted_wear = round(predicted_wear, 2)

    # 美化展示结果
    st.success("✅ 预测成功！")
    st.metric(
        label="衬垫磨损量预测值",
        value=f"{predicted_wear} μm",
        delta="参考误差±15 μm（基于8组实验数据）"
    )

# ====================== 4. 辅助信息 ======================
with st.expander("📊 查看实验原始数据（点击展开）"):
    st.dataframe(df, use_container_width=True)

st.divider()
st.caption("⚠️ 说明：本工具基于实测实验数据训练，预测结果仅作工程参考，实际磨损量以实测为准。")