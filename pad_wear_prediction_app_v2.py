# -*- coding: utf-8 -*-
import sys
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 强制设置Python编码（解决本地中文乱码核心）
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ====================== 访问人数统计（简易本地版） ======================
if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 1
else:
    st.session_state.visit_count += 1

# ====================== 样式设置（新增响应式适配） ======================
def set_page_style():
    st.markdown(
        '''
        <style>
        /* 全局基础样式 - 中文微软雅黑，西文Times New Roman */
        * {
            font-size: 16px !important;
            font-family: "Microsoft YaHei", "Times New Roman", serif !important;
        }
        /* 背景改为纯白色 */
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
        }
        /* 右上角时间与访问人数 */
        .top-right-info {
            position: absolute;
            top: 15px;
            right: 25px;
            font-size: 14px !important;
            color: #6B7280 !important;
            font-weight: 500;
            z-index: 9999;
            /* 手机端调整右上角信息位置 */
            @media (max-width: 768px) {
                position: relative;
                top: 0;
                right: 0;
                display: block;
                text-align: center;
                margin-bottom: 15px;
                padding: 5px;
            }
        }
        /* 取消页面最大宽度限制，减少留白 */
        .main > .block-container {
            max-width: 95% !important;
            padding-left: 2% !important;
            padding-right: 2% !important;
        }
        /* 主标题卡片高级美化 */
        .header-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
            border-radius: 20px !important;
            padding: 35px 30px !important;
            margin: 20px auto !important;
            box-shadow: 
                0 8px 24px rgba(149, 157, 165, 0.08),
                0 1px 3px rgba(0, 0, 0, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
            max-width: 95% !important;
            border: 1px solid #e2e8f0 !important;
            position: relative;
            overflow: hidden;
            /* 手机端适配标题卡片 */
            @media (max-width: 768px) {
                padding: 20px 15px !important;
                margin: 10px auto !important;
            }
        }
        /* 主标题卡片动态顶边装饰 */
        .header-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #3498db, #27ae60, #3498db) !important;
            background-size: 200% 100%;
            animation: gradientMove 8s linear infinite;
        }
        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }
        /* 自定义标题样式 */
        .main-title {
            font-size: 40px !important;
            color: #2c3e50 !important;
            text-align: center !important;
            font-weight: bold !important;
            margin: 0 0 20px 0 !important;
            line-height: 1.3 !important;
            /* 手机端标题字号适配 */
            @media (max-width: 768px) {
                font-size: 28px !important;
            }
        }
        /* 模型性能 */
        .model-metric {
            font-size: 15px !important;
            color: #2980b9 !important;
            text-align: center !important;
            margin: 10px 0 !important;
            /* 手机端适配 */
            @media (max-width: 768px) {
                font-size: 14px !important;
            }
        }
        /* 测试条件说明 */
        .pred-desc {
            font-size: 20px !important;
            color: #34495e !important;
            text-align: center !important;
            font-weight: bold !important;
            margin-bottom: 20px !important;
            /* 手机端适配 */
            @media (max-width: 768px) {
                font-size: 16px !important;
            }
        }
        /* 渐变分隔线 */
        .gradient-divider {
            height: 3px !important;
            background: linear-gradient(to right, #3498db, #27ae60) !important;
            border: none !important;
            border-radius: 3px !important;
            margin: 30px auto !important;
            max-width: 95% !important;
        }
        /* 功能卡片：保持原样 + 响应式适配 */
        .func-card {
            background: linear-gradient(135deg, #f5f9ff 0%, #e8f4f8 100%) !important;
            border-radius: 12px !important;
            padding: 15px 20px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
            border-left: 4px solid #3498db !important;
            /* 手机端功能卡片适配 */
            @media (max-width: 768px) {
                padding: 15px 10px !important;
                margin-bottom: 15px !important;
            }
        }
        .func-card.query {
            border-left: 4px solid #27ae60 !important;
        }
        .func-title {
            font-size: 26px !important;
            color: #2c3e50 !important;
            font-weight: bold !important;
            margin: 0 !important;
            text-align: left !important;
            /* 手机端功能标题适配 */
            @media (max-width: 768px) {
                font-size: 20px !important;
                text-align: center !important;
            }
        }
        /* 输入框样式：核心修复手机端居中问题 */
        [data-testid="stSelectbox"], [data-testid="stNumberInput"] {
            max-width: 280px !important;
            margin: 0 auto 10px auto !important; /* 改为auto居中 */
            display: block !important; /* 块级元素确保居中 */
            /* 手机端输入框宽度适配 */
            @media (max-width: 768px) {
                max-width: 90% !important; /* 手机端占满宽度 */
                margin: 0 auto 10px auto !important; /* 强制居中 */
            }
        }
        label {
            font-size: 18px !important;
            font-weight: 500 !important;
            display: block !important;
            text-align: center !important; /* 标签也居中 */
            margin-bottom: 5px !important;
            /* 手机端标签适配 */
            @media (max-width: 768px) {
                font-size: 16px !important;
                text-align: center !important;
            }
        }
        /* 按钮样式：强制居中 */
        [data-testid="baseButton-primary"] {
            font-size: 18px !important;
            border-radius: 8px !important;
            padding: 0.6rem 2.2rem !important;
            border: none !important;
            max-width: 300px !important;
            margin: 10px auto !important; /* 改为auto居中 */
            display: block !important; /* 块级元素确保居中 */
            /* 手机端按钮适配 */
            @media (max-width: 768px) {
                max-width: 90% !important;
                margin: 15px auto !important;
                font-size: 16px !important;
            }
        }
        [data-testid="baseButton-primary"][key="pred_btn"] {
            background-color: #3498db !important;
            color: white !important;
        }
        [data-testid="baseButton-primary"][key="pred_btn"]:hover {
            background-color: #2980b9 !important;
        }
        [data-testid="baseButton-primary"][key="query_btn"] {
            background-color: #27ae60 !important;
            color: white !important;
        }
        [data-testid="baseButton-primary"][key="query_btn"]:hover {
            background-color: #219653 !important;
        }
        /* 结果样式：手机端适配 */
        .pred-result-value, .query-result-value {
            font-size: 40px !important;
            font-weight: bold !important;
            text-align: center !important; /* 结果也居中 */
            margin: 10px 0 !important;
            /* 手机端适配 */
            @media (max-width: 768px) {
                font-size: 32px !important;
                text-align: center !important;
            }
        }
        .pred-result-value {
            color: #27ae60 !important;
        }
        .query-result-value {
            color: #e74c3c !important;
        }
        .pred-result-label, .query-result-label {
            font-size: 20px !important;
            color: #2c3e50 !important;
            text-align: center !important; /* 标签居中 */
            /* 手机端适配 */
            @media (max-width: 768px) {
                font-size: 18px !important;
                text-align: center !important;
            }
        }
        .pred-result-delta, .query-result-desc {
            font-size: 18px !important;
            color: #7f8c8d !important;
            text-align: center !important; /* 描述居中 */
            /* 手机端适配 */
            @media (max-width: 768px) {
                font-size: 16px !important;
                text-align: center !important;
            }
        }
        /* 底部备注与卡片间距设为15px */
        .bottom-note {
            margin-top: 15px !important;
            /* 手机端适配 */
            @media (max-width: 768px) {
                margin-top: 10px !important;
                padding: 0 5px !important;
            }
        }
        /* 核心修复：手机端双列布局改为单列 */
        @media (max-width: 768px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: none !important;
                margin: 0 !important;
            }
            /* 取消手机端列间距 */
            .stColumns {
                gap: 0 !important;
            }
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

# ====================== 页面配置（替换为科幻轴承图标） ======================
st.set_page_config(
    page_title="模塑型自润滑关节轴承衬垫磨损量预测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
set_page_style()

# ====================== 右上角显示：实时时间 + 访问人数（修复时区） ======================
# 核心修改：强制使用北京时间（UTC+8），解决时区差8小时问题
now = datetime.now()  # 获取本地时间
# 兼容服务器时区问题：如果检测到是UTC时区，自动+8小时
if datetime.utcnow().hour == now.hour:  # 说明当前是UTC时区
    now = now + timedelta(hours=8)
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

visit_num = st.session_state.visit_count
st.markdown(f'''
<div class="top-right-info">
    当前时间：{now_str} &nbsp;&nbsp; 访问人数：{visit_num}
</div>
''', unsafe_allow_html=True)

# ====================== 加载原始数据 ======================
def load_original_data():
    data = {
        "润滑填料含量(%)": [40,40,40,40,40,40,40,40,40,40,35,35,35,35,35,35,35,35,35,35],
        "轴承外圈直径尺寸(cm)": [25.4,47,25.4,25.4,25.4,25.4,25.4,47,47,47,25.4,47,25.4,25.4,25.4,25.4,25.4,47,47,47],
        "固化时间(h)": [12,12,4,4,12,12,12,4,12,12,4,12,4,12,12,12,12,4,12,12],
        "测试频率(Hz)": [0.5,0.5,0.5,0.17,0.17,0.33,0.83,0.5,0.33,0.17,0.17,0.5,0.5,0.17,0.33,0.5,0.83,0.5,0.33,0.17],
        "磨损量(um)": [68.8,139.12,102.5,49.98,33.6,43.93,95.40,147.22,115.28,102.74,82.79,173.31,109.23,41.57,44.98,86.05,122.05,233.27,159.15,133.2]
    }
    df = pd.DataFrame(data)
    return df

original_df = load_original_data()

# ====================== 提取原始数据唯一值（查询用） ======================
content_options = sorted(original_df["润滑填料含量(%)"].unique())
size_options = sorted(original_df["轴承外圈直径尺寸(cm)"].unique())
# 固化时间显示两位小数
time_options = [("4.00", 4.0), ("12.00", 12.0)]
freq_options = sorted(original_df["测试频率(Hz)"].unique())

# ====================== 模型训练 ======================
@st.cache_resource
def train_model():
    X = original_df[["润滑填料含量(%)", "轴承外圈直径尺寸(cm)", "固化时间(h)", "测试频率(Hz)"]]
    y = original_df["磨损量(um)"]
    
    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X, y)
    
    loo = LeaveOneOut()
    y_pred_cv = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        m = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        m.fit(X_train, y_train)
        y_pred_cv.append(m.predict(X_test)[0])
    
    r2 = round(r2_score(y, y_pred_cv), 3)
    mae = round(mean_absolute_error(y, y_pred_cv), 2)
    mse = round(mean_squared_error(y, y_pred_cv), 2)
    return model, r2, mae, mse

model, r2_val, mae_val, mse_val = train_model()

# ====================== 数据查询函数 ======================
def query_wear_data(query_content, query_size, query_time, query_freq):
    mask = (
        (original_df["润滑填料含量(%)"] == query_content) &
        (original_df["轴承外圈直径尺寸(cm)"] == query_size) &
        (original_df["固化时间(h)"] == query_time) &
        (original_df["测试频率(Hz)"] == query_freq)
    )
    matched_data = original_df[mask]
    
    if not matched_data.empty:
        return matched_data["磨损量(um)"].values[0]
    else:
        return None

# ====================== 页面交互（标题嵌入科幻轴承图标） ======================
# 头部卡片（已美化 + 标题嵌入轴承图标）
st.markdown('''
<div class="header-card">
    <div class="main-title">
        🤖 模塑型自润滑关节轴承衬垫磨损量预测模型
    </div>
    <div class="pred-desc">（该预测为在275 MPa载荷、室温环境、自润滑关节轴承摆动25000次时衬垫的磨损量）</div>
    <div class="model-metric">📊 模型性能：R² = {r2_val} | 平均绝对误差 MAE = {mae_val} μm | 均方误差 MSE = {mse_val} μm²</div>
</div>
'''.format(r2_val=r2_val, mae_val=mae_val, mse_val=mse_val), unsafe_allow_html=True)

# 渐变分隔线
st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# 双列并排布局
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # 1. 原始数据查询
    st.markdown('<div class="func-card query"><div class="func-title">🔍 原始数据查询——查询实测数据</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        query_content = st.selectbox("🧪 材料制备——润滑填料含量 (%)", content_options, key="query_content")
        query_size = st.selectbox("⚙️ 结构设计——轴承外圈直径 (cm)", size_options, key="query_size")
    with col2:
        # 固化时间下拉框（显示4.00/12.00）
        query_time_item = st.selectbox(
            "⏱️ 成型工艺——固化时间 (h)",
            time_options,
            format_func=lambda x: x[0],
            key="query_time"
        )
        query_time_value = query_time_item[1]
        query_freq = st.selectbox("🔄 测试工况——测试频率 (Hz)", freq_options, key="query_freq")

    # 查询按钮
    if st.button("🔍 点击查询原始数据磨损量", type="primary", key="query_btn"):
        wear_result = query_wear_data(query_content, query_size, query_time_value, query_freq)
        
        if wear_result is not None:
            st.success("✅ 数据查询成功！")
            st.markdown(f'<div class="query-result-label">🔍 原始实验数据磨损量</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="query-result-value">{wear_result} μm</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="query-result-desc">该结果为实验室实测数据，无误差</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ 未查询到匹配的原始实验数据！")

with col_right:
    # 2. 模型预测 - 保持原样
    st.markdown('<div class="func-card"><div class="func-title">📈 模型预测——预测衬垫磨损量</div></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        lubricant_content = st.selectbox("🧪 材料制备——润滑填料含量 (%)", [35, 40], key="pred_content")
        structure_size = st.number_input("⚙️ 结构设计——轴承外圈直径 (cm)", value=25.4, step=0.1, key="pred_size")
    with col4:
        molding_time = st.number_input("⏱️ 成型工艺——固化时间 (h)", value=12.0, step=1.0, key="pred_time")
        working_frequency = st.number_input("🔄 测试工况——测试频率 (Hz)", value=0.5, step=0.01, key="pred_freq")

    # 预测按钮
    if st.button("🚀 点击预测磨损量", type="primary", key="pred_btn"):
        input_features = np.array([[lubricant_content, structure_size, molding_time, working_frequency]])
        predicted_wear = model.predict(input_features)[0]
        predicted_wear = round(predicted_wear, 2)
        st.success("✅ 预测完成！")
        
        st.markdown(f'<div class="pred-result-label">📈 衬垫磨损量预测值</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-result-value">{predicted_wear} μm</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-result-delta">参考误差 ±{mae_val} μm</div>', unsafe_allow_html=True)

# 底部备注 - 与卡片间距设为15px
st.markdown('<div class="bottom-note">', unsafe_allow_html=True)
st.divider()
st.caption("⚠️ 原始数据查询结果为实验室实测值；模型预测结果仅供科研与工程参考。")
st.markdown('</div>', unsafe_allow_html=True) 