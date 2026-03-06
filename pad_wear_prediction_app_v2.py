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

# 静态可视化库导入
import matplotlib.pyplot as plt
# 替换原来的字体设置，优先使用服务器自带的无衬线字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 保留负号显示

# 强制设置Python编码（解决本地中文乱码核心）
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ====================== 访问人数统计（简易本地版） ======================
if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 1
else:
    st.session_state.visit_count += 1

# ====================== 样式设置（响应式适配 + 无乱码） ======================
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

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="模塑型自润滑关节轴承衬垫磨损量预测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
set_page_style()

# ====================== 右上角实时时间 + 访问人数 ======================
now = datetime.now()
if datetime.utcnow().hour == now.hour:
    now = now + timedelta(hours=8)
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

visit_num = st.session_state.visit_count
st.markdown(f'''
<div class="top-right-info">
    当前时间：{now_str} &nbsp;&nbsp; 累计访问人数：{visit_num}
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

# ====================== 提取查询用参数 ======================
content_options = sorted(original_df["润滑填料含量(%)"].unique())
size_options = sorted(original_df["轴承外圈直径尺寸(cm)"].unique())
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

# ====================== 主页面交互 ======================
# 头部卡片
st.markdown('''
<div class="header-card">
    <div class="main-title">
        🤖 模塑型自润滑关节轴承衬垫磨损量预测模型
    </div>
    <div class="pred-desc">（该预测为在275 MPa载荷、室温环境、自润滑关节轴承摆动25000次时衬垫的磨损量）</div>
    <div class="model-metric">📊 模型性能：R² = {r2_val} | 平均绝对误差 MAE = {mae_val} μm | 均方误差 MSE = {mse_val} μm²</div>
</div>
'''.format(r2_val=r2_val, mae_val=mae_val, mse_val=mse_val), unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# 双列布局：查询 + 预测
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="func-card query"><div class="func-title">🔍 原始数据查询——查询实测数据</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        query_content = st.selectbox("🧪 材料制备——润滑填料含量 (%)", content_options, key="query_content")
        query_size = st.selectbox("⚙️ 结构设计——轴承外圈直径 (cm)", size_options, key="query_size")
    with col2:
        query_time_item = st.selectbox(
            "⏱️ 成型工艺——固化时间 (h)",
            time_options,
            format_func=lambda x: x[0],
            key="query_time"
        )
        query_time_value = query_time_item[1]
        query_freq = st.selectbox("🔄 测试工况——测试频率 (Hz)", freq_options, key="query_freq")
    
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
    st.markdown('<div class="func-card"><div class="func-title">📈 模型预测——预测衬垫磨损量</div></div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        lubricant_content = st.selectbox("🧪 材料制备——润滑填料含量 (%)", [35, 40], key="pred_content")
        structure_size = st.number_input("⚙️ 结构设计——轴承外圈直径 (cm)", value=25.4, step=0.1, key="pred_size")
    with col4:
        molding_time = st.number_input("⏱️ 成型工艺——固化时间 (h)", value=12.0, step=1.0, key="pred_time")
        working_frequency = st.number_input("🔄 测试工况——测试频率 (Hz)", value=0.5, step=0.01, key="pred_freq")
    
    if st.button("🚀 点击预测磨损量", type="primary", key="pred_btn"):
        input_features = np.array([[lubricant_content, structure_size, molding_time, working_frequency]])
        predicted_wear = model.predict(input_features)[0]
        predicted_wear = round(predicted_wear, 2)
        st.success("✅ 预测完成！")
        st.markdown(f'<div class="pred-result-label">📈 衬垫磨损量预测值</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-result-value">{predicted_wear} μm</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-result-delta">参考误差 ±{mae_val} μm</div>', unsafe_allow_html=True)

# ====================== 数据可视化模块（按钮控制显示/隐藏） ======================
# 初始化状态：默认不显示
if 'show_viz_section' not in st.session_state:
    st.session_state['show_viz_section'] = False

# 切换按钮
st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
if st.button(
    "📊 数据可视化分析（点击展开/收起）",
    type="secondary",
    use_container_width=True
):
    st.session_state['show_viz_section'] = not st.session_state['show_viz_section']

# 显示可视化内容
if st.session_state['show_viz_section']:
    st.markdown('<div class="func-card"><div class="func-title">📊 静态数据可视化分析</div></div>', unsafe_allow_html=True)
    
    # 模型与网站介绍 - 修改后的样式
    st.markdown("""
   <div style="background: linear-gradient(135deg, #F8F4E9 0%, #F0E6D2 100%); 
            padding: 7px 12px; 
            border-radius: 6px; /* 复古小圆角，替代现代大圆角 */
            margin-bottom: 10px; 
            border-left: 3px solid #B85450; /* 复古砖红色侧边线，替代亮红 */
            border: 1px solid #D9C7A7; /* 复古米色边框 */
            box-shadow: 0 2px 6px rgba(150, 120, 80, 0.12); /* 暖调复古阴影 */
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1IiBoZWlnaHQ9IjUiPgo8cmVjdCB3aWR0aD0iNSIgaGVpZ2h0PSI1IiBmaWxsPSIjRjhGNEU5Ii8+CjxyZWN0IHdpZHRoPSIxIiBoZWlnaHQ9IjEiIGZpbGw9IiNlOGU0ZGQiLz4KPC9zdmc+'); /* 复古纸张纹理 */
            background-size: 15px 15px;">
    <p style="font-size: 11px; 
              color: #5A4A38; /* 复古深棕文字，替代纯黑 */
              line-height: 1.4; 
              font-weight: normal; 
              margin: 0;
              font-family: 'SimSun', 'Microsoft YaHei', serif; /* 宋体+微软雅黑，增强复古感 */
              letter-spacing: 0.3px; /* 轻微字间距，复古排版特征 */">
        模型说明：本网站搭载的随机森林预测模型，核心基于聚酰胺酸酯树脂基自润滑关节轴承衬垫的实验数据训练而成。
        <br>重要提示：材料配方中树脂的化学结构是衬垫磨损性能的决定性因素，若将本模型应用于其他种类树脂材料的磨损量预测，结果可能存在显著偏差，请谨慎使用。
    </p>
</div>
    """, unsafe_allow_html=True)
    
    # 准备数据
    df_viz = load_original_data()
    df_viz["固化时间_str"] = df_viz["固化时间(h)"].apply(lambda x: f"{x}.00 h")
    X_all = df_viz[["润滑填料含量(%)", "轴承外圈直径尺寸(cm)", "固化时间(h)", "测试频率(Hz)"]]
    df_viz["预测磨损量"] = model.predict(X_all)
    
    # --- 图1：磨损量 vs 测试频率 ---
    from matplotlib.lines import Line2D
    fig1, ax1 = plt.subplots(figsize=(6.5, 4), dpi=100)
    colors = {35: '#FF7F0E', 40: '#14558F'}
    edge_colors = {35: '#D65F00', 40: '#14558F'}
    
    for content in df_viz["润滑填料含量(%)"].unique():
        df_sub = df_viz[df_viz["润滑填料含量(%)"] == content]
        point_size = (df_sub["轴承外圈直径尺寸(cm)"] - 20) * 25
        for x, y, s in zip(df_sub["测试频率(Hz)"], df_sub["磨损量(um)"], point_size):
            ax1.scatter(x + 0.002, y - 0.8, s=s, color=edge_colors[content], alpha=0.3, zorder=1)
            ax1.scatter(x, y, s=s, color=colors[content], edgecolor=edge_colors[content], linewidth=1.2, alpha=0.9, zorder=2)
    
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    ax1.set_xlabel("测试频率 (Hz)", fontsize=10, labelpad=8)
    ax1.set_ylabel("磨损量 (μm)", fontsize=10, labelpad=8)
    ax1.set_title("1. 磨损量与测试频率关系图（点大小表示轴承外圈直径大小）", fontsize=12, fontweight="bold", pad=12)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='填料含量 35%',
               markerfacecolor='#FF7F0E', markeredgecolor='#D65F00', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='填料含量 40%',
               markerfacecolor='#14558F', markeredgecolor='#14558F', markersize=8)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--', color='#999999', linewidth=0.6)
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(20, 250)
    ax1.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown("<p style='font-size:10px;color:#555;text-align:center;'>该图显示磨损量随测试频率升高呈上升趋势；35%填料含量的样本磨损量整体高于40%，且轴承直径越大，数据点尺寸越大，直观呈现了多变量的关联。</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 图2：不同固化时间+轴承直径的磨损量（颜色分组调整） ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4), dpi=100)
    df_viz["分组"] = df_viz["固化时间_str"] + " | " + df_viz["轴承外圈直径尺寸(cm)"].astype(str) + "cm"
    box_plot = df_viz.boxplot(
        column="磨损量(um)",
        by="分组",
        ax=ax2,
        patch_artist=True,
        return_type='dict',
        medianprops={'color': '#2ca02c', 'linewidth': 1.8},  # 默认绿色
        whiskerprops={'color': '#1f77b4', 'linewidth': 1.0},
        capprops={'color': '#1f77b4', 'linewidth': 1.0},
        flierprops={'marker': 'o', 'markerfacecolor': '#FF6B6B', 'markeredgecolor': '#333', 'markersize': 4}
    )
    
    # 前两个中位数线设为过渡色（钢蓝色 #4682B4）
    median_lines = box_plot['磨损量(um)']['medians']
    if len(median_lines) >= 2:
        median_lines[0].set_color('#4682B4')  # 第一个中位数线
        median_lines[1].set_color('#4682B4')  # 第二个中位数线
    
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    # 核心修改：第1、2个箱体为浅蓝 #87CEEB；第3、4个箱体为浅绿 #90EE90
    fill_colors = ['#87CEEB', '#87CEEB', '#90EE90', '#90EE90']
    for i, box in enumerate(box_plot['磨损量(um)']['boxes']):
        box.set_facecolor(fill_colors[i % len(fill_colors)])
        box.set_edgecolor('#333333')
        box.set_linewidth(0.8)
    
    group_names = df_viz["分组"].unique()
    medians = df_viz.groupby("分组")["磨损量(um)"].median()
    for i, group in enumerate(group_names):
        median_val = medians[group]
        ax2.text(
            x=i+1,
            y=median_val,
            s=f'{median_val:.1f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#000000'
        )
    
    ax2.set_xlabel("固化时间 | 轴承外圈直径", fontsize=10, labelpad=8)
    ax2.set_ylabel("磨损量 (μm)", fontsize=10, labelpad=8)
    ax2.set_title("2. 不同固化时间与轴承直径的磨损量分布（标注中位数）", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(alpha=0.3, linestyle='--', color='#999999', linewidth=0.6)
    ax2.tick_params(axis='both', labelsize=8)
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("<p style='font-size:10px;color:#555;text-align:center;'>箱线图清晰展示了不同分组的磨损量分布与中位数水平；固化时间和轴承直径的组合显著影响磨损量，中位数标签可快速对比各组数据的集中趋势。</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 图3：预测值 vs 实测值 ---
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.5), dpi=100)
    for spine in ax3.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    df_plot = df_viz.reset_index()
    ax3.plot(df_plot["index"], df_plot["磨损量(um)"], color='#FF4B5C', linewidth=2.0, marker='o', markersize=4, label='实测值')
    ax3.plot(df_plot["index"], df_plot["预测磨损量"], color='#1E90FF', linewidth=2.0, linestyle='--', marker='s', markersize=4, label='预测值')
    
    ax3.set_xlabel("样本序号", fontsize=10, labelpad=8)
    ax3.set_ylabel("磨损量 (μm)", fontsize=10, labelpad=8)
    ax3.set_title("3. 磨损量实测值与模型预测值对比", fontsize=12, fontweight="bold", pad=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--', color='#999999', linewidth=0.6)
    ax3.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown("<p style='font-size:10px;color:#555;text-align:center;'>模型预测值与实测值的折线趋势高度一致，表明该模型能较好地拟合磨损量数据，可用于后续磨损量的预测分析。</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 图4：特征重要性 ---
    st.markdown("### 🧠 特征重要性分析")
    fig4, ax4 = plt.subplots(figsize=(6.5, 3.5), dpi=100)
    for spine in ax4.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    feature_names = ["润滑填料含量(%)", "轴承外圈直径(cm)", "固化时间(h)", "测试频率(Hz)"]
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', ['#64B5F6', '#1565C0'])
    colors_gradient = [cmap(i/len(sorted_importances)) for i in range(len(sorted_importances))]
    edge_colors = ['#0D47A1' for _ in range(len(sorted_importances))]
    
    bar_width = 0.6
    x_pos = range(len(sorted_features))
    
    ax4.bar(
        [x + 0.015 for x in x_pos],
        [h - 0.004 for h in sorted_importances],
        width=bar_width,
        color='#0A2E70',
        alpha=0.3,
        zorder=1
    )
    
    bars = ax4.bar(
        x_pos,
        sorted_importances,
        width=bar_width,
        color=colors_gradient,
        edgecolor=edge_colors,
        linewidth=1.2,
        alpha=0.9,
        zorder=2
    )
    
    y_max = max(sorted_importances) * 1.15
    ax4.set_ylim(0, y_max)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.002,
            f'{height:.3f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color='#000000',
            zorder=3
        )
    
    ax4.set_ylabel("特征重要性", fontsize=10, labelpad=8)
    ax4.set_title("4. 各参数对磨损量的影响程度排序", fontsize=12, fontweight="bold", pad=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sorted_features, rotation=15, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', color='#999999', linewidth=0.6)
    ax4.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown("<p style='font-size:10px;color:#555;text-align:center;'>该图按重要性排序展示了各参数对磨损量的影响，数值标签清晰呈现具体重要性得分，可明确后续优化的核心参数方向。</p>", unsafe_allow_html=True)

# 底部备注 - 与卡片间距设为15px
st.markdown('<div class="bottom-note">', unsafe_allow_html=True)
st.divider()
st.caption("⚠️ 原始数据查询结果为实验室实测值；模型预测结果仅供科研与工程参考。")
st.markdown('</div>', unsafe_allow_html=True)