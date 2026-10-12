import pandas as pd
import streamlit as st

from rf_predict import get_model

st.set_page_config(page_title="Severe AKI 预测", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .result-card {
        background: linear-gradient(135deg, #f5f9ff 0%, #eef4fb 100%);
        border: 1px solid #d6e4f5;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 16px;
    }
    .prob-number {font-size: 56px; font-weight: 700; color: #1a56db; line-height: 1.1;}
    .prob-label {font-size: 15px; color: #4b5563; margin-top: 4px;}
    .risk-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-size: 14px; font-weight: 600; margin-top: 10px;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Severe AKI 预测")
st.caption("基于随机森林模型的重症急性肾损伤风险预测 — 请在左侧输入患者指标")

model = get_model()


def section(title):
    st.markdown(f"**{title}**")


with st.sidebar:
    st.header("📝 患者指标")

    section("基础疾病")
    renal_disease = st.radio("Renal disease（肾脏疾病）", ["0", "1"], index=0, horizontal=True)
    diabetes = st.radio("Diabetes（糖尿病）", ["0", "1"], index=0, horizontal=True)
    chf = st.radio("CHF（充血性心力衰竭）", ["0", "1"], index=0, horizontal=True)
    cci = st.slider("CCI（查尔森合并症指数）", 0, 16, 3)

    section("疾病严重度评分")
    sofa = st.slider("SOFA 评分", 2, 19, 18)
    apsiii = st.slider("APSIII 评分", 6, 101, 43)
    gcs = st.slider("GCS 评分", 3, 12, 3)

    section("实验室指标")
    creatinine = st.slider("Creatinine 肌酐 (mg/dL)", 0.1, 11.0, 1.7, step=0.1)
    urineoutput = st.slider("Urine output 尿量 (mL)", 4, 7975, 305)
    aptt = st.slider("APTT (s)", 15.9, 96.7, 62.7, step=0.1)
    bun = st.slider("BUN 尿素氮 (mg/dL)", 3.5, 142.5, 53.3, step=0.1)
    chloride = st.slider("Chloride 氯 (mmol/L)", 83.3, 128.0, 110.0, step=0.1)
    bicarbonate = st.slider("Bicarbonate 碳酸氢盐 (mmol/L)", 12.2, 34.7, 19.0, step=0.1)
    rbc = st.slider("RBC 红细胞 (×10¹²/L)", 2.0, 5.9, 3.0, step=0.1)
    rdw = st.slider("RDW 红细胞分布宽度 (%)", 11.6, 89.0, 55.8, step=0.1)

row = {"Renal_disease": float(renal_disease), "Diabetes": float(diabetes),
       "CHF": float(chf), "CCI": cci, "SOFA": sofa, "APSIII": apsiii,
       "GCS": gcs, "Creatinine": creatinine, "Urineoutput": urineoutput,
       "APTT": aptt, "BUN": bun, "Chloride": chloride,
       "Bicarbonate": bicarbonate, "RBC": rbc, "RDW": rdw}
values = [row[f] for f in model.features]  # 按模型训练时的特征顺序对齐

probs = model.predict_proba([values])[0]
# 与原 Shiny 一致：显示 predict(type="prob") 的第一列（P(类别 i)）
main_prob = float(probs[0])

if main_prob >= 0.7:
    risk_color, risk_text = "#fee2e2", "高风险"
elif main_prob >= 0.4:
    risk_color, risk_text = "#fef3c7", "中风险"
else:
    risk_color, risk_text = "#dcfce7", "低风险"

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(f"""
    <div class="result-card">
        <div class="prob-number">{main_prob:.3f}</div>
        <div class="prob-label">Severe AKI 发生概率</div>
        <span class="risk-badge" style="background:{risk_color};">{risk_text}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("高风险阈值", "≥ 0.70")
    c2.metric("中风险阈值", "0.40 – 0.70")
    c3.metric("低风险阈值", "< 0.40")

with right:
    st.subheader("📋 输入摘要")
    display = pd.DataFrame({
        "指标": ["Renal disease", "Diabetes", "CHF", "CCI", "SOFA", "APSIII", "GCS",
                "Creatinine", "Urine output", "APTT", "BUN", "Chloride",
                "Bicarbonate", "RBC", "RDW", "Prob"],
        "数值": [renal_disease, diabetes, chf, cci, sofa, apsiii, gcs,
               creatinine, urineoutput, aptt, bun, chloride,
               bicarbonate, rbc, rdw, round(main_prob, 3)],
    })
    st.dataframe(display, hide_index=True, use_container_width=True, height=590)

st.divider()
st.caption("⚠️ 本工具仅供临床参考，不能替代专业医师判断。主概率（Prob）对应原 R/Shiny 模型的输出列。")
