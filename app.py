import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import joblib
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

st.set_page_config(page_title="交通事故数据可视化", layout="wide")

ROAD_FEATURES = [
    "Amenity",
    "Crossing",
    "Junction",
    "Stop",
    "Traffic_Signal",
    "Bump",
    "Give_Way",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Traffic_Calming",
]



class XGBSeverityClassifier(BaseEstimator, ClassifierMixin):
    """
    兼容主项目中保存的 XGBoost 严重程度预测模型。

    训练阶段模型被序列化为 src.models.training.XGBSeverityClassifier。
    该兼容类只用于让 joblib 正常反序列化同学提供的真实模型文件；
    不会重新训练模型，也不会修改模型内部参数。
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier

        y_encoded = np.asarray(y) - 1
        self.classes_ = np.array([1, 2, 3, 4])
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_encoded, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X):
        return self.model_.predict(X) + 1

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


# 让 joblib.load 能找到原模型文件中记录的 src.models.training.XGBSeverityClassifier。
# 这里只注册同名兼容类，不改变模型文件中的训练参数和树结构。
_compat_src_module = types.ModuleType("src")
_compat_src_module.__path__ = [str(APP_DIR / "src"), str(PROJECT_ROOT / "src")]
_compat_models_module = types.ModuleType("src.models")
_compat_models_module.__path__ = [str(APP_DIR / "src" / "models"), str(PROJECT_ROOT / "src" / "models")]
_compat_training_module = types.ModuleType("src.models.training")
_compat_training_module.XGBSeverityClassifier = XGBSeverityClassifier

sys.modules.setdefault("src", _compat_src_module)
sys.modules.setdefault("src.models", _compat_models_module)
sys.modules["src.models.training"] = _compat_training_module


# 预测模块固定使用报告对应的 XGBoost 模型，不再回退到其他模型。
XGBOOST_MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "XGBoost_random_search_macro_f1.joblib",
    APP_DIR / "models" / "XGBoost_random_search_macro_f1.joblib",
]

# ── 读取数据 ──
@st.cache_data
def load_data():
    df = pd.read_csv("src/data/accidents_cleaned_sample.csv")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    # 单位转换 ——> 国际单位制（SI）
    df["Temperature(C)"] = round((df["Temperature(F)"] - 32) * 5 / 9, 1)
    df["Visibility(km)"] = round(df["Visibility(mi)"] * 1.60934, 2)
    df["Wind_Speed(kmh)"] = round(df["Wind_Speed(mph)"] * 1.60934, 1)
    df["Precipitation(mm)"] = round(df["Precipitation(in)"] * 25.4, 1)
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month
    df["Year"] = df["Start_Time"].dt.year
    df["Weekday"] = df["Start_Time"].dt.day_name()
    return df

df = load_data()


@st.cache_resource
def load_predictor():
    """固定加载报告对应的 XGBoost 严重程度预测模型。"""
    for model_path in XGBOOST_MODEL_CANDIDATES:
        if model_path.exists():
            return joblib.load(model_path), model_path
    return None, None


def most_common_value(data: pd.DataFrame, column: str, fallback: str = "Unknown") -> str:
    values = data[column].dropna() if column in data.columns else pd.Series(dtype=object)
    if values.empty:
        return fallback
    return str(values.mode().iloc[0])


def add_prediction_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Lat_Bin"] = (data["Start_Lat"] * 10).apply(lambda x: int(x // 1) / 10)
    data["Lng_Bin"] = (data["Start_Lng"] * 10).apply(lambda x: int(x // 1) / 10)
    data["Location_Bin"] = data["Lat_Bin"].round(1).astype(str) + "_" + data["Lng_Bin"].round(1).astype(str)
    data["LowVisibility"] = (data["Visibility(mi)"] < 2.0).astype(int)
    data["HasPrecipitation"] = (data["Precipitation(in)"] > 0).astype(int)
    data["BadWeatherJunction"] = ((data["Visibility(mi)"] < 2.0) & (data["Junction"] == 1)).astype(int)
    return data


predictor, predictor_path = load_predictor()

# ── 侧边栏 ──
st.sidebar.header("🔍 筛选条件")
states = ["全部"] + sorted(df["State"].unique().tolist())
sel_state = st.sidebar.selectbox("州 (State)", states)

severity_opts = ["全部"] + sorted(df["Severity"].dropna().unique().tolist())
sel_severity = st.sidebar.multiselect("严重程度 (Severity)", severity_opts, default=severity_opts)

year_range = range(int(df["Year"].min()), int(df["Year"].max()) + 1)
sel_years = st.sidebar.slider("年份范围", min_value=min(year_range), max_value=max(year_range),
                               value=(min(year_range), max(year_range)))

# 应用筛选
mask = (df["Year"] >= sel_years[0]) & (df["Year"] <= sel_years[1])
if sel_state != "全部":
    mask &= df["State"] == sel_state
if "全部" not in sel_severity:
    mask &= df["Severity"].isin(sel_severity)
dff = df[mask].copy()

st.sidebar.markdown(f"**筛选后数据：{len(dff):,} 条**")
st.sidebar.markdown("---")

# ── 标题 ──
st.title("🚗 美国交通事故数据可视化")
st.markdown(f"*数据范围：{sel_years[0]}–{sel_years[1]}　｜　共 {len(dff):,} 条事故记录*")
st.divider()

# ══════════════════════════════════════════════
# 仪表盘 KPI
# ══════════════════════════════════════════════
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if not dff.empty:
        avg_sev = dff["Severity"].mean()
        st.metric("平均严重程度", f"{avg_sev:.2f}")
with col2:
    if not dff.empty:
        avg_temp = dff["Temperature(C)"].mean()
        st.metric("平均温度 (°C)", f"{avg_temp:.1f}")
with col3:
    if not dff.empty:
        avg_vis = dff["Visibility(km)"].mean()
        st.metric("平均能见度 (km)", f"{avg_vis:.1f}")
with col4:
    if not dff.empty:
        avg_wind = dff["Wind_Speed(kmh)"].mean()
        st.metric("平均风速 (km/h)", f"{avg_wind:.1f}")
with col5:
    if not dff.empty:
        avg_precip = dff["Precipitation(mm)"].mean()
        st.metric("平均降水量 (mm)", f"{avg_precip:.2f}")

st.divider()

if dff.empty:
    st.warning("当前筛选条件下没有数据，请调整筛选条件。")
    st.stop()

# ══════════════════════════════════════════════
# 图表区域
# ══════════════════════════════════════════════

# ── 1. 严重程度分布 (柱状图) ──
with st.container():
    st.subheader("📊 1. 事故严重程度分布")
    col_a, col_b = st.columns(2)
    with col_a:
        sev_counts = dff["Severity"].value_counts().sort_index().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig1 = px.bar(sev_counts, x="Severity", y="Count",
                      color="Severity", color_continuous_scale="Reds",
                      text_auto=True, labels={"Count": "事故数量", "Severity": "严重程度 (1=最轻, 4=最重)"})
        fig1.update_traces(textposition="outside", width=0.4)
        fig1.update_layout(bargap=0.6)
        st.plotly_chart(fig1, width="stretch")
    with col_b:
        sev_pct = dff["Severity"].value_counts(normalize=True).sort_index() * 100
        fig1b = px.pie(values=sev_pct.values, names=sev_pct.index,
                       title="严重程度占比",
                       hole=0.4, color_discrete_sequence=px.colors.sequential.Reds_r)
        fig1b.update_traces(textinfo="percent+label")
        st.plotly_chart(fig1b, width="stretch")

st.divider()

# ── 2. 时间趋势 ──
with st.container():
    st.subheader("📈 2. 事故时间趋势")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        monthly = dff.set_index("Start_Time").resample("ME").size().reset_index(name="count")
        fig2a = px.line(monthly, x="Start_Time", y="count",
                        labels={"Start_Time": "月份", "count": "事故数量"},
                        title="月度趋势",
                        markers=True, line_shape="spline")
        fig2a.update_traces(fill="tozeroy", line_color="#e63946")
        st.plotly_chart(fig2a, width="stretch")
    with col_b:
        month_order = range(1, 13)
        month_all = dff["Month"].value_counts().reindex(month_order, fill_value=0).reset_index()
        month_all.columns = ["Month", "Count"]
        month_sev = dff.groupby(["Month", "Severity"]).size().reset_index(name="count")
        fig2b = go.Figure()
        fig2b.add_trace(go.Bar(x=month_all["Month"], y=month_all["Count"],
                                name="总计", width=0.5,
                                marker=dict(color="#a8d8ea", line=dict(color="#7eb8cc", width=1)),
                                opacity=0.75))
        for sev in sorted(dff["Severity"].dropna().unique()):
            sdf = month_sev[month_sev["Severity"] == sev].sort_values("Month")
            fig2b.add_trace(go.Scatter(x=sdf["Month"], y=sdf["count"],
                                        mode="lines+markers",
                                        name=f"Severity {sev}",
                                        line=dict(width=2)))
        fig2b.update_layout(title="1-12月分布", xaxis_title="月份", yaxis_title="事故数量")
        st.plotly_chart(fig2b, width="stretch")
    with col_c:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd_all = dff["Weekday"].value_counts().reindex(weekday_order).reset_index()
        wd_all.columns = ["Weekday", "Count"]
        wd_sev = dff.groupby(["Weekday", "Severity"]).size().reset_index(name="count")
        wd_sev["Weekday"] = pd.Categorical(wd_sev["Weekday"], categories=weekday_order, ordered=True)
        fig2c = go.Figure()
        fig2c.add_trace(go.Bar(x=wd_all["Weekday"], y=wd_all["Count"],
                                name="总计",
                                marker=dict(color="#f9d97e", line=dict(color="#d4b55a", width=1)),
                                opacity=0.75))
        for sev in sorted(dff["Severity"].dropna().unique()):
            sdf = wd_sev[wd_sev["Severity"] == sev].sort_values("Weekday")
            fig2c.add_trace(go.Scatter(x=sdf["Weekday"], y=sdf["count"],
                                        mode="lines+markers",
                                        name=f"Severity {sev}",
                                        line=dict(width=2)))
        fig2c.update_layout(title="一周各天分布", xaxis_title="星期", yaxis_title="事故数量")
        st.plotly_chart(fig2c, width="stretch")

    # 第二行：小时分布 + 季节分布
    col_d, col_e = st.columns(2)
    with col_d:
        hour_sev = dff.groupby(["Hour", "Severity"]).size().reset_index(name="count")
        fig2d = go.Figure()
        fig2d.add_trace(go.Bar(x=list(range(24)), y=dff.groupby("Hour").size().reindex(range(24), fill_value=0).values,
                                name="总计",
                                marker=dict(color="#b5d9a6", line=dict(color="#8fb87e", width=1)),
                                opacity=0.75))
        for sev in sorted(dff["Severity"].dropna().unique()):
            sdf = hour_sev[hour_sev["Severity"] == sev].sort_values("Hour")
            fig2d.add_trace(go.Scatter(x=sdf["Hour"], y=sdf["count"],
                                       mode="lines+markers",
                                       name=f"Severity {sev}",
                                       line=dict(width=2)))
        fig2d.update_layout(title="各小时事故分布", xaxis_title="小时 (24h)", yaxis_title="事故数量")
        st.plotly_chart(fig2d, width="stretch")
    with col_e:
        season_map = {12: "冬", 1: "冬", 2: "冬",
                      3: "春", 4: "春", 5: "春",
                      6: "夏", 7: "夏", 8: "夏",
                      9: "秋", 10: "秋", 11: "秋"}
        dff["季节"] = dff["Month"].map(season_map)
        season_order = ["春", "夏", "秋", "冬"]
        season_all = dff["季节"].value_counts().reindex(season_order).reset_index()
        season_all.columns = ["季节", "Count"]
        season_sev = dff.groupby(["季节", "Severity"]).size().reset_index(name="count")
        fig2e = go.Figure()
        fig2e.add_trace(go.Bar(x=season_all["季节"], y=season_all["Count"],
                                name="总计", width=0.4,
                                marker=dict(
                                    color=["#a8d8ea", "#f4d03f", "#e67e22", "#5dade2"],
                                    line=dict(color=["#7eb8cc", "#d4b55a", "#d4956a", "#3b8ec2"], width=1)),
                                opacity=0.75))
        season_sev["季节"] = pd.Categorical(season_sev["季节"], categories=season_order, ordered=True)
        for sev in sorted(dff["Severity"].dropna().unique()):
            sdf = season_sev[season_sev["Severity"] == sev].sort_values("季节")
            fig2e.add_trace(go.Scatter(x=sdf["季节"], y=sdf["count"],
                                       mode="lines+markers",
                                       name=f"Severity {sev}",
                                       line=dict(width=2)))
        fig2e.update_layout(title="季节分布", xaxis_title="季节", yaxis_title="事故数量")
        st.plotly_chart(fig2e, width="stretch")

st.divider()

# ── 3. 天气条件与气象指标柱状图 ──
with st.container():
    st.subheader("🌦️ 3. 天气条件与气象指标")
    col_a, col_b = st.columns(2)
    with col_a:
        top_wx = dff["Weather_Condition"].value_counts().head(12).reset_index()
        top_wx.columns = ["condition", "count"]
        fig8 = px.bar(top_wx, x="condition", y="count",
                      title="Top 12 天气条件",
                      color="count", color_continuous_scale="Tealgrn",
                      labels={"condition": "天气", "count": "事故数量"})
        fig8.update_xaxes(tickangle=30)
        st.plotly_chart(fig8, width="stretch")
    with col_b:
        sev_group = dff.groupby("Severity")[["Temperature(C)", "Humidity(%)",
                                              "Visibility(km)", "Wind_Speed(kmh)"]].mean().round(1)
        fig8b = go.Figure(data=[
            go.Bar(name="温度 (°C)", x=sev_group.index, y=sev_group["Temperature(C)"]),
            go.Bar(name="湿度 (%)", x=sev_group.index, y=sev_group["Humidity(%)"]),
            go.Bar(name="能见度 (km)", x=sev_group.index, y=sev_group["Visibility(km)"]),
            go.Bar(name="风速 (km/h)", x=sev_group.index, y=sev_group["Wind_Speed(kmh)"]),
        ])
        fig8b.update_layout(barmode="group", title="各严重程度下气象指标均值",
                            xaxis_title="严重程度", yaxis_title="均值")
        st.plotly_chart(fig8b, width="stretch")

st.divider()

# ── 温度与湿度 (直方图 + 散点) ──
with st.container():
    st.subheader("🌡️ 温度与湿度")
    col_a, col_b = st.columns(2)
    with col_a:
        fig3a = px.histogram(dff, x="Temperature(C)", nbins=40,
                             labels={"Temperature(C)": "温度 (°C)", "count": "事故数量"},
                             title="温度分布直方图",
                             color_discrete_sequence=["#457b9d"])
        fig3a.update_layout(bargap=0.02)
        st.plotly_chart(fig3a, width="stretch")
    with col_b:
        fig3b = px.histogram(dff, x="Humidity(%)", nbins=40,
                             labels={"Humidity(%)": "湿度 (%)", "count": "事故数量"},
                             title="湿度分布直方图",
                             color_discrete_sequence=["#2a9d8f"])
        fig3b.update_layout(bargap=0.02)
        st.plotly_chart(fig3b, width="stretch")

st.divider()

# ── 能见度与降水量 ──
with st.container():
    st.subheader("👁️ 能见度与降水量")
    col_a, col_b = st.columns(2)
    with col_a:
        fig4a = px.histogram(dff, x="Visibility(km)", nbins=40,
                             labels={"Visibility(km)": "能见度 (km)", "count": "事故数量"},
                             title="能见度分布直方图",
                             color_discrete_sequence=["#457b9d"])
        fig4a.update_layout(bargap=0.02)
        st.plotly_chart(fig4a, width="stretch")
    with col_b:
        fig4b = go.Figure()
        fig4b.add_trace(go.Histogram(x=dff["Precipitation(mm)"], nbinsx=40,
                                      marker_color="#2a9d8f", opacity=0.8))
        fig4b.update_layout(title="降水量分布直方图",
                             xaxis_title="降水量 (mm)", yaxis_title="事故数量",
                             yaxis_type="log", bargap=0.02)
        st.plotly_chart(fig4b, width="stretch")

st.divider()

# ── 风速 ──
with st.container():
    st.subheader("💨 风速")
    col_a, col_b = st.columns(2)
    with col_a:
        bins = [0, 0.5, 1.5, 3.5, 5, 8.5, 10.5, 13.5, 16.5, 20, 23.5, 27.5, 31.5, 35]
        labels = ["<0.5", "0.5-1.5", "1.5-3.5", "3.5-5", "5-8.5", "8.5-10.5",
                  "10.5-13.5", "13.5-16.5", "16.5-20", "20-23.5", "23.5-27.5",
                  "27.5-31.5", "31.5-35"]
        dff["风速区间"] = pd.cut(dff["Wind_Speed(kmh)"], bins=bins, labels=labels,
                                include_lowest=True, right=False)
        ws_counts = dff["风速区间"].value_counts().reindex(labels).reset_index()
        ws_counts.columns = ["风速区间", "事故数量"]
        fig5a = px.bar(ws_counts, x="风速区间", y="事故数量",
                       title="事故数量 - 风速",
                       color="事故数量", color_continuous_scale="Reds")
        st.plotly_chart(fig5a, width="stretch")
    with col_b:
        # 风向玫瑰图（极坐标柱状图）
        dir_order = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        wind_counts = dff["Wind_Direction"].value_counts()
        wind_counts = wind_counts[wind_counts.index.isin(dir_order)].reindex(dir_order, fill_value=0).reset_index()
        wind_counts.columns = ["direction", "count"]
        # 角度：北=90° 顺时针，但 plotly 的 theta 默认 0=东 逆时针，这里直接用 direction 作为类别标签
        fig5b = go.Figure(go.Barpolar(
            r=wind_counts["count"].values,
            theta=wind_counts["direction"].values,
            marker=dict(
                color=wind_counts["count"].values,
                colorscale="Teal",
                cmin=0,
                cmax=wind_counts["count"].max(),
                showscale=True,
                colorbar=dict(title="事故数"),
            ),
            hovertemplate="<b>%{theta}</b><br>事故数量: %{r}<extra></extra>",
        ))
        fig5b.update_layout(
            title="各风向事故数量分布",
            polar=dict(
                angularaxis=dict(
                    direction="clockwise",
                    period=16,
                    tickmode="array",
                    tickvals=dir_order[::2],
                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                ),
                radialaxis=dict(showticklabels=True, ticks=""),
            ),
        )
        st.plotly_chart(fig5b, width="stretch")

st.divider()

# ── 4. 州 (条形图) ──
with st.container():
    st.subheader("🗺️ 4. 各州事故数量 Top 15")
    col_a, col_b = st.columns(2)
    with col_a:
        top_states = dff["State"].value_counts().head(15).reset_index()
        top_states.columns = ["State", "Count"]
        fig7a = px.bar(top_states, x="Count", y="State", orientation="h",
                       title="事故最多的 15 个州",
                       color="Count", color_continuous_scale=[[0, "#f4a6a6"], [1, "#8b0000"]],
                       text_auto=True)
        fig7a.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig7a, width="stretch")
    with col_b:
        top_cities = dff["City"].value_counts().head(15).reset_index()
        top_cities.columns = ["City", "Count"]
        fig7b = px.bar(top_cities, x="Count", y="City", orientation="h",
                       title="事故最多的 15 个城市",
                       color="Count", color_continuous_scale=[[0, "#a8d8ea"], [1, "#1d3557"]],
                       text_auto=True)
        fig7b.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig7b, width="stretch")

st.divider()

# ── 5. 道路设施特征 (分组柱状图) ──
with st.container():
    st.subheader("🚧 5. 道路设施特征与事故")
    feature_cols = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction",
                    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
                    "Traffic_Calming", "Traffic_Signal"]
    feat_present = {c: (dff[c] == 1).sum() for c in feature_cols}
    feat_df = pd.DataFrame(list(feat_present.items()), columns=["Feature", "Count"])
    feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ")

    fig9 = px.bar(feat_df, x="Count", y="Feature", orientation="h",
                  title="道路设施特征出现频次",
                  color="Count", color_continuous_scale=[[0, "#c9b3d4"], [1, "#4a148c"]],
                  text_auto=True)
    fig9.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig9, width="stretch")

st.divider()

# ── 6. 地理位置 ──
with st.container():
    st.subheader("📍 6. 事故地理分布")

    map_sample = dff.sample(min(20000, len(dff)))

    # 热力图 + 散点叠加在同一地图上
    fig = px.density_mapbox(
        map_sample,
        lat="Start_Lat",
        lon="Start_Lng",
        radius=15,
        center=dict(lat=39.5, lon=-98.0),
        zoom=3,
        mapbox_style="open-street-map",
        opacity=0.6,
        title="事故地理分布",
        color_continuous_scale="Viridis",
    )
    # 叠加散点层
    fig.add_scattermapbox(
        lat=map_sample["Start_Lat"],
        lon=map_sample["Start_Lng"],
        mode="markers",
        marker=dict(size=3, color="rgba(255,100,100,0.3)"),
        showlegend=False,
        hoverinfo="none",
    )
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")

    # ── DBSCAN 聚类地图 ──
    st.markdown("#### DBSCAN 聚类")
    st.markdown("DBSCAN 密度聚类 · 7 组参数 · **最优：eps=800m, min_samples=10** ｜ 覆盖率94.83%, 噪声比5.17%, 热点24,882个")
    dbscan_tabs = st.tabs(["🏆 平衡1(最优)", "保守1", "保守2", "平衡2", "平衡3", "平衡4", "宽松1"])
    dbscan_htmls = [
        "src/maps/map_平衡1_eps800_min10.html",
        "src/maps/map_保守1_eps500_min15.html",
        "src/maps/map_保守2_eps600_min12.html",
        "src/maps/map_平衡2_eps900_min9.html",
        "src/maps/map_平衡3_eps1000_min10.html",
        "src/maps/map_平衡4_eps1100_min9.html",
        "src/maps/map_宽松1_eps1200_min8.html",
    ]
    for tab, path in zip(dbscan_tabs, dbscan_htmls):
        with tab:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)

    # ── MiniBatchKMeans 聚类地图 ──
    st.markdown("#### MiniBatchKMeans 聚类")
    mbk_tabs = st.tabs(["宏观500", "中观2000", "微观5000"])
    mbk_htmls = [
        "src/maps/map_宏观_500_k500.html",
        "src/maps/map_中观_2000_k2000.html",
        "src/maps/map_微观_5000_k5000.html",
    ]
    for tab, path in zip(mbk_tabs, mbk_htmls):
        with tab:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)

st.divider()

# ── 7. 混淆矩阵 + 模型指标 ──
with st.container():
    st.subheader("🔮 7. 模型评估总览")

    # 读取指标
    metrics = pd.read_csv("src/outputs/reports/metrics_summary.csv")

    st.markdown("### 📊 指标对比")
    # 显示 metrics 表格（全部数据）
    display_df = metrics.copy()
    display_df = display_df.round(4)
    st.dataframe(display_df, width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        models_short = metrics["model"].str.replace("_", " ").tolist()
        fig_m1 = px.bar(metrics, x="model", y="accuracy",
                        title="各模型准确率",
                        color="accuracy",
                        range_color=[metrics["accuracy"].min() - 0.02, metrics["accuracy"].max() + 0.01],
                        color_continuous_scale=[[0, "#a8d8ea"], [1, "#1d3557"]],
                        labels={"model": "模型", "accuracy": "准确率"})
        fig_m1.update_xaxes(tickangle=30)
        st.plotly_chart(fig_m1, width="stretch")
    with col_b:
        fig_m2 = px.bar(metrics, x="model", y=["macro_f1", "weighted_f1"],
                        title="F1 分数对比",
                        barmode="group",
                        labels={"model": "模型", "value": "F1 分数", "variable": "指标"},
                        color_discrete_sequence=["#457b9d", "#e63946"])
        fig_m2.update_xaxes(tickangle=30)
        st.plotly_chart(fig_m2, width="stretch")

    st.markdown("#### 各严重程度 F1")
    sev_f1_cols = ["severity_1_f1", "severity_2_f1", "severity_3_f1", "severity_4_f1"]
    sev_long = metrics.melt(id_vars=["model"], value_vars=sev_f1_cols,
                             var_name="severity", value_name="f1")
    fig_m3 = px.bar(sev_long, x="model", y="f1", color="severity",
                    barmode="group",
                    labels={"model": "模型", "f1": "F1 分数", "severity": "严重程度"},
                    color_discrete_sequence=["#a8d8ea", "#457b9d", "#1d3557", "#e63946"])
    fig_m3.update_xaxes(tickangle=30)
    st.plotly_chart(fig_m3, width="stretch")

    st.markdown("### 🖼️ 混淆矩阵")
    cm_files = [
        ("Decision Tree (Baseline)", "src/outputs/figures/confusion_matrix_decisiontree_baseline.png"),
        ("LightGBM (Random Oversampler)", "src/outputs/figures/confusion_matrix_lightgbm_random_oversampler.png"),
        ("LightGBM (Balanced)", "src/outputs/figures/confusion_matrix_lightgbm_balanced.png"),
        ("LightGBM (Weight Power 0.35)", "src/outputs/figures/confusion_matrix_lightgbm_weight_power_035.png"),
        ("LightGBM (Weight Power 0.50)", "src/outputs/figures/confusion_matrix_lightgbm_weight_power_050.png"),
        ("LightGBM (Weight Power 0.60)", "src/outputs/figures/confusion_matrix_lightgbm_weight_power_060.png"),
        ("LightGBM (Weight Power 0.70)", "src/outputs/figures/confusion_matrix_lightgbm_weight_power_070.png"),
        ("Random Forest (Balanced)", "src/outputs/figures/confusion_matrix_randomforest_balanced.png"),
        ("XGBoost (Random Search)", "src/outputs/figures/confusion_matrix_xgboost_random_search_macro_f1.png"),
    ]
    for i in range(0, len(cm_files), 3):
        cols = st.columns(3)
        for col, (name, path) in zip(cols, cm_files[i:i+3]):
            with col:
                st.image(path, caption=name, width="stretch")

st.divider()

# ── 8. 事故严重程度预测 ──
with st.container():
    st.subheader("🧠 8. 事故严重程度预测")
    st.markdown(
        "输入事故发生时可获得的时间、天气、道路和地理条件，"
        "系统会调用已训练的 XGBoost 模型预测可能的 `Severity` 等级。"
    )

    if predictor is None:
        st.warning(
            "未检测到 XGBoost 预测模型文件。请先生成并上传 "
            "`XGBoost_random_search_macro_f1.joblib` 后再使用本模块。"
        )
    else:
        st.caption("当前预测模型：XGBoost（Random Search / Macro F1 优化模型）")

        with st.form("severity_prediction_form", clear_on_submit=False):
            st.markdown("##### 基本事故条件")
            col1, col2, col3 = st.columns(3)

            with col1:
                state = st.selectbox("州 State", sorted(df["State"].dropna().astype(str).unique()))
                state_df = df[df["State"].astype(str) == state]
                county_options = sorted(state_df["County"].dropna().astype(str).unique())
                city_options = sorted(state_df["City"].dropna().astype(str).unique())
                county = st.selectbox(
                    "县 County",
                    county_options if county_options else [most_common_value(df, "County")],
                )
                city = st.selectbox(
                    "城市 City",
                    city_options if city_options else [most_common_value(df, "City")],
                )

            with col2:
                weather = st.selectbox("天气 Weather Condition", sorted(df["Weather_Condition"].dropna().astype(str).unique()))
                wind_direction = st.selectbox("风向 Wind Direction", sorted(df["Wind_Direction"].dropna().astype(str).unique()))
                dayofweek = st.slider("星期 DayOfWeek（0=周一，6=周日）", 0, 6, 1)

            with col3:
                hour = st.slider("事故发生小时 Hour", 0, 23, 8)
                month = st.slider("月份 Month", 1, 12, 6)
                rush_hour = int(hour in [7, 8, 9, 16, 17, 18])

            st.markdown("##### 地理与天气数值条件")
            geo_weather_cols = st.columns(7)
            with geo_weather_cols[0]:
                lat = st.number_input("纬度 Start_Lat", value=float(df["Start_Lat"].median()), format="%.6f")
            with geo_weather_cols[1]:
                lng = st.number_input("经度 Start_Lng", value=float(df["Start_Lng"].median()), format="%.6f")
            with geo_weather_cols[2]:
                temp = st.number_input("温度 Temperature(F)", value=60.0)
            with geo_weather_cols[3]:
                humidity = st.number_input("湿度 Humidity(%)", value=60.0)
            with geo_weather_cols[4]:
                visibility = st.number_input("能见度 Visibility(mi)", value=10.0)
            with geo_weather_cols[5]:
                wind_speed = st.number_input("风速 Wind Speed(mph)", value=5.0)
            with geo_weather_cols[6]:
                precipitation = st.number_input("降水量 Precipitation(in)", value=0.0)

            st.markdown("##### 道路环境特征")
            road_col1, road_col2, road_col3, road_col4 = st.columns(4)
            with road_col1:
                amenity = st.checkbox("Amenity 公共设施")
                crossing = st.checkbox("Crossing 人行横道")
                junction = st.checkbox("Junction 交叉口")
            with road_col2:
                stop = st.checkbox("Stop 停车标志")
                traffic_signal = st.checkbox("Traffic Signal 信号灯")
                bump = st.checkbox("Bump 减速带")
            with road_col3:
                give_way = st.checkbox("Give Way 让行")
                no_exit = st.checkbox("No Exit 无出口")
                railway = st.checkbox("Railway 铁路")
            with road_col4:
                roundabout = st.checkbox("Roundabout 环岛")
                station = st.checkbox("Station 车站")
                traffic_calming = st.checkbox("Traffic Calming 交通缓行设施")

            submitted = st.form_submit_button("预测事故严重程度", type="primary")

        if submitted:
            input_data = pd.DataFrame(
                [
                    {
                        "Hour": hour,
                        "DayOfWeek": dayofweek,
                        "Month": month,
                        "IsWeekend": int(dayofweek in [5, 6]),
                        "RushHour": rush_hour,
                        "Start_Lat": lat,
                        "Start_Lng": lng,
                        "Temperature(F)": temp,
                        "Humidity(%)": humidity,
                        "Visibility(mi)": visibility,
                        "Wind_Speed(mph)": wind_speed,
                        "Precipitation(in)": precipitation,
                        "Amenity": int(amenity),
                        "Crossing": int(crossing),
                        "Junction": int(junction),
                        "Stop": int(stop),
                        "Traffic_Signal": int(traffic_signal),
                        "Bump": int(bump),
                        "Give_Way": int(give_way),
                        "No_Exit": int(no_exit),
                        "Railway": int(railway),
                        "Roundabout": int(roundabout),
                        "Station": int(station),
                        "Traffic_Calming": int(traffic_calming),
                        "State": state,
                        "County": county,
                        "City": city,
                        "Weather_Condition": weather,
                        "Wind_Direction": wind_direction,
                    }
                ]
            )
            input_data = add_prediction_features(input_data)

            pred = int(predictor.predict(input_data)[0])
            severity_explain = {
                1: "轻微影响事故",
                2: "一般影响事故",
                3: "较严重影响事故",
                4: "严重影响事故",
            }

            st.session_state["last_prediction"] = pred
            st.session_state["last_prediction_explain"] = severity_explain.get(pred, "未知等级")
            try:
                proba = predictor.predict_proba(input_data)[0]
                classes = getattr(predictor, "classes_", [1, 2, 3, 4])
                st.session_state["last_prediction_proba"] = pd.DataFrame(
                    {"Severity": classes, "Probability": proba}
                )
            except Exception as exc:
                st.session_state["last_prediction_proba"] = None
                st.session_state["last_prediction_error"] = str(exc)

        if "last_prediction" in st.session_state:
            st.markdown("##### 预测结果")
            st.success(f"预测结果：Severity = {st.session_state['last_prediction']}")
            st.info(f"结果解释：模型判断该事故更可能属于「{st.session_state['last_prediction_explain']}」。")

            proba_df = st.session_state.get("last_prediction_proba")
            if proba_df is not None:
                proba_df_plot = proba_df.copy()
                proba_df_plot["Severity"] = proba_df_plot["Severity"].astype(str)
                fig_pred = px.bar(
                    proba_df_plot,
                    x="Severity",
                    y="Probability",
                    color="Severity",
                    color_discrete_map={
                        "1": "#ff4b4b",
                        "2": "#e03131",
                        "3": "#c1121f",
                        "4": "#7f0000",
                    },
                    category_orders={"Severity": ["1", "2", "3", "4"]},
                    text=proba_df_plot["Probability"].apply(lambda x: f"{x:.2%}"),
                    title="各 Severity 等级预测概率",
                )
                fig_pred.update_traces(
                    textposition="inside",
                    textfont_color="white",
                    marker_line_color="white",
                    marker_line_width=1,
                )
                fig_pred.update_layout(
                    yaxis_tickformat=".0%",
                    xaxis_title="Severity 等级",
                    yaxis_title="预测概率",
                    showlegend=False,
                )
                st.plotly_chart(fig_pred, width="stretch")
            elif "last_prediction_error" in st.session_state:
                st.warning(f"概率输出失败，但分类预测已完成：{st.session_state['last_prediction_error']}")



# ── 底部 ──
st.divider()
st.markdown(f"*数据最后更新：{dff['Start_Time'].max().strftime('%Y-%m-%d')}　｜　数据记录数：{len(dff):,}*")
st.caption("数据来源：US Accidents 数据集")
