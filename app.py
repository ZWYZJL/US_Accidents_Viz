import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="交通事故数据可视化", layout="wide")

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

# ── 读取交互式预测模型 ──
@st.cache_resource
def load_predictor():
    model_path = Path("models/severity_predictor.joblib")
    if not model_path.exists():
        return None
    return joblib.load(model_path)

predictor = load_predictor()

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
            st.iframe(html_content, height=600)

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
            st.iframe(html_content, height=600)

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

# =========================================================
# 事故严重程度预测模块：使用 st.form，避免每改一个参数就刷新
# =========================================================
st.markdown("---")
st.header("🧠 事故严重程度预测")

st.markdown(
    "输入事故发生时的时间、天气、道路环境和地理位置等条件，"
    "系统将调用训练好的分类模型预测该事故可能对应的 Severity 等级。"
)

if predictor is None:
    st.warning("未检测到模型文件：models/severity_predictor.joblib。请先运行 train_predictor_for_app.py 生成模型。")
else:
    with st.form("severity_prediction_form", clear_on_submit=False):
        st.subheader("1. 基本事故条件")

        col1, col2, col3 = st.columns(3)

        with col1:
            state = st.selectbox(
                "州 State",
                sorted(df["State"].dropna().unique()),
                key="form_pred_state"
            )

            weather = st.selectbox(
                "天气 Weather Condition",
                sorted(df["Weather_Condition"].dropna().unique()),
                key="form_pred_weather"
            )

            wind_direction = st.selectbox(
                "风向 Wind Direction",
                sorted(df["Wind_Direction"].dropna().unique()),
                key="form_pred_wind_direction"
            )

        with col2:
            hour = st.slider(
                "事故发生小时 Hour",
                min_value=0,
                max_value=23,
                value=8,
                key="form_pred_hour"
            )

            month = st.slider(
                "月份 Month",
                min_value=1,
                max_value=12,
                value=6,
                key="form_pred_month"
            )

            dayofweek = st.slider(
                "星期 DayOfWeek（0=周一，6=周日）",
                min_value=0,
                max_value=6,
                value=1,
                key="form_pred_dayofweek"
            )

        with col3:
            lat = st.number_input(
                "纬度 Start_Lat",
                value=float(df["Start_Lat"].median()),
                format="%.6f",
                key="form_pred_lat"
            )

            lng = st.number_input(
                "经度 Start_Lng",
                value=float(df["Start_Lng"].median()),
                format="%.6f",
                key="form_pred_lng"
            )

        st.subheader("2. 天气数值条件")

        weather_col1, weather_col2, weather_col3, weather_col4, weather_col5 = st.columns(5)

        with weather_col1:
            temp = st.number_input(
                "温度 Temperature(F)",
                value=60.0,
                key="form_pred_temp"
            )

        with weather_col2:
            humidity = st.number_input(
                "湿度 Humidity(%)",
                value=60.0,
                key="form_pred_humidity"
            )

        with weather_col3:
            visibility = st.number_input(
                "能见度 Visibility(mi)",
                value=10.0,
                key="form_pred_visibility"
            )

        with weather_col4:
            wind_speed = st.number_input(
                "风速 Wind Speed(mph)",
                value=5.0,
                key="form_pred_wind_speed"
            )

        with weather_col5:
            precipitation = st.number_input(
                "降水量 Precipitation(in)",
                value=0.0,
                key="form_pred_precipitation"
            )

        st.subheader("3. 道路环境特征")

        road_col1, road_col2, road_col3, road_col4 = st.columns(4)

        with road_col1:
            amenity = st.checkbox("Amenity 公共设施", key="form_pred_amenity")
            bump = st.checkbox("Bump 减速带", key="form_pred_bump")
            crossing = st.checkbox("Crossing 人行横道", key="form_pred_crossing")

        with road_col2:
            give_way = st.checkbox("Give Way 让行", key="form_pred_give_way")
            junction = st.checkbox("Junction 交叉口", key="form_pred_junction")
            no_exit = st.checkbox("No Exit 无出口", key="form_pred_no_exit")

        with road_col3:
            railway = st.checkbox("Railway 铁路", key="form_pred_railway")
            roundabout = st.checkbox("Roundabout 环岛", key="form_pred_roundabout")
            station = st.checkbox("Station 车站", key="form_pred_station")

        with road_col4:
            stop = st.checkbox("Stop 停车标志", key="form_pred_stop")
            traffic_calming = st.checkbox("Traffic Calming 交通缓行设施", key="form_pred_traffic_calming")
            traffic_signal = st.checkbox("Traffic Signal 信号灯", key="form_pred_traffic_signal")

        submitted = st.form_submit_button("预测事故严重程度", type="primary")

    # 注意：预测逻辑写在 form 外面，但由 submitted 控制
    if submitted:
        input_data = pd.DataFrame([{
            "Start_Lat": lat,
            "Start_Lng": lng,
            "Temperature(F)": temp,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Hour": hour,
            "Month": month,
            "DayOfWeek": dayofweek,
            "IsWeekend": int(dayofweek in [5, 6]),
            "RushHour": int(hour in [7, 8, 9, 17, 18, 19]),
            "State": state,
            "Weather_Condition": weather,
            "Wind_Direction": wind_direction,
            "Amenity": int(amenity),
            "Bump": int(bump),
            "Crossing": int(crossing),
            "Give_Way": int(give_way),
            "Junction": int(junction),
            "No_Exit": int(no_exit),
            "Railway": int(railway),
            "Roundabout": int(roundabout),
            "Station": int(station),
            "Stop": int(stop),
            "Traffic_Calming": int(traffic_calming),
            "Traffic_Signal": int(traffic_signal),
        }])

        pred = predictor.predict(input_data)[0]

        severity_explain = {
            1: "轻微影响事故",
            2: "一般影响事故",
            3: "较严重影响事故",
            4: "严重影响事故"
        }

        st.session_state["last_prediction"] = int(pred)
        st.session_state["last_prediction_explain"] = severity_explain.get(int(pred), "未知等级")

        try:
            proba = predictor.predict_proba(input_data)[0]
            classes = predictor.classes_

            proba_df = pd.DataFrame({
                "Severity": classes,
                "Probability": proba
            })

            st.session_state["last_prediction_proba"] = proba_df

        except Exception as e:
            st.session_state["last_prediction_proba"] = None
            st.session_state["last_prediction_error"] = str(e)

    # 预测结果展示区：点击按钮后才出现
    if "last_prediction" in st.session_state:
        st.subheader("4. 预测结果")

        st.success(f"预测结果：Severity = {st.session_state['last_prediction']}")
        st.info(f"结果解释：模型判断该事故更可能属于「{st.session_state['last_prediction_explain']}」。")

        proba_df = st.session_state.get("last_prediction_proba")

        if proba_df is not None:
            # 保持与原网页一致的红色系风格
            proba_df_plot = proba_df.copy()
            proba_df_plot["Severity"] = proba_df_plot["Severity"].astype(str)

            severity_color_map = {
                "1": "#ff4b4b",
                "2": "#e03131",
                "3": "#c1121f",
                "4": "#7f0000",
            }

            fig = px.bar(
                proba_df_plot,
                x="Severity",
                y="Probability",
                color="Severity",
                color_discrete_map=severity_color_map,
                category_orders={"Severity": ["1", "2", "3", "4"]},
                text=proba_df_plot["Probability"].apply(lambda x: f"{x:.2%}"),
                title="各 Severity 等级预测概率"
            )

            fig.update_traces(
                textposition="inside",
                textfont_color="white",
                marker_line_color="white",
                marker_line_width=1
            )

            fig.update_layout(
                yaxis_tickformat=".0%",
                xaxis_title="Severity 等级",
                yaxis_title="预测概率",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            if "last_prediction_error" in st.session_state:
                st.warning(
                    f"概率输出失败，但分类预测已完成。错误信息：{st.session_state['last_prediction_error']}"
                )
           

# ── 底部 ──
st.divider()
st.markdown(f"*数据最后更新：{dff['Start_Time'].max().strftime('%Y-%m-%d')}　｜　数据记录数：{len(dff):,}*")
st.caption("数据来源：US Accidents 数据集")

# =========================================================
# 模型准确率自测模块
# =========================================================
st.markdown("---")
st.header("📊 预测模型准确率自测")

st.markdown(
    "该模块使用清洗后的样本数据重新划分测试集，"
    "调用已保存的 Severity 预测模型进行批量预测，"
    "并计算 Accuracy、Macro F1、Weighted F1 和混淆矩阵。"
)

if predictor is None:
    st.warning("未检测到模型文件：models/severity_predictor.joblib，无法进行模型自测。")
else:
    with st.expander("点击展开模型自测结果", expanded=False):

        # 和训练脚本保持一致的特征构造
        eval_df = df.copy()
        eval_df["Start_Time"] = pd.to_datetime(eval_df["Start_Time"])

        eval_df["Hour"] = eval_df["Start_Time"].dt.hour
        eval_df["Month"] = eval_df["Start_Time"].dt.month
        eval_df["DayOfWeek"] = eval_df["Start_Time"].dt.dayofweek
        eval_df["IsWeekend"] = eval_df["DayOfWeek"].isin([5, 6]).astype(int)
        eval_df["RushHour"] = eval_df["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

        target = "Severity"

        numeric_features = [
            "Start_Lat",
            "Start_Lng",
            "Temperature(F)",
            "Humidity(%)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "Hour",
            "Month",
            "DayOfWeek",
            "IsWeekend",
            "RushHour",
        ]

        categorical_features = [
            "State",
            "Weather_Condition",
            "Wind_Direction",
        ]

        binary_features = [
            "Amenity",
            "Bump",
            "Crossing",
            "Give_Way",
            "Junction",
            "No_Exit",
            "Railway",
            "Roundabout",
            "Station",
            "Stop",
            "Traffic_Calming",
            "Traffic_Signal",
        ]

        features = numeric_features + categorical_features + binary_features

        missing_cols = [col for col in features + [target] if col not in eval_df.columns]

        if missing_cols:
            st.error(f"当前数据缺少以下字段，无法评估模型：{missing_cols}")
        else:
            eval_df = eval_df.dropna(subset=[target])

            X = eval_df[features].copy()
            y = eval_df[target].astype(int)

            # 和训练脚本保持相同划分方式
            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            y_pred = predictor.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")

            col_a, col_b, col_c = st.columns(3)

            col_a.metric("Accuracy", f"{acc:.4f}")
            col_b.metric("Macro F1", f"{macro_f1:.4f}")
            col_c.metric("Weighted F1", f"{weighted_f1:.4f}")

            st.caption(
                "说明：Accuracy 表示整体预测正确率；Macro F1 更关注各类别均衡表现；"
                "Weighted F1 按类别样本量加权，更接近整体样本分布下的综合效果。"
            )

            # 混淆矩阵
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            cm_df = pd.DataFrame(
                cm,
                index=[f"真实 {i}" for i in labels],
                columns=[f"预测 {i}" for i in labels]
            )

            st.subheader("混淆矩阵")

            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Reds",
                title="Severity 预测混淆矩阵"
            )

            fig_cm.update_layout(
                xaxis_title="预测类别",
                yaxis_title="真实类别"
            )

            st.plotly_chart(fig_cm, use_container_width=True)

            # 展示部分预测样例
            st.subheader("部分测试样本预测结果")

            sample_result = X_test.copy()
            sample_result["真实 Severity"] = y_test.values
            sample_result["预测 Severity"] = y_pred
            sample_result["是否预测正确"] = sample_result["真实 Severity"] == sample_result["预测 Severity"]

            display_cols = [
                "State",
                "Weather_Condition",
                "Hour",
                "Month",
                "Start_Lat",
                "Start_Lng",
                "真实 Severity",
                "预测 Severity",
                "是否预测正确"
            ]

            st.dataframe(
                sample_result[display_cols].head(30),
                use_container_width=True
            )
