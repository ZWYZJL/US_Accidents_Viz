# 交通事故数据可视化 — 运行说明

## 启动

### 1. 激活虚拟环境

```bash
source venv/Scripts/activate
```

pip

### 2. 运行


```bash
streamlit run app.py --server.headless true --server.port 8501
```

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501


## 停止服务

```bash
pkill -9 -f streamlit
```