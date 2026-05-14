"""Optimizer Benchmark — Streamlit UI entry point."""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Optimizer Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────
st.sidebar.title("⚡ Optimizer Benchmark")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏋️ Train & Benchmark", "🔮 Try the Model", "📊 Compare Results"],
)

import glob as _glob
_model_counts = {
    "MNIST": len(_glob.glob(os.path.join("runs", "mnist_cnn", "models", "*.pt"))),
    "CIFAR-10": len(_glob.glob(os.path.join("runs", "cifar_resnet8", "models", "*.pt"))),
    "Housing": len(_glob.glob(os.path.join("runs", "housing_mlp", "models", "*.pt"))),
}
_total = sum(_model_counts.values())
if _total > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Trained Models: {_total}**")
    for _name, _count in _model_counts.items():
        if _count > 0:
            st.sidebar.caption(f"{_name}: {_count}")

# ── Page routing ─────────────────────────────────────────────────
if page == "🏋️ Train & Benchmark":
    from app.pages import train_page
    train_page.render()

elif page == "🔮 Try the Model":
    from app.pages import predict_page
    predict_page.render()

elif page == "📊 Compare Results":
    from app.pages import compare_page
    compare_page.render()