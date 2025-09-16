import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

from datetime import datetime
from brand_similarity_analysis import run_brand_similarity

st.set_page_config(page_title="Amazon Brand & Review Analyzer", layout="wide")
st.title("Amazon Brand & Review Analyzer")

st.markdown("""
<style>
/* center headers and cells for all Streamlit dataframes */
thead tr th, tbody tr td {
    text-align: center !important;
    vertical-align: middle !important;
}
/* center Plotly/Matplotlib titles (just in case) */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Controls")
search_url = st.sidebar.text_input("Amazon search URL", value="https://www.amazon.in/s?k=trimmer")
num_products = st.sidebar.slider("Number of Products", 5, 30, 15)
run_brand = st.sidebar.button("Run Brand Similarity")

OUT_DIR = "outputs"
REV_OUT = "outputs_reviews"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REV_OUT, exist_ok=True)

if run_brand:
    st.info("Running brand similarity...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUT_DIR, ts)
    os.makedirs(outdir, exist_ok=True)
    try:
        run_brand_similarity(st, search_url, num_products=num_products, out_dir=outdir)
        st.success(f"Results saved in {outdir}")
    except Exception as e:
        st.error(f"Error: {e}")



