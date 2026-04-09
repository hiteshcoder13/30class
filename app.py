import streamlit as st
from PIL import Image
import tempfile
import os

from query.query_image import query_image


st.set_page_config(page_title="Stone Color Search", layout="wide")

st.title("🪨 Stone Family Color Search")
st.write("Upload an image → get top matching stone families")

# ---- Upload Image ----
uploaded_file = st.file_uploader("Upload stone image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # ---- Run Query ----
    with st.spinner("🔍 Finding similar stone families..."):
        results = query_image(temp_path, top_k=200)

    os.remove(temp_path)

    # ---- Show Results ----
    st.subheader("🎯 Top Matching Stone Families")

    if not results:
        st.error("No results found")
    else:
        for i, (family, score) in enumerate(results, 1):
            st.write(f"{i}. **{family}** (score: {score})")