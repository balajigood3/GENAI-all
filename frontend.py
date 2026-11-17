import requests
import streamlit as st

st.set_page_config(page_title="AI Cancer Research Assistant", page_icon="🧬", layout="wide")
API = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
st.title("AI Cancer Research Assistant")
st.caption("உள்ளூர் ஆராய்ச்சி ஆவணங்களைத் தேடும் RAG Prototype | மருத்துவ முடிவுகளுக்காக அல்ல")

with st.sidebar:
    st.header("ஆவண மேலாண்மை")
    uploaded = st.file_uploader("ஆராய்ச்சி PDF பதிவேற்றவும்", type=["pdf"])
    if st.button("PDF Upload", disabled=uploaded is None):
        try:
            r = requests.post(f"{API}/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, timeout=120)
            r.raise_for_status()
            st.success(r.json().get("message", "Uploaded"))
            st.json(r.json())
        except Exception as e:
            st.error(f"Upload தோல்வி: {e}")
    if st.button("ஆவணப் பட்டியல் புதுப்பிக்கவும்"):
        try:
            st.json(requests.get(f"{API}/documents", timeout=10).json())
        except Exception as e:
            st.error(str(e))

st.subheader("ஆராய்ச்சி கேள்வி")
question = st.text_area("உதாரணம்: What are the limitations of the studies describing a specific cancer biomarker?", height=110)
top_k = st.slider("தேட வேண்டிய ஆதாரப் பகுதிகள்", 1, 10, 5)
if st.button("ஆராய்ச்சி செய்", type="primary"):
    if len(question.strip()) < 5:
        st.warning("குறைந்தது 5 எழுத்துகள் கொண்ட கேள்வியை உள்ளிடவும்.")
    else:
        with st.spinner("ஆவணங்களில் ஆதாரங்களைத் தேடுகிறது..."):
            try:
                response = requests.post(f"{API}/query", json={"question": question, "top_k": top_k}, timeout=180)
                response.raise_for_status()
                result = response.json()
                st.markdown("### ஆராய்ச்சி முடிவு")
                st.write(result["answer"])
                st.caption(f"Processing mode: {result['mode']}")
                st.markdown("### ஆதாரங்கள்")
                for i, item in enumerate(result["evidence"], 1):
                    with st.expander(f"[{i}] {item['source']} | பகுதி {item['chunk']} | score {item['score']}"):
                        st.write(item["excerpt"])
                st.markdown("### வரம்புகள்")
                for limitation in result["limitations"]:
                    st.warning(limitation)
            except Exception as e:
                st.error(f"Backend-ஐ அணுக முடியவில்லை: {e}")

st.divider()
st.info("இந்த Prototype பதிவேற்றப்பட்ட PDF-களிலிருந்து மட்டுமே தேடுகிறது. ஆதாரங்களின் தரம், ஆய்வு முடிவுகள் மற்றும் மருத்துவப் பொருள் ஆகியவற்றைத் தகுதியான நிபுணர் சரிபார்க்க வேண்டும்.")
