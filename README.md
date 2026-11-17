# AI Cancer Research Assistant (Project 01/100)

தமிழில் விளக்கம் + Python code. Local PDF RAG prototype; paid API key தேவையில்லை.

## அம்சங்கள்
- PDF upload மற்றும் text extraction
- TF-IDF அடிப்படையிலான local retrieval
- Optional Ollama local LLM generation
- Evidence excerpts மற்றும் source/chunk references
- FastAPI backend, Streamlit frontend
- Pytest unit tests

## தேவைகள்
- Python 3.11 அல்லது அதற்கு மேல்
- 4 GB RAM-ல் TF-IDF fallback சிறிய ஆவணத் தொகுப்புடன் இயங்கலாம்
- Ollama பயன்படுத்தினால் கூடுதல் RAM/வட்டு இடம் தேவைப்படும்

## நிறுவல்

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## இயக்குதல்

Terminal 1:
```bash
uvicorn app.main:app --reload
```

Terminal 2:
```bash
streamlit run frontend.py
```

Browser-ல் Streamlit காட்டும் local URL-ஐ திறக்கவும். Backend API docs:
`http://127.0.0.1:8000/docs`

## Optional: Local LLM (API key இல்லை)

Ollama நிறுவி, உதாரணமாக:
```bash
ollama pull qwen2.5:3b
ollama serve
```

`app/main.py`-ல் engine உருவாக்கும் வரியை:
```python
engine = ResearchEngine(str(DATA), use_ollama=True, ollama_model="qwen2.5:3b")
```
என்று மாற்றி backend-ஐ மறுதொடக்கம் செய்யவும். GPU இல்லாத consumer PC-ல் சிறிய quantized model பயன்படுத்தவும்; மெதுவாக இருக்கலாம். Ollama கிடைக்காவிட்டால் TF-IDF fallback இயங்கும்.

## சோதனை
```bash
pytest -q
```

## API
- `GET /health` – backend நிலை
- `GET /documents` – பதிவேற்றிய PDF பட்டியல்
- `POST /upload` – PDF upload (multipart field: `file`)
- `POST /query` – JSON: `{"question":"...","top_k":5}`

## வரம்புகள் மற்றும் பாதுகாப்பு
- Scanned PDF-களுக்கு OCR தேவை; இந்த version OCR செய்யாது.
- TF-IDF semantic understanding அல்ல; சொல் ஒற்றுமை அடிப்படையிலான retrieval.
- Ollama பதில்கள் தவறாக இருக்கலாம். மேற்கோள்கள் கொடுக்கப்பட்டாலும் அவை சரியான மருத்துவ முடிவை உறுதி செய்யாது.
- இந்த மென்பொருள் நோயறிதல், மருந்து/சிகிச்சை பரிந்துரை அல்லது அவசர மருத்துவ முடிவுகளுக்குப் பயன்படுத்தப்படக் கூடாது.
- நோயாளி அடையாளத் தகவல்களை பதிவேற்ற வேண்டாம். உள்ளூர் ஆவணங்களைப் பாதுகாப்பாக வைத்திருக்கவும்.
- உண்மையான ஆராய்ச்சிக்கு peer-reviewed மற்றும் நம்பகமான ஆதாரங்களை மனித நிபுணர் சரிபார்க்க வேண்டும்.

## Project structure
```text
ai_cancer_research_assistant/
├── app/
│   ├── __init__.py
│   ├── engine.py
│   └── main.py
├── data/documents/
├── tests/test_engine.py
├── frontend.py
├── requirements.txt
├── README.md
└── .gitignore
```
