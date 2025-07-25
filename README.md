
# 📘 Bangla-English PDF Chatbot System

This project is a **bilingual document-based intelligent chatbot system** designed to process and respond to user queries based on **Bangla-English mixed PDF documents**. The system leverages advanced techniques in **OCR, text chunking, semantic retrieval, dense vector search, and LLM reranking** to ensure accurate and contextually relevant answers. It is built with a **FastAPI (Python) backend** and a **Node.js + React frontend** using Axios for communication.


## 🛠️ Setup Instructions

### ✅ Backend (Python - FastAPI)

LLM groq api use my one if you test project : 

```bash
# if you use my one 
venv/Scripts/Activate.ps1

# Create and activate virtual environment
python -m venv venv310  #(not mandotory take time download)
source venv310/bin/activate  # On Windows: venv310\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### ✅ Frontend (Node.js + React) **!! After run reqiurements.txt just run this automatically backend will run **

```bash
cd frontend
npm install
npm run start
```

### Optional: Fix for bnltk-related TensorFlow Issues if get issues

```bash
pip install tensorflow==2.18.0 tf-keras==2.18.0 tensorboard==2.18.0 ml-dtypes==0.4.0
```

---




## 🚀 Key Features

- 🔤 **Bilingual Support**: Understands both Bangla and English PDF content and queries.
- 📄 **Robust PDF Handling**: Uses PyMuPDF and PyPDF2 for native PDFs, and EasyOCR + OpenCV for scanned image-based text.
- 🧠 **Semantic Retrieval**: Uses sentence-transformer models to embed and search document chunks.
- ⚡ **Fast Vector Search**: FAISS with HNSW index for approximate nearest neighbor search.
- 🎯 **Cross-Encoder Reranking**: Enhances answer accuracy with LLM-based semantic relevance scoring.
- 🌐 **API Gateway and Frontend**: Seamless integration via Node.js, React, and Axios.
- 🛠️ **Structure-Aware Chunking**: Bangla-English mixed content segmentation with sentence and section awareness.
- 📈 **Keyword Extraction**: TF-IDF-based keyword handling to assist retrieval relevance.
- ⚙️ **FastAPI Backend**: Lightweight and asynchronous Python backend using FastAPI for handling uploads, indexing, and query response.



## 📁 Project Structure

```bash
Bangla_chatbot/
├── backend/
│   ├── app/                          # Main backend application modules
│   │   └── handlers/                # Request handlers and API endpoints
│   ├── vector_store/                # Vector storage and retrieval logic
│   │   └── pdf_handler.py           # PDF processing utilities
│   └── main.py                      # Backend server entry point
├── frontend/                        # Node.js + React client interface
│   └── src/
│       └── components/
│           └── Chatbox.js           # Chat interface component
│       ├── chatwidget.js           # Chat widget component
│       ├── App.js                  # React main app file
│       └── index.js                # React app entry point
│   └── public/                      # Static public files
│   └── node_modules/               # Node dependencies (auto-generated)
│   └── package.json                # Frontend dependencies and scripts
│   └── package-lock.json           # Lockfile for package versions
├── Indic-OCR/                      # Optional/legacy OCR module for Indic scripts
├── my-app/                         # Optional testing or sandbox app
├── venv/                           # Python virtual environment for backend
└── README.md                       # Project overview and instructions
```
---

Description:
This project is a Knowledge-Based Bangla Chatbot with a backend REST API powered by Python and FastAPI, and a frontend built with React.js. It includes modules for PDF handling, vector search, and optional OCR support for Indic scripts.

To run the project, you need to:

1. Setup the Python virtual environment in `venv/`
2. Install backend dependencies
3. Run the backend server (`main.py`)
4. Setup and run the React frontend (`frontend/`)

For more detailed instructions, please refer to the README.md file.




## ⚙️ Technologies Used

| Component              | Technology                                                     |
| ----------------------|---------------------------------------------------------------  |
| Text Extraction        | PyMuPDF, PyPDF2, EasyOCR, OpenCV                               |
| Chunking               | Custom structure-aware splitter (Bangla+English, NLTK-based)   |
| Embedding              | `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers)|
| Vector Search          | FAISS (HNSW Index with cosine similarity)                      |
| Reranking              | `ms-marco-MiniLM-L-6-v2` (CrossEncoder)                        |
| Backend API            | FastAPI                                                        |
| Frontend + Gateway     | Node.js + React + Axios                                        |
| LLM Integration        | Groq API with Meta-LLaMA-4                                     |
| Keyword Extraction     | Scikit-learn's `TfidfVectorizer`                               |




## 💬 Sample Queries & Outputs

### 🔤 Bangla

**Query:** রবীন্দ্রনাথ ঠাকুর কত সালে জন্মগ্রহণ করেছেন এবং মৃত্যুবরণ করেছেন?  
**Answer:** রবীন্দ্রনাথ ঠাকুর কত সালে জন্মগ্রহণ করেছেন এবং মৃত্যুবরণ করেছেন A: রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালের ৭ই মে জন্মগ্রহণ করেন এবং ১৯৪১ সালের ৮ই আগস্ট মৃত্যুবরণ করেন।

**Query:** অপরিচিতা উপন্যাসে কল্যাণীর বয়স কত ছিল?  
**Answer:** আমি জানি না। hard code quesry becouse  ( less knowledge and  less consine simlilarity  )

**Query:**  অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
**Answer:** অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? A: অনুপমের মতে, সুপুরুষ বলতে বোঝায় এমন কোনো ব্যক্তি, যার সৌন্দর্য অনন্য। ( The answer is poor due to low knowledge and the model needing more fine-tuning)

### 🔤 English

**Query:** What is the main theme of Chapter 2?  
**Answer:** আমি জানি না। (উল্লেখ্য: প্রদত্ত টেক্সটের কোনো অংশে "Chapter2" উল্লেখ নেই।) (becouse our text book have  অধয্ায়  but didn;t have  any chapter )

**Query:** LLM ? best model in 2024? for bangla  
**Answer:** Here's a refined version: Q: What is the best LLM model in 2024 for Bangla? A: I'm not aware of the most up-to-date information on LLM models for Bangla in 2024. However, I can suggest some options to help you find the answer: * Check recent research papers and industry reports on LLMs for Bangla. * Look for reputable sources, such as language model benchmarking websites or AI research organizations, that may have evaluated and compared different LLM models for Bangla. * Keep an eye on updates from leading AI companies, such as Meta, Google, or Microsoft, which may have developed or fine-tuned LLM models for Bangla. Some popular LLM models that you can explore include: * Multilingual models like mBERT, XLM-R, and LaBSE, which may have Bangla support. * Bangla-specific models like BanglaBERT, which was trained on Bangla text data. Keep in mind that the best model may depend on your specific use case, such as language translation, text classification, or question answering.

---

## 📡 API Documentation (Fast API)

### `POST /chat`
- **Request:** `{ "query": "তুমি কে?" }`
- **Response:** `{ "answer": "আমি একটি চ্যাটবট সহকারী, তোমার সাহায্যের জন্য তৈরি।" }`

### `GET /status`
- Returns readiness status of the chatbot backend.

---

## 📊 Evaluation Matrix

| Metric             | Description                                                         |
|-------------------|--------------------------------------------------------------        |
| Top-k Accuracy     | Match between query intent and retrieved text chunks                |
| Semantic Relevance | Measured using CrossEncoder score and manual validation             |
| Language Handling  | Dual-language query performance (Bangla + English)                  |
| Fallback Safety    | Triggering safe responses on vague, ambiguous, or unmatched queries |

---

##  System Design Questions & Answers

### Q1. What method or library was used for text extraction, and why?

**Answer:**  
I used a hybrid PDF parsing method: primarily PyPDF2 for extracting na-tive text and EasyOCR for image-based OCR fallback. Each page was first attempted with
PyPDF2; if the output was empty or under 20 characters, the image of the page was processed
using OpenCV (for deskewing, binarization, and sharpening) and then passed to EasyOCR
with Bangla and English language models. OCR was essential due to mixed-script PDFs or
scanned documents lacking machine-readable text.



### Q2. What chunking strategy was used?

**Answer:**  
I  implemented a structure-aware hybrid chunking strategy. The text
was first split by section markers (e.g., অধয্ায়'', “Chapter”, “Example”). Longer segments
exceeding 512 characters were further tokenized into sentences using regex-based Bangla or
English splitters. Sentence groups were formed up to the 512-character threshold. This
approach ensures contextually meaningful yet model-compliant input for embedding and
retrieval, minimizing semantic drift across chunks.

---

### Q3. What embedding model was used and why?

**Answer:**  
I used `paraphrase-multilingual-MiniLM-L12-v2`from Sentence Transform-ers. 
It supports both Bangla and English and provides dense, semantic embeddings ideal for
multilingual tasks. This model captures paraphrastic similarity and sentence-level meaning
effectively, making it suitable for both factual and abstract semantic search across mixed-language chunks.
---

### Q4. How is query matching performed?

**Answer:**  
I first retrieve candidates using FAISS-based IndexHNSWFlat on normalized
embeddings `(cosine similarity)`, allowing fast approximate nearest neighbour search.
Top results are then reranked using CrossEncoder `(ms-marco-MiniLM-L-6-v2)`, which evaluates pairwise relevance between the user query and retrieved chunks. This two-step 
`(dense + reranker)` approach balances speed and precision.

---

### Q5. How are vague or irrelevant queries handled?

**Answer:**  
! LLM give wrong ans that why i create some hard query 

I normalize embeddings before similarity search and use reranking with a cross-encoder for accurate scoring. We also apply Maximal Marginal Relevance (MMR) to balance
diversity and relevance in retrieved chunks. If the user query is vague, the semantic similarity
scores will be low across the board. In such cases, we return fallback messages like “আিম জািন
না । ” or “Please clarify your question,” minimising misleading outputs.

---

### Q6. How relevant are the answers, and how could they be improved?

**Answer:**  
Relevance is generally high. Improvements could include:

- Fine-tuned domain-specific embeddings  
- Context window expansion  
- Clarification-based follow-up prompting  
- Integration of Bangla language toolkit (bnltk)
- Multilingual Cross-Encoder Reranking

---

## 📦 Requirements Summary

- See `requirements.txt` for Python dependencies  
- See `package.json` for Node.js dependencies

---

## 👥 Credits

**Developed by:** IQBAL HOSSAIN 

**Contact:** [Open an issue] 25ikbalhossain@gmail.com or contact the developer team.
