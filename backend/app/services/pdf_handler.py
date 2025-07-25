import os
import sys
import json
import fitz               # PyMuPDF
from PyPDF2 import PdfReader
from PIL import Image
import io
import cv2
import numpy as np
import re
import logging

from langdetect import detect
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import easyocr

sys.stdout.reconfigure(encoding='utf-8')

MAX_DISTANCE = 1.0
CHUNK_SIM_THRESHOLD = 0.3
SECTION_MARKERS = [
    "অধ্যায়", "উদাহরণ", "প্রশ্ন", "সংজ্ঞা", "অনুশীলনী", "উপসংহার",
    "Chapter", "Example", "Question", "Definition", "Exercise", "Conclusion"
]
GENERAL_RESPONSES = [
    "আমি জানি না।",
    "আপনার প্রশ্নটি আরো স্পষ্ট করুন।",
    "This question is beyond my knowledge."
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# Pure‐Python sentence splitters (no NLTK)
# ----------------------------------------
def bangla_sent_tokenize(text: str) -> list[str]:
    parts = re.split(r'(?<=[।!?\.])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def english_sent_tokenize(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

# -----------------------------
# MMR: balance relevance & div.
# -----------------------------
def mmr(ranked_ids, sims, lambda_param=0.5, top_k=5):
    selected = []
    candidates = ranked_ids.copy()
    # pick highest‐score as first
    selected.append(candidates.pop(0))
    while len(selected) < top_k and candidates:
        mmr_scores = []
        for c in candidates:
            # relevance = sim(query, doc)
            rel = sims[c]
            # diversity = max(sim(c, s) for s in selected)
            div = max(sims[c] * sims[s] for s in selected)
            mmr_scores.append((lambda_param * rel - (1 - lambda_param) * div, c))
        # pick doc with max MMR
        _, best = max(mmr_scores, key=lambda x: x[0])
        selected.append(best)
        candidates.remove(best)
    return selected

class PDFKnowledgeBase:
    def __init__(self):
        self.reader = easyocr.Reader(['bn', 'en'], gpu=False)
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device='cpu'
        )
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.emb_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.text_chunks: list[str] = []
        self.prev_query_cache: dict[str, list[str]] = {}
        self.id_to_chunk: dict[int, str] = {}

    def load_pdf(self, pdf_path: str) -> str:
        logger.info("Loading PDF...")
        reader = PdfReader(pdf_path)
        raw = ""
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt and len(txt.strip()) > 20:
                raw += txt
            else:
                logger.warning(f"Page {i} too short; OCR...")
                raw += self.ocr_pdf_page(pdf_path, i)
        if len(raw.strip()) < 100:
            raise ValueError("Extraction + OCR failed.")
        return raw

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        gray = np.array(image.convert('L'))
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, binar = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sharp = cv2.filter2D(
            binar, -1,
            np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        )
        return Image.fromarray(sharp)

    def ocr_pdf_page(self, pdf_path: str, pnum: int) -> str:
        page = fitz.open(pdf_path).load_page(pnum)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = self.preprocess_image(img)
        lines = self.reader.readtext(np.array(img), detail=0)
        text = "\n".join([l.strip() for l in " ".join(lines).splitlines() if l.strip()])
        try:
            lang = detect(text)
        except:
            lang = None
        if lang not in ('bn','en'):
            logger.warning(f"Page {pnum} lang '{lang}' unclear.")
        return text

    # ---------------------------------------------------
    # split into sections → cut >512 chars into sentences
    # ---------------------------------------------------
    def structure_aware_split(self, text: str) -> list[str]:
        # by section markers
        chunks, cur = [], []
        for line in text.split('\n'):
            if any(m in line for m in SECTION_MARKERS):
                if cur:
                    chunks.append(" ".join(cur).strip())
                    cur = []
            cur.append(line)
        if cur:
            chunks.append(" ".join(cur).strip())

        final = []
        for c in chunks:
            if len(c) <= 512:
                final.append(c)
                continue
            # sentence‐split then window
            lang = detect(c[:100]) if c else 'en'
            sents = bangla_sent_tokenize(c) if lang == 'bn' else english_sent_tokenize(c)
            buf = ""
            for s in sents:
                if len(buf) + len(s) <= 512:
                    buf += " " + s
                else:
                    final.append(buf.strip())
                    buf = s
            if buf:
                final.append(buf.strip())
        return [c for c in final if len(c) > 20]

    # -----------------------------------------
    # build an HNSW index on normalized vectors
    # -----------------------------------------
    def build_vector_store(self, pdf_path: str):
        raw = self.load_pdf(pdf_path)
        self.text_chunks = self.structure_aware_split(raw)

        # embed + normalize
        embs = self.model.encode(
            self.text_chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype('float32')

        # HNSW
        self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embs)

        # map ID→text
        self.id_to_chunk = {i: txt for i, txt in enumerate(self.text_chunks)}

        # persist
        faiss.write_index(self.index, "index_hnsw.faiss")
        with open("chunks.json","w",encoding='utf-8') as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Stored {len(self.text_chunks)} chunks.")

    def load_vector_store(self, pdf_path=None):
        if not os.path.exists("index_hnsw.faiss") or not os.path.exists("chunks.json"):
            if not pdf_path:
                raise FileNotFoundError("PDF path needed to rebuild.")
            self.build_vector_store(pdf_path)
            return
        self.index = faiss.read_index("index_hnsw.faiss")
        with open("chunks.json","r",encoding='utf-8') as f:
            self.text_chunks = json.load(f)
        self.id_to_chunk = {i:txt for i,txt in enumerate(self.text_chunks)}
        logger.info(f"Loaded {len(self.text_chunks)} chunks.")

    def query(self, user_q: str, top_k=5) -> list[str]:
        key = user_q.strip().lower()
        if key in self.prev_query_cache:
            return self.prev_query_cache[key]

        # encode + normalize
        q_emb = self.model.encode([user_q], normalize_embeddings=True).astype('float32')

        # fetch faster + bigger pool
        D, I = self.index.search(q_emb, top_k*4)
        sims = {i: 1 - d for d, i in zip(D[0], I[0]) if i >= 0}

        # MMR for diversity
        ranked = sorted(sims, key=lambda x: sims[x], reverse=True)
        pick_ids = mmr(ranked, sims, lambda_param=0.7, top_k=top_k*2)

        # cross‐encoder only on half
        cand_pairs = [(user_q, self.id_to_chunk[i]) for i in pick_ids]
        scores = self.cross_encoder.predict(cand_pairs)
        reranked = sorted(zip(pick_ids, scores), key=lambda x: x[1], reverse=True)
        final_ids = [i for i,_ in reranked[:top_k]]
        results = [ self.id_to_chunk[i] for i in final_ids ]

        # cache & return
        self.prev_query_cache[key] = results
        return results

    def safe_print(self, msg: str):
        try:
            print(msg)
        except:
            print(msg.encode('utf-8',errors='replace').decode('utf-8'))