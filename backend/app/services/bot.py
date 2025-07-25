import os
import re
import groq
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ.pop("SSL_CERT_FILE", None)

from .pdf_handler import PDFKnowledgeBase

client = groq.Client(api_key="REMOVED")

def clean_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text.strip()

def extract_keywords(text, top_k=5):
    vectorizer = TfidfVectorizer(max_features=top_k, stop_words='english')
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

kb = PDFKnowledgeBase()
executor = ThreadPoolExecutor(max_workers=1)
pdf_path = r"C:\Users\25ikb\OneDrive\Desktop\Bangla_chatbot\backend\data\bangla_class11.pdf"

async def load_pdf_async(pdf_path: str):
    loop = asyncio.get_event_loop()
    print("[INFO] Starting async PDF vector store load...")
    await loop.run_in_executor(executor, lambda: kb.load_vector_store(pdf_path))
    print("[INFO] Completed async PDF vector store load.")

def load_pdf_on_startup(pdf_path: str):
    if os.path.exists("index.faiss") and os.path.exists("chunks.json"):
        try:
            kb.load_vector_store(pdf_path)
            print("[INFO] Loaded existing vector store.")
        except Exception as e:
            print(f"[WARN] Failed loading vector store: {e}")
            print("[INFO] Rebuilding vector store from PDF...")
            kb.build_vector_store(pdf_path)
            print("[INFO] Vector store rebuilt and saved.")
    else:
        print("[INFO] Vector store files missing. Building from PDF...")
        kb.build_vector_store(pdf_path)
        print("[INFO] Vector store built and saved.")

def is_kb_ready():
    return bool(kb.text_chunks) and kb.index.is_trained and kb.index.ntotal > 0

async def call_llm(messages, max_tokens=300):
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            stream=True,
            max_tokens=max_tokens
        )
        full_reply = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
        return clean_text(full_reply)
    except Exception:
        print(" Error during Groq LLM call:")
        traceback.print_exc()
        return " Unable to generate answer. Please try again later."

async def generate_response(user_input: str) -> str:
    if not is_kb_ready():
        return "Knowledge base is not ready yet. Please try again later."

    user_input = user_input.strip()
    fallback = "আমি জানি না।"

    # Hardcoded fallback responses
    hardcoded = {
        "তোমার নাম কী": "আমার নাম এআই সহকারী।",
        "তুমি কে": "আমি একটি চ্যাটবট সহকারী, তোমার সাহায্যের জন্য তৈরি।",
        "তুমি কি পারো": "আমি প্রশ্নের উত্তর দিতে পারি যদি তা আমার তথ্যভাণ্ডারে থাকে।",
        "what's your name": "I'm an AI assistant created to help you.",
        "who are you": "I'm your helpful chatbot assistant.",
    }
    if user_input.lower() in hardcoded:
        return hardcoded[user_input.lower()]

    try:
        # Optional: Expand query with extracted keywords
        keywords = extract_keywords(user_input)
        expanded_query = user_input + " " + " ".join(keywords)

        # Retrieve top 50 chunks
        results = kb.query(expanded_query, top_k=50)
        filtered_chunks = [c for c in results if c.strip()]

        # Fallback if all chunks are weak or irrelevant
        if not filtered_chunks or all("আমি জানি না" in c or "I don't know" in c for c in filtered_chunks):
            return fallback

        # Combine retrieved text into context
        context_text = "\n".join(filtered_chunks)
        keyword_line = "মূল শব্দ / Keywords: " + ", ".join(keywords)

        lang_notice = "Respond in the same language as the user's query (Bangla or English). Use the knowledge base only."
        full_context = f"{keyword_line}\n\n{lang_notice}\n\n{context_text}"

        # Primary LLM prompt
        prompt = f"""
You are a helpful assistant. Answer using only the information below. If the answer is not found, reply with "I don't know" or "আমি জানি না।"

Knowledge:
{full_context}

Question: {user_input}

Answer clearly and within 100 words:
"""

        # First-pass LLM response
        initial_answer = await call_llm([
            {"role": "system", "content": "You are a bilingual assistant. Use only the context."},
            {"role": "user", "content": prompt}
        ])

        # Direct return if short answer
        if len(initial_answer.strip()) <= 120:
            return initial_answer.strip()

        # Refinement pass
        refine_prompt = f"""
Refine this answer to be more natural and clear:\n
Q: {user_input}
A: {initial_answer}

Refined:
"""
        refined_answer = await call_llm([
            {"role": "system", "content": "You are a language editor. Use plain and natural text."},
            {"role": "user", "content": refine_prompt}
        ])

        # Log interaction
        with open("logs.txt", "a", encoding="utf-8") as log:
            log.write(f"USER: {user_input}\nINITIAL: {initial_answer}\nREFINED: {refined_answer}\n---\n")

        return refined_answer.strip()

    except Exception:
        print("❌ Error in generate_response():")
        traceback.print_exc()
        return fallback
