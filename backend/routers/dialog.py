from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import os
import json
from typing import Dict, Optional, Tuple
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from collections import deque
from PIL import Image, ImageEnhance
import io
import logging
import base64
import pytesseract
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def initialize_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY", "ключ-токе с сайта Google AI Studio")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=google_api_key)

def create_vector_store(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=600, chunk_overlap=80, length_function=len)
    docs = text_splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def create_retriever(vector_store):
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

def extract_text_with_ocr(image: Image.Image) -> str:
    enhancer = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Sharpness(enhancer).enhance(2.0).convert('L')
    text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 6 --oem 3 -c preserve_interword_spaces=1')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_meaningful_text(text: str, min_word_length: int = 5, min_meaningful_words: int = 10) -> bool:
    if not text:
        return False
    words = re.findall(r'\b\w{3,}\b', text)
    medical_keywords = ['заключение', 'результат', 'анализ', 'диагноз', 'исследование', 'пациент', 'врач', 'дата', 'рекомендация', 'лечение', 'беременность', 'ультразвук']
    meaningful_words = [word for word in words if len(word) >= min_word_length]
    has_medical_terms = any(keyword in text.lower() for keyword in medical_keywords)
    return len(meaningful_words) >= min_meaningful_words or has_medical_terms

def process_text_with_gemini(ocr_text: str, user_text: str, llm) -> str:
    prompt = f"""
Пользователь предоставил медицинский документ и задал вопрос: 
"{user_text}"

Текст документа, распознанный системой:
---
{ocr_text}
---

Ваша задача:
1. Проанализировать документ в контексте вопроса пользователя
2. Выделить ключевую медицинскую информацию
3. Дать четкий ответ на вопрос, опираясь ТОЛЬКО на содержимое документа
4. Если в документе недостаточно информации, сообщить об этом
5. Ответ на русском языке, профессиональный, но понятный
6. Напомнить о необходимости консультации с врачом
"""
    if not user_text:
        prompt = prompt.replace('и задал вопрос: \n"{user_text}"\n', '')
    response = llm.invoke(prompt)
    answer = response.content.strip()
    if "врач" not in answer.lower() and "консультац" not in answer.lower():
        answer += "\n\n⚠️ Важно: Обратитесь к врачу для точной диагностики и лечения."
    return answer

class ConversationHistory:
    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        return "\n".join(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in self.history)

def generate_response(query: str, retriever, llm, conversation_history: ConversationHistory) -> str:
    qa_prompt_template = """
Role:Ты — тёплый и поддерживающий помощник по планированию беременности. 
Отвечай только на основе документа и истории переписки. 
Если в документе нет информации, говори: «У меня нет конкретной информации, но...» — и давай полезный ответ. 
Используй историю для персонализации, обращайся к пользователю мягко, веди беседу естественно. 
Всегда сохраняй доброжелательный тон.

Conversation History:
{history}

Document Context:
{context}

Question: {question}

Answer:
"""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "Контекст не найден."
    prompt = PromptTemplate(template=qa_prompt_template, input_variables=["history", "context", "question"])
    history_context = conversation_history.get_context()
    formatted_prompt = prompt.format(history=history_context, context=context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip()

def generate_simple_response(query: str, llm, conversation_history: ConversationHistory) -> str:
    prompt_template = """
Role:Ты заботливый помощник по планированию беременности. 
Отвечай тёпло, с поддержкой. 
Помогай на всех этапах, основываясь на предыдущем разговоре. 
Не повторяй приветствие.

Conversation History:
{history}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question"])
    history_context = conversation_history.get_context()
    formatted_prompt = prompt.format(history=history_context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip()

class DialogRouter:
    def __init__(self):
        self.llm = initialize_llm()
        self.vector_store = create_vector_store("/home/user/HpProject/documents/copy.txt")
        self.retriever = create_retriever(self.vector_store)
        self.conversation_history = ConversationHistory(max_history=10)

    async def handle_message(self, message: str, image: Optional[UploadFile] = None) -> Dict:
        try:
            message_dict = json.loads(message)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Некорректный формат сообщения")
        action = message_dict.get("action", "dialog")
        user_input = message_dict.get("content", "").strip()
        if action == "important_question":
            if not user_input:
                raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
            answer = generate_response(user_input, self.retriever, self.llm, self.conversation_history)
            self.conversation_history.add_message("user", user_input)
            self.conversation_history.add_message("assistant", answer)
            return {"response": answer, "next_action": "dialog"}
        elif action == "upload_image":
            if not image:
                raise HTTPException(status_code=400, detail="Изображение не предоставлено")
            image_data = await image.read()
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            ocr_text = extract_text_with_ocr(image_pil)
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            answer = process_text_with_gemini(ocr_text, user_input, self.llm) if is_meaningful_text(ocr_text) else "Не удалось распознать текст. Загрузите четкое изображение документа."
            self.conversation_history.add_message("user", user_input or "Подано изображение")
            self.conversation_history.add_message("assistant", answer)
            return {"response": answer, "image": image_base64, "next_action": "dialog"}
        else:
            if not user_input:
                raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
            answer = generate_simple_response(user_input, self.llm, self.conversation_history)
            self.conversation_history.add_message("user", user_input)
            self.conversation_history.add_message("assistant", answer)
            return {"response": answer, "next_action": "dialog"}

dialog_router = DialogRouter()

@router.post("/dialog")
async def dialog_endpoint(message: str = Form(...), image: Optional[UploadFile] = File(None)):
    return await dialog_router.handle_message(message, image)

@router.get("/model_status")
async def model_status():
    return {
        "huggingface_authenticated": bool(os.getenv("HUGGINGFACE_TOKEN")),
        "model_used": "gemini-1.5-flash-latest",
        "transformers_version": transformers.__version__,
        "pytorch_version": torch.__version__,
        "gpu_available": torch.cuda.is_available(),
        "message": "System available with Gemini-based text processing"
    }