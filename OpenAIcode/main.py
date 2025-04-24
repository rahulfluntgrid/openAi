import logging
import os
import uuid
from fastapi import FastAPI, Request, Form, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from cachetools import TTLCache
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect
import asyncio


# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for frequently asked queries (TTL 60 seconds)
cache = TTLCache(maxsize=100, ttl=60)

# Load environment variables for paths (ensure they are set in your environment)
billing_path = r"C:\Users\rahul.k\Desktop\javafor practice\Practice1.java\JAVA program\OpenAIcode\OpenAIcode\master table data.xlsx"
complaints_path = r"C:\Users\rahul.k\Desktop\javafor practice\Practice1.java\JAVA program\OpenAIcode\OpenAIcode\complaints.xlsx"
# Function to load Excel data with error handling
def load_excel_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading file {file_path}")

# Load Excel data with error handling
billing_df = load_excel_data(billing_path)
complaints_df = load_excel_data(complaints_path)

# Load DialoGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Templates directory for frontend (if needed)
templates = Jinja2Templates(directory="templates")

# Global variable for chat history
chat_history = {}

# Function to generate a unique session ID for each request (if not provided)
def get_session_id(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

# Detect language (Hindi or English)
def detect_language(text):
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except:
        return "en"

# Generate LLM response from DialoGPT
def get_local_llm_response(prompt, session_id):
    if session_id not in chat_history:
        chat_history[session_id] = None

    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    if chat_history[session_id] is not None:
        bot_input_ids = torch.cat([chat_history[session_id], inputs], dim=-1)
    else:
        bot_input_ids = inputs

    chat_history[session_id] = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return tokenizer.decode(chat_history[session_id][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# Format response into an HTML table block
def create_html_table(data: dict, title: str, lang: str = "en"):
    rows = "".join([f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in data.items()])
    return f"""
        <div class='response-box'>
            <h3>{title}</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="width: 100%; font-family: Arial, sans-serif;">
                <thead>
                    <tr>
                        <th>{'संपत्ति' if lang == 'hi' else 'Property'}</th>
                        <th>{'मान' if lang == 'hi' else 'Value'}</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    """

# Core query processor
def process_query(query: str, session_id: str) -> str:
    lang = detect_language(query)
    q_lower = query.lower()

    # Check if query is cached
    if query in cache:
        return cache[query]

    if any(word in q_lower for word in ["bill", "amount", "payment", "account", "status"]):
        for acc in billing_df["ACCOUNT_NO"].astype(str):
            if acc.lower() in q_lower:
                data = billing_df[billing_df["ACCOUNT_NO"].astype(str).str.lower() == acc.lower()]
                if not data.empty:
                    details = {
                        "नाम" if lang == "hi" else "Name": data.iloc[0]["NAME"],
                        "उपभोक्ता लोड" if lang == "hi" else "Consumer Load": f"{data.iloc[0]['CONSUMER_LOAD']} kW",
                        "मीटर नंबर" if lang == "hi" else "Meter Number": data.iloc[0]["METER_NUMBER"],
                        "बिलिंग स्थिति" if lang == "hi" else "Billing Status": data.iloc[0]["BILLING_STATUS"],
                        "अंतिम बिल महीना" if lang == "hi" else "Last Bill Month-Year": data.iloc[0]["LAST_BILLMONTH_YEAR"],
                        "बिल राशि" if lang == "hi" else "Bill Amount": f"₹{data.iloc[0]['BILL_AMOUNT']}",
                        "भुगतान राशि" if lang == "hi" else "Amount Paid": f"₹{data.iloc[0]['AMOUNT_PAID']}",
                        "कार्यालय" if lang == "hi" else "Office": data.iloc[0]["OFFICE_NAME"],
                    }
                    title = f"{'खाते ' + acc + ' की जानकारी' if lang == 'hi' else 'Account ' + acc + ' Details'}"
                    response = create_html_table(details, title, lang)
                    cache[query] = response
                    return response
        return "❌ कृपया मान्य खाता संख्या दें।" if lang == "hi" else "❌ Please provide a valid account number."

    elif "complaint" in q_lower or "issue" in q_lower:
        for acc in complaints_df["ACCOUNT_NO"].astype(str):
            if acc.lower() in q_lower:
                data = complaints_df[complaints_df["ACCOUNT_NO"].astype(str).str.lower() == acc.lower()]
                if not data.empty:
                    details = {
                        "शिकायत प्रकार" if lang == "hi" else "Complaint Type": data.iloc[0]["REQUEST_TYPE"],
                        "स्थिति" if lang == "hi" else "Status": data.iloc[0]["APP_STATUS"],
                    }
                    title = "📢 **शिकायत स्थिति**" if lang == "hi" else "📢 **Complaint Status**"
                    response = create_html_table(details, title, lang)
                    cache[query] = response
                    return response
        return "कोई शिकायत नहीं मिली।" if lang == "hi" else "No complaints found for this account."

    # Predefined FAQs
    predefined_responses = {
        "power cut": {
            "hi": "🔌 बिजली रखरखाव के कारण बंद है, 2 घंटे में बहाल की जाएगी।",
            "en": "🔌 Due to maintenance, power will be restored in 2 hours."
        },
        "new connection": {
            "hi": "⚡ नया कनेक्शन के लिए https://www.sbpdcl.co.in पर ऑनलाइन आवेदन करें या नजदीकी विद्युत कार्यालय जाएं।",
            "en": "⚡ To apply for a new connection, visit https://www.sbpdcl.co.in or go to your nearest electricity office."
        },
        "office hours": {
            "hi": "📅 कार्यालय समय: सोमवार से शनिवार, सुबह 10:00 से शाम 5:00 बजे तक। रविवार को अवकाश।",
            "en": "📅 Office Hours: Monday to Saturday, 10:00 AM to 5:00 PM. Closed on Sundays."
        },
        "helpline": {
            "hi": "☎️ हेल्पलाइन: 1912 या 0612-3500120\n📧 ईमेल: support@sbpdcl.co.in",
            "en": "☎️ Helpline: 1912 or 0612-3500120\n📧 Email: support@sbpdcl.co.in"
        },
        "load shedding": {
            "hi": "🕒 लोड शेडिंग आमतौर पर शाम 6-8 बजे होती है। सही जानकारी के लिए क्षेत्रीय कार्यालय से संपर्क करें।",
            "en": "🕒 Load shedding usually occurs between 6-8 PM. For details, contact your local office."
        },
        "tariff": {
            "hi": "💡 घरेलू उपभोक्ताओं के लिए ₹6.10/यूनिट तक दरें लागू हैं। पूरी जानकारी: https://www.sbpdcl.co.in",
            "en": "💡 Domestic consumer rates go up to ₹6.10/unit. Full info: https://www.sbpdcl.co.in"
        },
        "subsidy": {
            "hi": "🎁 अनुसूचित जाति/जनजाति व BPL उपभोक्ताओं को सरकार द्वारा सब्सिडी दी जाती है। विवरण के लिए कार्यालय जाएं।",
            "en": "🎁 Subsidies are available for SC/ST and BPL consumers. Visit your office for details."
        },
        "smart meter": {
            "hi": "📡 स्मार्ट मीटर से आप मोबाइल ऐप से रियल-टाइम उपयोग देख सकते हैं। अधिक जानकारी के लिए sbpdcl.co.in पर जाएं।",
            "en": "📡 Smart meters let you monitor usage via app. Visit sbpdcl.co.in for more info."
        },
        "due date": {
            "hi": "📅 बिल का अंतिम तिथि हर महीने की 15 तारीख होती है।",
            "en": "📅 Bill due date is typically the 15th of every month."
        },
        "pay": {
            "hi": "💳 भुगतान UPI, नेट बैंकिंग, या CSC केंद्र से करें। पोर्टल: https://www.sbpdcl.co.in",
            "en": "💳 Pay via UPI, net banking, or nearby CSC center. Visit https://www.sbpdcl.co.in"
        }
    }

    for key, value in predefined_responses.items():
        if key in q_lower:
            response = value[lang]
            cache[query] = response
            return response

    return get_local_llm_response(query, session_id)

# Endpoint to receive chat queries with rate limiting
@app.post("/chat")
@limiter.limit("5/minute")
async def chat(request: Request, session_id: str = Depends(get_session_id)):
    try:
        data = await request.json()
        query = data.get("query", "")
        logger.info(f"Query received: {query} from session {session_id}")
        return {"response": process_query(query, session_id)}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Optional: Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Optional: Handle chat with form post
@app.post("/chat/")
@limiter.limit("5/minute")
async def chat_post(request: Request, message: str = Form(...), session_id: str = Depends(get_session_id)):
    try:
        reply = process_query(message, session_id)
        return JSONResponse(content={"reply": reply})
    except asyncio.CancelledError:
        logger.error("The task was canceled")
        raise HTTPException(status_code=500, detail="Request was canceled.")

