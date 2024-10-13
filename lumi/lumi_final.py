# code ollama ทำงานช้า

from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    TextSendMessage, CarouselTemplate, CarouselColumn, 
    URIAction, TemplateSendMessage, MessageAction,
    QuickReply, QuickReplyButton
)
import json
import ollama
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import requests
import threading
import time
import functools


# Neo4j setup
URI = "neo4j://localhost"
AUTH = ("neo4j", "GIft_4438")

# ตรวจสอบการเชื่อมต่อกับ Neo4j
driver = None
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Connected to Neo4j successfully.")
except Exception as e:
    print("Failed to connect to Neo4j:", str(e))

# สร้างโมเดล SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Ollama client setup
ollama_client = ollama.Client(host='http://localhost:11434')

# ฟังก์ชันสำหรับการบันทึก user_id ลงใน Neo4j
def save_user_id(user_id):
    cypher_create_user = '''
    MERGE (u:User {id: $user_id})
    RETURN u
    '''
    try:
        with driver.session() as session:
            session.run(cypher_create_user, user_id=user_id)
            print(f"User {user_id} has been saved or already exists in the database.")
    except Exception as e:
        print(f"Failed to save user {user_id}: {str(e)}")

# ฟังก์ชันสำหรับบันทึกประวัติการสนทนา
def save_chat_history(user_id, question, answer):
    cypher_create_chat = '''
    MATCH (u:User {id: $user_id})
    CREATE (c:ChatHistory {
        timestamp: datetime(),
        question: $question,
        answer: $answer
    })
    CREATE (u)-[:HAS_CHAT]->(c)
    RETURN c
    '''
    try:
        with driver.session() as session:
            session.run(cypher_create_chat, 
                user_id=user_id,
                question=question,
                answer=answer
            )
            print(f"Chat history saved for user {user_id}")
    except Exception as e:
        print(f"Failed to save chat history: {str(e)}")

# ฟังก์ชันสำหรับดึงประวัติการสนทนาของผู้ใช้
def get_user_chat_history(user_id):
    cypher_get_history = '''
    MATCH (u:User {id: $user_id})-[:HAS_CHAT]->(c:ChatHistory)
    RETURN c.timestamp as timestamp, c.question as question, c.answer as answer
    ORDER BY c.timestamp DESC
    LIMIT 10
    '''
    try:
        with driver.session() as session:
            results = session.run(cypher_get_history, user_id=user_id)
            history = [{
                "timestamp": record["timestamp"],
                "question": record["question"],
                "answer": record["answer"]
            } for record in results]
            return history
    except Exception as e:
        print(f"Failed to retrieve chat history: {str(e)}")
        return []

def truncate_text(text, max_length):
    return (text[:max_length - 3] + '...') if len(text) > max_length else text

# ฟังก์ชันสำหรับดึงข้อมูลผลิตภัณฑ์ตามหมวดหมู่เรียงตามราคา
def get_products_by_category_sorted_by_price(category_name):
    query = '''
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name})
    RETURN p.name as product_name, p.price as price, p.UrlPage as url_page, p.image_url as image_url
    ORDER BY p.price ASC
    '''
    try:
        with driver.session() as session:
            results = session.run(query, category_name=category_name)
            products = [{
                "name": record["product_name"], 
                "price": record["price"], 
                "url_page": record["url_page"],
                "image_url": record["image_url"]
            } for record in results]
            if not products:
                print(f"No products found for category: {category_name}")
            return products
    except Exception as e:
        print(f"Failed to retrieve products sorted by price: {str(e)}")
        return []

# ฟังก์ชันสำหรับดึงข้อมูลผลิตภัณฑ์ตามหมวดหมู่เรียงตามรีวิว
def get_products_by_category_sorted_by_reviews(category_name):
    query = '''
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category_name})
    RETURN p.name as product_name, p.reviews as reviews, p.UrlPage as url_page, p.image_url as image_url
    ORDER BY p.reviews DESC
    '''
    try:
        with driver.session() as session:
            results = session.run(query, category_name=category_name)
            products = [{
                "name": record["product_name"], 
                "reviews": record["reviews"], 
                "url_page": record["url_page"],
                "image_url": record["image_url"]
            } for record in results]
            if not products:
                print(f"No products found for category: {category_name}")
            return products
    except Exception as e:
        print(f"Failed to retrieve products sorted by reviews: {str(e)}")
        return []

def get_product_details(product_name):
    query = '''
    MATCH (p:Product {name: $product_name})
    RETURN p.description as description, p.details as details, p.use as use
    '''
    try:
        with driver.session() as session:
            result = session.run(query, product_name=product_name).single()
            if result:
                return {
                    "description": result["description"],
                    "details": result["details"],
                    "use": result["use"]
                }
            return None
    except Exception as e:
        print(f"Failed to retrieve product details: {str(e)}")
        return None

def show_quick_reply_product_details(reply_token, product_name):
    quick_reply_message = [
        TextSendMessage(
            text=f"กรุณาเลือกรายระเอียดที่ต้องการดู {product_name}",
            quick_reply=QuickReply(
                items=[
                    QuickReplyButton(action=MessageAction(label="📋 รายละเอียดของผลิตภัณฑ์", text=f"รายละเอียดของ {product_name}")),
                    QuickReplyButton(action=MessageAction(label="📝 คำอธิบายของผลิตภัณฑ์", text=f"คำอธิบายของ {product_name}")),
                    QuickReplyButton(action=MessageAction(label="🔍 วิธีการใช้งาน", text=f"วิธีการใช้งานของ {product_name}"))
                ]
            )
        )
    ]
    return quick_reply_message

def format_chat_history(history):
    formatted_text = "ประวัติการสนทนา 10 รายการล่าสุด:\n\n"
    for i, chat in enumerate(history, 1):
        timestamp = chat["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        formatted_text += f"{i}. เวลา: {timestamp}\n"
        formatted_text += f"คำถาม: {chat['question']}\n"
        formatted_text += f"คำตอบ: {chat['answer']}\n"
        formatted_text += "-" * 30 + "\n"
    return formatted_text

# เพิ่มฟังก์ชันใหม่สำหรับการอ่านไฟล์ข้อมูล La Mer

def read_la_mer_data():
    neo4j_data = get_neo4j_data()
    txt_data = read_txt_data(r'C:\Users\User\OneDrive\เดสก์ท็อป\code2\final2\data.txt')  # แทนที่ด้วยพาธจริงของไฟล์ .txt
    processed_txt_data = preprocess_txt_data(txt_data)
    return neo4j_data, processed_txt_data
    
def get_neo4j_data():
    cypher_query = '''
    MATCH (n:fq)
    RETURN n.name as name, n.msg_reply as reply
    '''
    neo4j_data = {}
    if driver:
        try:
            with driver.session() as session:
                results = session.run(cypher_query)
                for record in results:
                    neo4j_data[record['name'].lower()] = record['reply']
        except Exception as e:
            print("Failed to run query:", str(e))
    return neo4j_data

def get_relevant_info(question, neo4j_data, txt_products):
    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Combine all data sources
    all_data = list(neo4j_data.items()) + [(f"txt_{i}", txt) for i, txt in enumerate(txt_products)]
    
    # Encode all data
    corpus_embeddings = model.encode([item[1] for item in all_data], convert_to_tensor=True)
    
    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
    
    # Get the most relevant information
    top_result = torch.topk(cos_scores, k=1)
    
    if top_result.values[0] > 0.5:  # Threshold for relevance
        index = top_result.indices[0]
        source = "Neo4j" if index < len(neo4j_data) else "TXT"
        return all_data[index][1], source
    else:
        return None, None

# ฟังก์ชันสำหรับดึงข้อมูลคำทักทายและคำถามจาก Neo4j
def get_greetings_and_questions():
    cypher_query = '''
    MATCH (n:fq) 
    RETURN n.name as name, n.msg_reply as reply
    '''
    corpus = []
    replies = []
    
    if driver:
        try:
            with driver.session() as session:
                results = session.run(cypher_query)
                for record in results:
                    corpus.append(record['name'])
                    replies.append(record['reply'])
        except Exception as e:
            print("Failed to run query:", str(e))
    
    return corpus, replies

# ดึงข้อมูลและสร้าง embeddings
greeting_corpus, greeting_replies = get_greetings_and_questions()
if len(greeting_corpus) > 0:
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
else:
    greeting_vec = None

# ฟังก์ชันสำหรับหาคำตอบที่เหมาะสมที่สุด
def find_best_response(sentence):
    if greeting_vec is None:
        return "ขออภัยครับ ไม่สามารถให้บริการได้ในขณะนี้", 0

    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    
    try:
        greeting_scores = util.cos_sim(greeting_vec, ask_vec)
        greeting_np = greeting_scores.cpu().numpy()
        max_index = np.argmax(greeting_np)
        max_score = greeting_np[max_index][0]  # Get the scalar value

        if max_score > 0.6:
            return greeting_replies[max_index] + " [ตอบจาก Neo4j]", max_score
        else:
            return None, max_score
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการคำนวณ: {str(e)}")
        return None, 0


def read_txt_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_txt_data(data):
    products = data.split('---')
    return [product.strip() for product in products if product.strip()]

def get_greeting_from_neo4j(greeting_text):
    cypher_query = '''
    MATCH (g:Greeting2)
    WHERE toLower(g.name) = toLower($greeting_text)
    RETURN g.msg_reply as reply
    '''
    try:
        with driver.session() as session:
            result = session.run(cypher_query, greeting_text=greeting_text).single()
            if result:
                return result['reply']
    except Exception as e:
        print(f"Failed to retrieve greeting from Neo4j: {str(e)}")
    return None

# เพิ่มฟังก์ชันใหม่สำหรับการสร้างคำตอบโดยใช้ Ollama

@functools.lru_cache(maxsize=100)
def cached_ollama_generate(model, prompt):
    return ollama_client.generate(model=model, prompt=prompt)


def get_ollama_response(access_token, user_id, prompt, context, source):
    try:
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=loading_animation, args=(access_token, user_id, stop_event))
        loading_thread.start()

        full_prompt = f"""ข้อมูลเกี่ยวกับผลิตภัณฑ์ La Mer:

{context}

คำถามของลูกค้า: {prompt}

โปรดตอบคำถามโดยใช้ข้อมูลที่ให้มาข้างต้น ใช้ภาษาที่สุภาพและเป็นมิตร ตอบให้กระชับแต่ครบถ้วน ห้ามเกิน 50 คำ"""

        response = cached_ollama_generate("llama3.2", full_prompt)

        stop_event.set()
        loading_thread.join()

        return response['response'] + f" [ปรับปรุงโดย ollama]"
    except Exception as ollama_error:
        stop_event.set()
        loading_thread.join()
        return "ขออภัยค่ะ เกิดปัญหาในการประมวลผลคำตอบ กรุณาลองถามใหม่อีกครั้งนะคะ"

# ฟังก์ชันสำหรับการประมวลผลคำถาม
def process_question(access_token, user_id, question, neo4j_data, txt_products):
    relevant_info, source = get_relevant_info(question, neo4j_data, txt_products)
    if relevant_info:
        return get_ollama_response(access_token, user_id, question, relevant_info, source)
    else:
        return "ขออภัยค่ะ ฉันไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในฐานข้อมูลของเรา คุณต้องการถามคำถามอื่นหรือให้ฉันช่วยอธิบายเกี่ยวกับผลิตภัณฑ์ La Mer ทั่วๆ ไปไหมคะ?"
    
def start_loading_animation(channel_access_token, user_id):
    url = 'https://api.line.me/v2/bot/chat/loading/start'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }
    data = {
        "chatId": user_id,
        "loadingSeconds": 5  # You can adjust this value as needed
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print("Loading animation started successfully")
        print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to start loading animation: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")


def stop_loading_animation(channel_access_token, user_id):
    url = 'https://api.line.me/v2/bot/chat/loading/stop'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }
    data = {
        'to': user_id
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print("Loading animation stopped successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to stop loading animation: {str(e)}")

def loading_animation(access_token, user_id, stop_event):
    while not stop_event.is_set():
        start_loading_animation(access_token, user_id)
        time.sleep(4.5)  # รอเวลาสั้นๆ ก่อนเริ่ม animation ใหม่
        if not stop_event.is_set():
            stop_loading_animation(access_token, user_id)
        time.sleep(0.5)  # รอเวลาสั้นๆ ก่อนเริ่ม cycle ใหม่

# อ่านข้อมูล La Mer เมื่อเริ่มต้นแอพพลิเคชัน
la_mer_data = read_la_mer_data()
neo4j_data, txt_products = read_la_mer_data()

app = Flask(__name__)

# ฟังก์ชันสำหรับแสดง Quick Reply
def show_quick_reply(reply_token):
    quick_reply_message = [
        TextSendMessage(
            text="เลือกวิธีการเรียงข้อมูล:",
            quick_reply=QuickReply(
                items=[
                    QuickReplyButton(action=MessageAction(label="💰 เรียงตามราคา", text="เรียงตามราคา")),
                    QuickReplyButton(action=MessageAction(label="⭐ เรียงตามรีวิว", text="เรียงตามรีวิว"))
                ]
            )
        )
    ]
    return quick_reply_message
user_selected_category = {}
user_data = {}
def save_user_data(user_id, skin_problem, time_of_day):
    user_data[user_id] = {
        "skin_problem": skin_problem,
        "time_of_day": time_of_day
    }


@app.route("/", methods=['POST'])
def linebot():
    global user_selected_category

    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = '8PaaGz+J0J1mm4VefTzK7ZaUPX+YR7fll5DqhCQaUtkwBNjOMjFNey+Ol+lHNvR4T3A9abZLWJub1407ZWqzP0/CGzVPEGVaKQQA7YuH3Rqq+uIcqlNX3VlioTnWXh1PTHN8VtCjQhjSq7o6pi+EIgdB04t89/1O/w1cDnyilFU='
        secret = '88fdadd23cbb53a21e351a930a30af5f'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']

        handler.handle(body, signature)

        if json_data.get('events'):
            event = json_data['events'][0]
            user_id = event['source']['userId']
            question = event['message']['text']      
            tk = event['replyToken']
            
            # เก็บคำตอบที่จะส่งกลับไป
            answer = None

            # บันทึก user_id
            save_user_id(user_id)

            # ตรวจสอบคำสั่งดูประวัติการสนทนา
            if question.lower() == "ดูประวัติการสนทนา":
                history = get_user_chat_history(user_id)
                answer = format_chat_history(history)
                line_bot_api.reply_message(tk, TextSendMessage(text=answer))
                return 'OK'

            greeting_answer = get_greeting_from_neo4j(question.lower())
            if greeting_answer:
                answer1 = "🌸 สวัสดีค่ะ! ฉันชื่อ LUMI BOT ผู้ช่วยความงามจาก La Mer 🌿✨ ยินดีต้อนรับค่ะ!\n\n💬 ก่อนที่ฉันจะแนะนำผลิตภัณฑ์ อยากทราบข้อมูลเพิ่มเติมจากคุณหน่อยค่ะ"
                answer2 = "1️⃣ คุณมีปัญหาผิวประเภทไหนที่ต้องการแก้ไขบ้างคะ?"
                line_bot_api.reply_message(tk,
                            [TextSendMessage(text=answer1),
                             TextSendMessage(text=answer2)])
                save_chat_history(user_id, question, f"{answer1} {answer2}")
                # เพิ่มข้อมูลผู้ใช้เพื่อติดตามสถานะการสนทนา
                user_data[user_id] = {"state": "waiting_skin_problem"}
                return 'OK'

            # ตรวจสอบว่าผู้ใช้กำลังตอบคำถามเกี่ยวกับปัญหาผิวหรือไม่
            if user_id in user_data and user_data[user_id].get("state") == "waiting_skin_problem":
                user_data[user_id]["skin_problem"] = question
                user_data[user_id]["state"] = "waiting_time_of_day"
                answer = "2️⃣ คุณกำลังมองหาผลิตภัณฑ์เพื่อบำรุงผิวในช่วงเวลาไหนคะ? 🕒"
                line_bot_api.reply_message(tk, TextSendMessage(text=answer))
                save_chat_history(user_id, question, answer)
                return 'OK'

            # ตรวจสอบว่าผู้ใช้กำลังตอบคำถามเกี่ยวกับช่วงเวลาบำรุงผิวหรือไม่
            if user_id in user_data and user_data[user_id].get("state") == "waiting_time_of_day":
                user_data[user_id]["time_of_day"] = question
                user_data[user_id]["state"] = "completed"
                answer4 = "🔎 วิธีการใช้งาน:\n1. เลือกประเภทผลิตภัณฑ์ที่คุณสนใจ 🌟\n2. เลือกวิธีการเรียงข้อมูล 🔢\n3. เลือกรายละเอียดผลิตภัณฑ์ที่ต้องการ 🔍\n4. สอบถามเพิ่มเติมได้ทุกเมื่อค่ะ 💬 ฉันพร้อมให้คำแนะนำเสมอค่ะ ☺️\n"
                answer3 = "ขอบคุณสำหรับข้อมูลนะคะ 💖"
                carousel_columns = [
                    CarouselColumn(
                        thumbnail_image_url=r'https://www.lamer.co.th/media/export/cms/products/responsive/lm_sku_46HM01_4x5_0.png?width=900&height=1125',
                        title="ครีมกลางคืน",
                        text="เลือกครีมกลางคืนที่เหมาะกับผิวของคุณ",
                        actions=[
                            MessageAction(label="product ที่แนะนำ", text="แนะนำครีมกลางคืน")
                        ]
                    ),
                    CarouselColumn(
                        thumbnail_image_url=r'https://www.lamer.co.th/media/export/cms/products/responsive/lm_sku_4J6T01_4x5_0.png?width=900&height=1125',
                        title="ครีมให้ความชุ่มชื้น",
                        text="ครีมให้ความชุ่มชื้นเพื่อผิวที่สดใส",
                        actions=[
                            MessageAction(label="product ที่แนะนำ", text="แนะนำครีมให้ความชุ่มชื้น")
                        ]
                    ),
                    CarouselColumn(
                        thumbnail_image_url=r'https://www.lamer.co.th/media/export/cms/products/responsive/lm_sku_46LJ01_4x5_0.png?width=900&height=1125',
                        title="เซ็ตผลิตภัณฑ์",
                        text="เซ็ตผลิตภัณฑ์เพื่อการดูแลผิวครบวงจร",
                        actions=[
                            MessageAction(label="product ที่แนะนำ", text="แนะนำเซ็ตผลิตภัณฑ์")
                        ]
                    ),
                    CarouselColumn(
                        thumbnail_image_url=r'https://www.lamer.co.th/media/export/cms/products/responsive/lm_sku_5XPX01_4x5_0.png?width=900&height=1125',
                        title="อิมัลชั่น",
                        text="ค้นหาอิมัลชั่นที่ตอบโจทย์ผิวของคุณ",
                        actions=[
                            MessageAction(label="product ที่แนะนำ", text="แนะนำอิมัลชั่น")
                        ]
                    )
                ]

                carousel_template = CarouselTemplate(columns=carousel_columns)
                carousel_message = TemplateSendMessage(
                    alt_text="เลือกผลิตภัณฑ์", 
                    template=carousel_template
                )
                
                line_bot_api.reply_message(
                    tk, 
                    [TextSendMessage(text=answer3),TextSendMessage(text=answer4), carousel_message]
                )
                
                save_chat_history(user_id, question, answer)
                return 'OK'

            elif question in ["แนะนำครีมกลางคืน", "แนะนำครีมให้ความชุ่มชื้น", "แนะนำเซ็ตผลิตภัณฑ์", "แนะนำอิมัลชั่น"]:
                category_map = {
                    "แนะนำครีมกลางคืน": "Night Creams",
                    "แนะนำครีมให้ความชุ่มชื้น": "Moisturizing Creams",
                    "แนะนำเซ็ตผลิตภัณฑ์": "Hydration Sets",
                    "แนะนำอิมัลชั่น": "Emulsions"
                }
                selected_category = category_map.get(question, None)
                
                if selected_category:
                    user_selected_category[user_id] = selected_category
                    answer = f"กรุณาเลือกวิธีการเรียงข้อมูลสำหรับ {selected_category}"
                    line_bot_api.reply_message(tk, show_quick_reply(tk))
                    
                    # บันทึกประวัติการสนทนา
                    save_chat_history(user_id, question, answer)
                return 'OK'

            elif question == "เรียงตามราคา":
                if user_id in user_selected_category:
                    selected_category = user_selected_category[user_id]
                    sorted_products = get_products_by_category_sorted_by_price(selected_category)
                    
                    if sorted_products:
                        carousel_columns = [
                            CarouselColumn(
                                thumbnail_image_url=product['image_url'],
                                title=truncate_text(product['name'], 40),
                                text=f"ราคา: {product['price']} ฿",
                                actions=[
                                    MessageAction(
                                        label="รายละเอียดแยกต่างๆ", 
                                        text=f"ดูรายระเอียด {truncate_text(product['name'], 200)}"
                                    ),
                                    URIAction(
                                        label="ดูบนเว็บไซต์",
                                        uri=product['url_page']
                                    )
                                ]
                            ) for product in sorted_products[:5]
                        ]
                        
                        carousel_template = CarouselTemplate(columns=carousel_columns)
                        carousel_message = TemplateSendMessage(
                            alt_text=f"Products sorted by price in {selected_category}",
                            template=carousel_template
                        )
                        
                        answer = f"แสดงสินค้าในหมวด {selected_category} เรียงตามราคา"
                        line_bot_api.reply_message(tk, carousel_message)
                        
                        # บันทึกประวัติการสนทนา
                        save_chat_history(user_id, question, answer)
                    else:
                        answer = "ไม่พบผลิตภัณฑ์ในหมวดหมู่ที่เลือก"
                        line_bot_api.reply_message(
                            tk, 
                            TextSendMessage(text=answer)
                        )
                        save_chat_history(user_id, question, answer)
                return 'OK'

            elif question == "เรียงตามรีวิว":
                if user_id in user_selected_category:
                    selected_category = user_selected_category[user_id]
                    sorted_products = get_products_by_category_sorted_by_reviews(selected_category)
                    
                    if sorted_products:
                        carousel_columns = [
                            CarouselColumn(
                                thumbnail_image_url=product['image_url'],
                                title=truncate_text(product['name'], 40),
                                text=f"รีวิว: {product['reviews']} รีวิว",
                                actions=[
                                    MessageAction(
                                        label="รายละเอียดแยกต่างๆ", 
                                        text=f"ดูรายระเอียด {truncate_text(product['name'], 200)}"
                                    ),
                                    URIAction(
                                        label="ดูบนเว็บไซต์",
                                        uri=product['url_page']
                                    )
                                ]
                            ) for product in sorted_products[:5]
                        ]
                        
                        carousel_template = CarouselTemplate(columns=carousel_columns)
                        carousel_message = TemplateSendMessage(
                            alt_text=f"Products sorted by reviews in {selected_category}",
                            template=carousel_template
                        )
                        
                        answer = f"แสดงสินค้าในหมวด {selected_category} เรียงตามจำนวนรีวิว"
                        line_bot_api.reply_message(tk, carousel_message)
                        
                        # บันทึกประวัติการสนทนา
                        save_chat_history(user_id, question, answer)
                    else:
                        answer = "ไม่พบผลิตภัณฑ์ในหมวดหมู่ที่เลือก"
                        line_bot_api.reply_message(
                            tk, 
                            TextSendMessage(text=answer)
                        )
                        save_chat_history(user_id, question, answer)
                else:
                    answer = "กรุณาเลือกหมวดหมู่ก่อนทำการเรียงตามรีวิว"
                    line_bot_api.reply_message(
                        tk, 
                        TextSendMessage(text=answer)
                    )
                    save_chat_history(user_id, question, answer)
                return 'OK'

            elif question.startswith("ดูรายระเอียด "):
                product_name = question.replace("ดูรายระเอียด ", "")
                answer = f"เลือกดูข้อมูลของ {product_name}"
                line_bot_api.reply_message(
                    tk,
                    show_quick_reply_product_details(tk, product_name)
                )
                save_chat_history(user_id, question, answer)
                return 'OK'

            elif question.startswith("รายละเอียดของ ") or question.startswith("คำอธิบายของ ") or question.startswith("วิธีการใช้งานของ "):
                is_details = question.startswith("รายละเอียดของ ")
                is_description = question.startswith("คำอธิบายของ ")
                is_usage = question.startswith("วิธีการใช้งานของ ")
                
                product_name = question.replace("รายละเอียดของ ", "").replace("คำอธิบายของ ", "").replace("วิธีการใช้งานของ ", "")
                
                product_info = get_product_details(product_name)
                if product_info:
                    if is_details:
                        answer = f"รายละเอียดของ {product_name}:\n{product_info['details']}"
                    elif is_description:
                        answer = f"คำอธิบายของ {product_name}:\n{product_info['description']}"
                    elif is_usage:
                        answer = f"วิธีการใช้งานของ {product_name}:\n{product_info['use']}"
                    else:
                        answer = f"ไม่พบข้อมูลของ {product_name}"
                    line_bot_api.reply_message(
                    tk,
                    TextSendMessage(text=answer)
                )
                save_chat_history(user_id, question, answer)
                return 'OK'

            # ตรวจสอบคำสั่ง "ตอบกลับด่วน"
            elif question.lower() == "ตอบกลับด่วน":
                answer = process_question(access_token, user_id, question, neo4j_data, txt_products)
                line_bot_api.reply_message(tk, TextSendMessage(text=answer))
                save_chat_history(user_id, question, answer)
                return 'OK'

            # ถ้าไม่ตรงกับเงื่อนไขข้างต้น ใช้ process_question ตอบคำถาม
            else:
                answer = get_ollama_response(access_token, user_id, question, neo4j_data, txt_products)
                line_bot_api.reply_message(tk, TextSendMessage(text=answer))
                save_chat_history(user_id, question, answer)
                return 'OK'

        return 'OK'

    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        return 'Invalid signature', 400
    except Exception as e:
        print("An error occurred:", str(e))
        print("Request Body:", body)
        return 'Error', 500
if __name__ == '__main__':
    app.run(port=5000)
