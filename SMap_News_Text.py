import requests
from playwright.sync_api import sync_playwright
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from dotenv import load_dotenv
import os
import psycopg2
import re
import calendar

def naver_news_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            page.wait_for_selector('#dic_area', timeout=30000)
            title_el = page.query_selector('h2.media_end_head_headline')
            title_text = title_el.inner_text().strip() if title_el else '제목 없음'
            content_el = page.query_selector('#dic_area')
            content_text = content_el.inner_text().strip() if content_el else '본문 없음'
        except Exception as e:
            print(f"네이버 뉴스와 구조 다름 : {url}") #건너뛰기
            title_text = None
            content_text = None
        browser.close()
        return title_text, content_text
    
keyword_model = KeyBERT(model='all-MiniLM-L6-v2')
sbert = SentenceTransformer('all-MiniLM-L6-v2')

crime_categories = {
    "경제 범죄": ["사기", "보이스피싱", "다단계", "횡령", "유사수신"],
    "폭력/살인 범죄": ["살인", "살인미수", "폭행", "상해", "칼부림"],
    "성범죄": ["성폭행", "강간", "성추행", "몰카", "디지털성범죄", "성매매"],
    "약물 범죄": ["마약", "필로폰", "코카인"],
    "교통 범죄": ["교통사고", "음주운전", "무면허", "뺑소니"],
    "사이버 범죄": ["해킹", "랜섬웨어", "사이버 공격", "피싱"],
    "재난/재해": ["화재", "자연재해", "산사태", "폭풍", "홍수", "지진", "태풍", "정전", "운행정지", "장애", "폭발", "사고"]
}

category_labels = list(crime_categories.keys())
category_keywords = [" ".join(words) for words in crime_categories.values()]
cat_embeddings = sbert.encode(category_keywords, convert_to_tensor=True)

event_terms = set([w for words in crime_categories.values() for w in words])
event_terms.update(["운행정지","운행","중단","재개","사고","조사","폭발","복구","피해","발생","사망","부상"])