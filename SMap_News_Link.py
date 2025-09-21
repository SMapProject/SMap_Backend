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

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

SEARCH_QUERY = "사고" #종류
DISPLAY_COUNT = 1  #개수

def news_link(query, count):
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    params = {"query": query, "display": count, "start": 1, "sort": "sim"}
    response = requests.get(NAVER_NEWS_URL, headers=headers, params=params)
    if response.status_code != 200:
        print(f"NAVER NEWS API 호출 실패 : {response.status_code}")
        return []
    data = response.json()
    return [item.get("link") for item in data.get("items", [])]