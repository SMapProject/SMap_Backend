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

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

SEARCH_QUERY = "범죄"
DISPLAY_COUNT = 20

def news_link(query, count):
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    params = {"query": query, "display": count, "start": 1, "sort": "sim"}
    response = requests.get(NAVER_NEWS_URL, headers=headers, params=params)
    if response.status_code != 200:
        print(f"NAVER NEWS API 호출 실패 : {response.status_code}")
        return []
    data = response.json()
    return [item.get("link") for item in data.get("items", [])]

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
            print(f"네이버 뉴스와 구조 다름 : {url}")
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

def parse_event_datetime(content, news_datetime):
    final_dt = news_datetime
    m = re.search(r'(\d{1,2})일\s*(오전|오후)?\s*(\d{1,2})시\s*(\d{1,2})분', content)
    if m:
        day, ampm, hour, minute = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if ampm == "오후" and hour < 12: hour += 12
        if ampm == "오전" and hour == 12: hour = 0
        year, month = final_dt.year, final_dt.month
        last_day = calendar.monthrange(year, month)[1]
        final_dt = final_dt.replace(day=min(day,last_day), hour=hour, minute=minute, second=0, microsecond=0)
        return final_dt
    m_day = re.search(r'(\d{1,2})일', content)
    if m_day:
        day = int(m_day.group(1))
        year, month = final_dt.year, final_dt.month
        last_day = calendar.monthrange(year, month)[1]
        final_dt = final_dt.replace(day=min(day,last_day))
        return final_dt
    m_time = re.search(r'(오전|오후)?\s*(\d{1,2})시\s*(\d{1,2})분', content)
    if m_time:
        ampm, hour, minute = m_time.group(1), int(m_time.group(2)), int(m_time.group(3))
        if ampm == "오후" and hour < 12: hour += 12
        if ampm == "오전" and hour == 12: hour = 0
        final_dt = final_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return final_dt
    return final_dt

def extract_location(title, content):
    doc_full = title + " " + content
    location_candidates = set()
    facility_patterns = [
        r'[가-힣0-9·\-]{2,}(?:역|정류장|터미널|공항|학교|병원|대학교|고등학교|중학교|초등학교)',
        r'[가-힣0-9·\-]{2,}(?:빌딩|타워|센터|프라자|백화점|마트|아파트|오피스텔|주차장|호텔|펜션|관광호텔|리조트)',
        r'[가-힣0-9·\-]{2,}(?:공장|창고|주유소|공원|시장|놀이공원|지하주차장|대형마트|컨벤션센터)',
    ]
    for p in facility_patterns:
        location_candidates.update(re.findall(p, doc_full))
    context_patterns = [
        r'([가-힣0-9·\-]{2,}(?:역|터미널|빌딩|병원|학교|마트|아파트|공원))\s*(?:에서|근처|인근)',
        r'([가-힣0-9·\-]{2,}(?:로|길))\s*(?:에서|근처|인근)'
    ]
    for p in context_patterns:
        location_candidates.update(re.findall(p, doc_full))
    admin_patterns = [
        r'[가-힣]{2,}(?:시|도|군|구|읍|면|동|리)',
        r'[가-힣0-9·\-]{2,}(?:로|길)'
    ]
    for p in admin_patterns:
        location_candidates.update(re.findall(p, doc_full))
    location_candidates.update(re.findall(r'([가-힣0-9·\-]{2,})\s*\(', doc_full))
    location_candidates = {c for c in location_candidates if not re.search(r'\d+원|\d+억|\d+달러|\d+명|\d+세', c)}
    if not location_candidates:
        return "위치 불명"
    scored = []
    priority_suffix = ("빌딩", "타워", "센터", "역", "병원", "마트", "아파트", "학교", "프라자", "호텔", "공항", "터미널")
    admin_suffix = ("시", "도", "군", "구", "읍", "면", "동", "리")
    for c in location_candidates:
        score = 0
        if c in title:
            score += 0.4
        if any(c.endswith(suf) for suf in priority_suffix):
            score += 0.3
        if any(c.endswith(suf) for suf in admin_suffix):
            score += 0.35
        score += len(c) * 0.02
        scored.append((score, c))
    m_context = re.findall(r'([가-힣]{2,}(?:시|도|군|구))\s*(?:에서|일대|지역|주거지)', content)
    for loc in m_context:
        scored = [(s+0.5 if c==loc else s, c) for s,c in scored]
    scored.sort(reverse=True)
    return scored[0][1]

def news_keyword(title, content, news_datetime):
    doc = title + " " + content
    keywords = keyword_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n=12)
    crime_type = "분류 불가"
    for category, kw_list in crime_categories.items():
        for kw, _ in keywords:
            if any(word in kw for word in kw_list):
                crime_type = f"{category} 사건"
                break
        if crime_type != "분류 불가":
            break
    if crime_type == "분류 불가":
        main_kw = keywords[0][0] if keywords else "사건"
        kw_embedding = sbert.encode(main_kw, convert_to_tensor=True)
        cosine_scores = util.cos_sim(kw_embedding, cat_embeddings)[0]
        best_match_idx = int(cosine_scores.argmax())
        crime_type = f"{category_labels[best_match_idx]} 사건"
    location = extract_location(title, content)
    dt = parse_event_datetime(content, news_datetime)
    crime_day = dt.strftime("%Y-%m-%d %H:%M:%S")
    investigation = any(word in content for word in ["조사", "수사", "국토부", "경찰", "검찰"])
    summary = f"{location}에서 {crime_day} {crime_type}이 발생했다."
    if investigation:
        summary += " 관련 기관에서 조사가 진행 중이다."
    return {
        "범죄유형": crime_type,
        "위치": location,
        "범죄날짜": crime_day,
        "요약내용": summary
    }

def save_db(result, news_link):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    insert_sql = """
    INSERT INTO news ("crimeType", "location", "crimeDay", "newsLink", "title")
    VALUES (%s, %s, %s, %s, %s)
    """
    cur.execute(insert_sql, (
        result["범죄유형"],
        result["위치"],
        result["범죄날짜"],
        news_link,
        result["요약내용"]
    ))
    conn.commit()
    cur.close()
    conn.close()

news_links = news_link(SEARCH_QUERY, DISPLAY_COUNT)
for link in news_links:
    title, content = naver_news_text(link)
    if not title or not content:
        continue
    news_datetime = datetime.now()
    result = news_keyword(title, content, news_datetime)
    save_db(result, link)
    print(f"DB 저장 완료: {title}")
