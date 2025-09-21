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

    # 위치 후보 추출 (본문+제목)
    regex_candidates = []

    doc_full = title + " " + content

    # 기존 후보
    regex_candidates += re.findall(r'[가-힣]{2,}(?:시|도|군|구|읍|면)', doc_full)
    regex_candidates += re.findall(r'[가-힣0-9·\-]{2,}(?:역|정류장|터미널|공항|학교|병원)', doc_full)
    regex_candidates += re.findall(r'([가-힣0-9·\-]{2,})\s*\(', doc_full)

    # 행정동/리 추가
    regex_candidates += re.findall(r'[가-힣]{2,}(?:동|리)', doc_full)

    # 숫자/금액 제외
    regex_candidates = [c for c in regex_candidates if not re.search(r'\d|억원|만원|달러', c)]

    location = "위치 불명"
    if regex_candidates:
        location_seed = "서울 부산 인천 대구 대전 광주 용인 수원 파주 경기 강원 충북 충남 전북 전남 경북 경남 제주 역 정류장 터미널 공항 학교 병원"
        seed_emb = sbert.encode(location_seed, convert_to_tensor=True)
        best_score, best_loc = -1, None
        for c in regex_candidates:
            emb = sbert.encode(c, convert_to_tensor=True)
            sim = float(util.cos_sim(emb, seed_emb)[0][0])
            # 제목 포함 시 보너스 가산
            if c in title:
                sim += 0.25
            if sim > best_score:
                best_score, best_loc = sim, c
        if best_loc:
            location = best_loc

    # 날짜/시간 결합
    final_event_dt = parse_event_datetime(content, news_datetime)
    crime_day = final_event_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 조사 여부
    investigation = any(word in content for word in ["조사", "수사", "국토부", "경찰", "검찰"])

    # 요약
    summary = f"{location}에서 {crime_day} {crime_type}이 발생했다."
    if investigation:
        summary += " 관련 기관에서 조사가 진행 중이다."

    return {
        "범죄유형": crime_type,
        "위치": location,
        "범죄날짜": crime_day,
        "요약내용": summary
    }