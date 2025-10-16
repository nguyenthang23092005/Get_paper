import os
import glob
import json
import time
import requests
import re
from dateutil import parser
from datetime import datetime
import pandas as pd
from google.oauth2.service_account import Credentials

from googleapiclient.discovery import build
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import GenerateContentConfig
load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = Client(api_key=GOOGLE_API_KEY)

RESULTS_DIR = "results"
DOCUMENT_ID = "19S3OprOCXXxmo8FjkivBtz_t2t5isenYg-AVqqzA2-U"
creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
SPREADSHEET_ID = "1snMFj6e4X3YUK_48xXJlb8VhLwcS4vSxb69LgoBcDO4"

# Hàm lọc trùng các bài báo tìm được và chuẩn hóa Doi -> Link
def remove_duplicates_and_convert_doi(papers):
    unique_papers = []
    seen_dois = set()
    seen_titles = set()

    for paper in papers:
        doi = paper.get("doi")
        title = paper.get("title", "").lower().strip()

        authors = paper.get("authors", [])
        if isinstance(authors, list):
            paper["authors"] = ", ".join(authors) if authors else "Not Available"

        if doi:
            doi_link = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            if doi not in seen_dois:
                paper["doi"] = doi_link  
                unique_papers.append(paper)
                seen_dois.add(doi)
        elif title and title not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title)

    return unique_papers


# Hàm lấy file trong results
def get_latest_json():
    """
    Lấy file JSON mới nhất theo ngày có dạng: YYYY-MM-DD_search_paper.json
    """
    pattern = os.path.join(RESULTS_DIR, "*_search_paper.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print("⚠️ Không tìm thấy file JSON nào trong thư mục results/")
        return None

    # Lấy ngày từ tên file và chọn ngày mới nhất
    files_with_dates = []
    for f in json_files:
        base = os.path.basename(f)
        try:
            date_part = base.split("_")[0]  # Lấy phần YYYY-MM-DD
            datetime.strptime(date_part, "%Y-%m-%d")  # kiểm tra format
            files_with_dates.append((date_part, f))
        except Exception:
            continue

    if not files_with_dates:
        print("⚠️ Không tìm thấy file JSON hợp lệ theo ngày")
        return None

    # Chọn file có ngày mới nhất
    latest_file = max(files_with_dates, key=lambda x: x[0])[1]
    print(f"📂 File JSON mới nhất theo ngày: {latest_file}")
    return latest_file

# Hàm lưu file vào results
def save_results_to_json(data, output_dir=RESULTS_DIR, prefix="search_paper"):
    """
    Lưu kết quả vào file JSON với tên chứa timestamp.
    Nếu cùng 1 ngày đã có file -> load dữ liệu cũ, merge thêm dữ liệu mới (lọc trùng theo DOI hoặc title), rồi ghi đè lại.
    """
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")

    # --- 1️⃣ Tìm file trong ngày hôm nay ---
    existing_file = None
    for fname in os.listdir(output_dir):
        if fname.startswith(today_str) and fname.endswith(f"{prefix}.json"):
            existing_file = os.path.join(output_dir, fname)
            break

    # --- 2️⃣ Đọc dữ liệu cũ nếu có ---
    merged_data = []
    if existing_file:
        try:
            with open(existing_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            merged_data = old_data
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file cũ {existing_file}: {e}")
            merged_data = []
    else:
        filename = f"{today_str}_{prefix}.json"
        existing_file = os.path.join(output_dir, filename)

    # --- 3️⃣ Tạo set chứa DOI hoặc title để kiểm tra trùng ---
    existing_keys = set()
    for item in merged_data:
        key = (item.get("doi") or item.get("title") or "").strip().lower()
        if key:
            existing_keys.add(key)

    # --- 4️⃣ Lọc các bài mới chưa có trong existing_keys ---
    new_filtered = []
    for p in data:
        key = (p.get("doi") or p.get("title") or "").strip().lower()
        if key and key not in existing_keys:
            new_filtered.append(p)
            existing_keys.add(key)

    # --- 5️⃣ Nếu không có bài mới thì bỏ qua ---
    if not new_filtered:
        print("⏩ Không có dữ liệu mới để thêm.")
        return existing_file

    # --- 6️⃣ Gộp và ghi lại ---
    merged_data.extend(new_filtered)
    try:
        with open(existing_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã cập nhật file: {existing_file} (thêm {len(new_filtered)} bài báo mới)")
        return existing_file
    except Exception as e:
        print(f"❌ Lỗi khi lưu file JSON: {e}")
        return None


# Hàm định nghĩa GG Docx
def get_creds():
    scopes = ["https://www.googleapis.com/auth/documents"]

    # Dành cho GitHub Actions
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        return Credentials.from_service_account_file(creds_path, scopes=scopes)

    # Dành cho chạy local
    local_path = r"D:\GitHub\Key_gg_sheet\eternal-dynamo-474316-f6-382e31e4ae72.json"
    if os.path.exists(local_path):
        return Credentials.from_service_account_file(local_path, scopes=scopes)

    raise FileNotFoundError("❌ Không tìm thấy file Google credential nào hợp lệ.")

# Hàm thực hiện ghi dữ liệu lên GG Docx
def append_json_to_gdoc(df, date_str):
    creds = get_creds()
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=DOCUMENT_ID).execute()
    content = doc.get('body', {}).get('content', [])

    # --- 1️⃣ Tìm vị trí của phần "Ngày <date_str>" ---
    target_header = f"Ngày {date_str}"
    today_start = today_end = None

    for i, element in enumerate(content):
        text_elements = element.get("paragraph", {}).get("elements", [])
        if not text_elements:
            continue
        text = "".join([e.get("textRun", {}).get("content", "") for e in text_elements])
        if target_header.strip() in text.strip():
            today_start = element.get("startIndex", 1)
            # tìm endIndex của phần hôm nay = vị trí "Ngày " kế tiếp
            for j in range(i + 1, len(content)):
                next_text_elements = content[j].get("paragraph", {}).get("elements", [])
                next_text = "".join([e.get("textRun", {}).get("content", "") for e in next_text_elements])
                if next_text.startswith("Ngày "):  # phần ngày khác (hôm sau hoặc hôm qua)
                    today_end = content[j].get("startIndex", content[j].get("endIndex"))
                    break
            # nếu không có ngày kế tiếp → phần này kéo tới cuối file
            if not today_end:
                today_end = content[-1].get("endIndex")
            break

    requests = []

    # --- 2️⃣ Nếu tìm thấy "Ngày hôm nay" → xóa ---
    if today_start and today_end:
        print(f"🗑️ Đã tìm thấy 'Ngày {date_str}', tiến hành ghi đè.")
        delete_request = [{
            "deleteContentRange": {
                "range": {"startIndex": today_start, "endIndex": today_end - 1}
            }
        }]
        service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": delete_request}).execute()
        insert_index = today_start
    else:
        # nếu chưa có ngày hôm nay → thêm mới ở cuối
        insert_index = content[-1].get("endIndex", 1) - 1
        print(f"🆕 Không tìm thấy 'Ngày {date_str}', thêm mới vào cuối tài liệu.")

    # --- 3️⃣ Tạo tiêu đề ngày ---
    day_header = f"\nNgày {date_str}\n\n"
    requests.append({
        "insertText": {"location": {"index": insert_index}, "text": day_header}
    })
    requests.append({
        "updateTextStyle": {
            "range": {"startIndex": insert_index, "endIndex": insert_index + len(day_header)},
            "textStyle": {
                "bold": True,
                "foregroundColor": {"color": {"rgbColor": {"red": 1, "green": 0, "blue": 0}}}
            },
            "fields": "bold,foregroundColor"
        }
    })

    current_index = insert_index + len(day_header)

    # --- 4️⃣ Ghi nội dung từng bài báo ---
    for i, row in df.iterrows():
        title = row.get("title", "Không có tiêu đề")
        authors = row.get("authors", "Không rõ tác giả")
        pubdate = row.get("pubdate", "Không rõ ngày")
        innovative = row.get("innovative", "").strip()
        journal = row.get("journal", "Không rõ tạp chí")
        doi = row.get("doi", "")
        link = row.get("link", "")
        hyperlink = doi if doi else link

        text_block = (
            f"{i+1}. {title}\n"
            f"Tác giả: {authors}\n"
            f"Tạp chí: {journal}\n"
            f"Ngày xuất bản: {pubdate}\n"
            f"Innovative: {innovative}\n"
        )

        # thêm mô tả
        requests.append({
            "insertText": {"location": {"index": current_index}, "text": text_block}
        })

        title_start = current_index + len(f"{i+1}. ")
        title_end = title_start + len(title)

        # làm đậm tiêu đề
        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": title_start, "endIndex": title_end},
                "textStyle": {"bold": True},
                "fields": "bold"
            }
        })

        current_index += len(text_block)

        # thêm “Xem bài báo” nếu có link
        if hyperlink:
            link_text = "Xem bài báo\n\n"
            requests.append({
                "insertText": {"location": {"index": current_index}, "text": link_text}
            })
            requests.append({
                "updateTextStyle": {
                    "range": {
                        "startIndex": current_index,
                        "endIndex": current_index + len("Xem bài báo")
                    },
                    "textStyle": {
                        "link": {"url": hyperlink},
                        "bold": True,
                        "foregroundColor": {"color": {"rgbColor": {"blue": 1.0}}}
                    },
                    "fields": "link,bold,foregroundColor"
                }
            })
            current_index += len(link_text)
        else:
            requests.append({
                "insertText": {"location": {"index": current_index}, "text": "\n"}
            })
            current_index += 1

    # --- 5️⃣ Gửi toàn bộ request cập nhật ---
    service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": requests}).execute()

    print(f"✅ Đã ghi đè nội dung của ngày {date_str} (các ngày khác được giữ nguyên).")


# Hàm ghi dữ liệu lên GG Docx
def convert_latest_json_to_gdoc():
    latest_file = get_latest_json()  
    if not latest_file:
        print("⚠️ Không tìm thấy file JSON.")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    base = os.path.basename(latest_file)
    file_date = base.split("_")[0]

    if file_date != today_str:
        print("ℹ️ File JSON mới nhất không phải của hôm nay.")
        return

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    append_json_to_gdoc(df, today_str)


# Hàm thực hiện lấy Abstract và Pubdate 
def fetch_abstract_and_pubdate_firecrawl(url):
    if not FIRECRAWL_API_KEY:
        raise ValueError("Thiếu FIRECRAWL_API_KEY, hãy set trong biến môi trường.")

    api_url = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {
        "url": url,
        "formats": ["markdown"],  
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[Firecrawl Error] {e}")
        return {"abstract": "Not Available", "pubdate": "Not Available"}

    content = data.get("data", {}).get("markdown", "")
    if not content:
        return {"abstract": "Not Available", "pubdate": "Not Available"}

    lines = content.splitlines()
    abstract_lines = []
    capture = False

    for line in lines:
        low = line.lower().strip()
        # Bắt đầu từ Abstract / Tóm tắt
        if "abstract" in low or "tóm tắt" in low:
            capture = True
            continue
        # Nếu gặp Keywords / Introduction thì dừng lại
        if capture and ("keywords" in low or "introduction" in low or "references" in low):
            break
        if capture:
            abstract_lines.append(line.strip())

    abstract = " ".join(abstract_lines).strip()
    
    # --- Trích xuất pubdate từ markdown ---
    pubdate = "Not Available"
    # tìm các mẫu như "Published: 2025-01-20" hoặc "Ngày xuất bản: 20 Jan 2025"
    date_patterns = [
        r'published[:\s]+(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'published[:\s]+(\d{1,2}\s\w+\s\d{4})', # DD Month YYYY
        r'ngày xuất bản[:\s]+(\d{1,2}\s\w+\s\d{4})',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                pubdate = str(parser.parse(match.group(1)).date())
                break
            except:
                continue
    return {"abstract": abstract, "pubdate": pubdate}


# Hàm lấy Abstract và Pubdate 
def enrich_with_firecrawl(results):
    for paper in results:
        needs_fetch = (
            (not paper.get("abstract") or paper["abstract"] == "Not Available") or
            (not paper.get("pubdate") or paper["pubdate"] == "Not Available")
        )
        url = paper.get("link") or paper.get("doi") or paper.get("landing_page") or "Not Available"
        if needs_fetch and url != "Not Available":
            print(f"Fetching abstract & pubdate with Firecrawl for: {paper['title']}")
            data = fetch_abstract_and_pubdate_firecrawl(url)
            paper["abstract"] = data["abstract"]
            paper["pubdate"] = data["pubdate"]
            time.sleep(6)
    return results


# Hàm thực hiện đánh giá liên quan
def evaluate_paper_relevance(abstract, keywords):
    prompt = f"""
    You are a senior researcher specialized in Non-Destructive Testing (NDT), Pulsed Eddy Current (PEC), 
    and related electromagnetic or signal processing methods in engineering.

    Task:
    1. Carefully read the abstract below.
    2. Think step by step (silently) about whether the abstract’s main topic, method, or application 
    is conceptually or methodologically related to any of the following:
    - {", ".join(keywords)}
    - Non-Destructive Testing (NDT)
    - Pulsed Eddy Current (PEC)
    3. If there is at least one clear or close relation, answer "YES". 
    If there is no clear connection at all, answer "NO".
    4. Double-check your decision consistency before finalizing.

    IMPORTANT:
    - Think internally, but output ONLY one word: YES or NO.
    - Do not provide explanation or reasoning.

    Example guidance:
    Abstract: "This paper applies pulsed eddy current sensing to detect corrosion in pipelines."
    → Output: YES

    Abstract: "This study uses seismic inversion for subsurface mapping."
    → Output: NO

    Now analyze the following abstract:
    {abstract}
    """


    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(temperature=0)
        )

        text = response.text.strip().upper()
        related = "YES" in text
        return {"related": related}

    except Exception as e:
        print(f"[Gemini Error - Relevance Evaluation] {e}")
        return {"related": False}



# Hàm lọc các bài báo không liên quan liên quan
def filter_relevant_papers(results, keywords):
    relevant_papers = []

    for paper in results:
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "").strip()

        if not abstract or abstract.lower() == "not available":
            continue

        print(f"Checking relevance for: {title}")
        evaluation = evaluate_paper_relevance(abstract, keywords)

        if evaluation["related"]:
            relevant_papers.append(paper)
        else:
            print(f"❌ Paper '{title}' is not relevant.")

        time.sleep(4)  

    return relevant_papers


# Hàm lấy ranking của journal 
def get_journal_rank(journal_name):
    if not journal_name:
        return None
    try:
        url = f"https://api.openalex.org/sources?search={journal_name}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "results" in data and data["results"]:
            return data["results"][0].get("best_quartile", "Unknown")
    except Exception as e:
        print(f"[OpenAlex Error] {e}")
    return None


# Hàm lấy h-index của author
def get_author_hindex(author_name):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={author_name}&limit=1"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "data" in data and data["data"]:
            author_id = data["data"][0]["authorId"]
            detail = requests.get(
                f"https://api.semanticscholar.org/graph/v1/author/{author_id}?fields=hIndex",
                timeout=10
            ).json()
            return detail.get("hIndex")
    except Exception as e:
        print(f"[SemanticScholar Error] {e}")
    return None


# Hàm đánh giá chất lượng bài báo, jounal và arthur bằng llm
def evaluate_paper_quality_llm(abstract):
    prompt = f"""
    You are a senior reviewer for high-impact engineering journals.

    Read the following abstract and rate its RESEARCH QUALITY from 0–100,
    based on:
    - Novelty and originality of the idea.
    - Technical depth and clarity of methods.
    - Contribution and scientific value.
    - Writing clarity and coherence.

    Output strictly as: "SCORE: <number>"

    Abstract:
    {abstract}
    """
    time.sleep(4)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(temperature=0)
        )
        text = response.text.strip().upper()
        match = re.search(r"SCORE[:\s]*([0-9]+)", text)
        return int(match.group(1)) if match else 50
    except Exception as e:
        print(f"[Gemini Error - LLM Quality] {e}")
        return 50
    


# Hàm đánh giá chất lượng
def evaluate_paper_quality(article, keywords):
    """
    Đánh giá tổng thể chất lượng bài báo gồm:
    - Journal rank (25%)
    - Author h-index (25%)
    - LLM đánh giá nội dung (50%)
    """

    if isinstance(article, list):
        return [evaluate_paper_quality(a, keywords) for a in article]

    title = article.get("title", "Untitled")
    abstract = article.get("abstract", "").strip()
    authors = article.get("authors", "Not Available")
    journal = article.get("journal", "")

    # ⚠️ Không có abstract
    if not abstract or abstract.lower() == "not available":
        article.update({
            "related": False,
            "score": 0,
            "evaluation": "❌ Không có abstract để đánh giá.",
            "quality_level": "⚠️ Thiếu dữ liệu"
        })
        return article

    comments = []

    # --- 2️⃣ Journal (25%) ---
    journal_rank = get_journal_rank(journal)
    journal_score_map = {
        "Q1": 100, "Q2": 80, "Q3": 60, "Q4": 40,
        "A*": 100, "A": 90, "B": 70, "C": 50
    }
    raw_journal_score = journal_score_map.get(journal_rank, 30)
    journal_score = raw_journal_score * 0.25 / 100 * 100  # quy đổi ra %
    comments.append(f"🏫 Journal '{journal}' rank {journal_rank or 'Unknown'} (+{journal_score:.1f} điểm).")

    # --- 3️⃣ Author (25%) ---
    if authors == "Not Available":
        author_score = 0
        comments.append("👤 Không có thông tin tác giả (0 điểm).")
    else:
        first_author = authors.split(",")[0].strip()
        h_index = get_author_hindex(first_author)
        if h_index is None:
            author_score = 5
            comments.append(f"👤 Không lấy được h-index của {first_author} (+5 điểm tạm).")
        elif h_index >= 40:
            author_score = 25
            comments.append(f"👤 {first_author} có h-index cao ({h_index}) (+25 điểm).")
        elif h_index >= 20:
            author_score = 18
            comments.append(f"👤 {first_author} có h-index trung bình ({h_index}) (+18 điểm).")
        else:
            author_score = 10
            comments.append(f"👤 {first_author} có h-index thấp ({h_index}) (+10 điểm).")

    # --- 4️⃣ LLM (50%) ---
    llm_raw = evaluate_paper_quality_llm(abstract)
    llm_score = llm_raw * 0.5 / 100 * 100
    comments.append(f"🤖 LLM đánh giá chất lượng abstract {llm_raw}/100 (+{llm_score:.1f} điểm).")

    # --- 5️⃣ Tổng điểm ---
    total_score = round(journal_score + author_score + llm_score, 2)
    total_score = min(total_score, 100)

    # --- 6️⃣ Xếp loại ---
    quality_level = (
        "🏅 Xuất sắc" if total_score >= 85 else
        "👍 Tốt" if total_score >= 70 else
        "⚖️ Trung bình" if total_score >= 50 else
        "⚠️ Yếu"
    )

    # --- 7️⃣ Cập nhật kết quả ---
    article.update({
        "related": True,
        "journal_rank": journal_rank or "Unknown",
        "score": total_score,
        "evaluation": " | ".join(comments),
        "quality_level": quality_level
    })

    return article



# Hàm lấy top các bài báo được đánh giá hay nhất
def get_top_quality_papers(evaluated_papers, top_n=10):
    # Chỉ lấy các bài liên quan và có điểm > 0
    valid_papers = [p for p in evaluated_papers if p.get("related") and p.get("score", 0) > 0]

    # Sắp xếp theo điểm số giảm dần
    sorted_papers = sorted(valid_papers, key=lambda x: x.get("score", 0), reverse=True)

    # Lấy top N bài
    top_papers = sorted_papers[:top_n]

    return top_papers


# Hàm thực hiện phân tích và tìm điểm sáng tạo của bài báo
def innovative_with_genai(abstract):
    prompt = f"""
    You are an experienced academic reviewer evaluating research abstracts.
    Your task is to identify the methodological innovation in the study described below.

    Follow this reasoning process silently before answering:

    Understand what general topic or goal the research addresses (but do not include this in the answer).

    Examine what method, technique, framework, model, or experimental process is used.

    Compare it mentally with common or traditional approaches in the same field.

    Identify what is new, improved, or distinctive about the method or process.

    Then, write the final output clearly and concisely:

    Focus only on the methodological innovation (not findings or topic).

    Write 2–4 sentences in simple academic English.

    Avoid vague terms like “novel,” “innovative,” or “unique” unless you explain how.

    Abstract:
    {abstract}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(temperature=0.3)
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error - Innovation] {e}")
        return "Tìm điểm sáng tạo về phương pháp không thành công"



# Hàm phân tích và tìm điểm sáng tạo của bài báo
def innovative_filtered_papers(filtered_papers):
    for paper in filtered_papers:
        abstract = paper.get("abstract", "").strip()
        title = paper.get("title", "Untitled")
        
        if abstract:
            print(f"Innovating for: {title}")
            paper["innovative"] = innovative_with_genai(abstract)
            time.sleep(8)  

    return filtered_papers
