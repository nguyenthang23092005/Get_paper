import os
from dotenv import load_dotenv
from api_search import search_openalex,search_crossref
from utils import remove_duplicates_and_convert_doi,save_results_to_json, enrich_with_firecrawl, filter_relevant_papers, innovative_filtered_papers, convert_latest_json_to_gdoc,evaluate_paper_quality,get_top_quality_papers
import datetime


RESULTS_DIR = "results"
DATABASE_DIR = "database"
DATABASE_FILE = "papers_db.json"
ENV_PATH = ".env"

load_dotenv(ENV_PATH)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(ENV_PATH):
    open(ENV_PATH, "a").close()
    

# key_word = "Non-Destructive Testing"
key_word = '"non destructive testing" OR "nondestructive evaluation" OR "ultrasonic testing" OR "eddy current testing"'
number_results = 100

date = datetime.datetime.now().strftime("%Y-%m-%d")

openalex = search_openalex(key_word, rows=number_results, date=date)
crossref = search_crossref(key_word, rows=number_results, date=date)

merged_results = openalex + crossref
merged_results = remove_duplicates_and_convert_doi(merged_results)

print("⏳ Đang bổ sung abstract...")
enriched_results = enrich_with_firecrawl(merged_results)

print("⏳ Đang lọc bài báo...")
relevant_papers = filter_relevant_papers(enriched_results, keywords=key_word)

print("⏳ Đang đánh giá chất lượng bài báo...")
paper_quality = evaluate_paper_quality(relevant_papers,keywords=key_word)

print("⏳ Đang lấy những bài có điểm cao nhất...")
top_quality_papers = get_top_quality_papers(paper_quality)

print("⏳ Đang tìm điểm sáng tạo về phương pháp...")
innovative_results = innovative_filtered_papers(top_quality_papers)

saved_file = save_results_to_json(
    innovative_results,
    output_dir=RESULTS_DIR,
    prefix=f"search_paper"
)

convert_latest_json_to_gdoc()
