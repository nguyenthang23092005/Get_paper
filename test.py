from api_search import search_openalex,search_crossref
from utils import remove_duplicates_and_convert_doi,get_latest_json,save_results_to_json, convert_latest_json_to_gdoc,get_latest_json
import datetime
# convert_latest_json_to_gdoc()

key_word = "non destructive testing"
number_results = 5

date = datetime.datetime.now().strftime("%Y-%m-%d")

openalex = search_openalex(key_word, rows=number_results, date=date)
crossref = search_crossref(key_word, rows=number_results, date=date)

merged_results = openalex + crossref

for i in merged_results:
    print(i)
