import requests

# Open Alex
def decode_openalex_abstract(inverted_index):
    """Chuyển inverted index của OpenAlex thành abstract dạng văn bản."""
    if not inverted_index:
        return None
    # sắp xếp theo vị trí
    words = sorted([(pos, word) for word, positions in inverted_index.items() for pos in positions])
    return " ".join(word for pos, word in words)

def search_openalex(query: str, rows: int = 10, date=None):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": rows,
        "sort": "publication_date:desc"
    }
    if date:
        # lọc theo ngày nếu cần (ví dụ '2025-10-15')
        params["filter"] = f"from_publication_date:{date},to_publication_date:{date}"

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get('results', []):
        title = item.get('title')
        
        # Lấy danh sách tác giả
        authors = []
        for a in item.get('authorships', []):
            author_name = a.get('author', {}).get('display_name')
            if author_name:
                authors.append(author_name)

        abstract = decode_openalex_abstract(item.get('abstract_inverted_index'))
        doi = item.get('doi')
        pub_date = item.get('publication_date')
        journal = item.get('host_venue', {}).get('display_name')

        results.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "doi": doi,
            "publication_date": pub_date,
            "journal": journal
        })

    return results

# CrossRef
def search_crossref(query: str, rows: int = 10, date=None):
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "sort": "published",  # sắp xếp theo ngày xuất bản
        "order": "desc"
    }
    if date:
        # lọc theo ngày (YYYY-MM-DD)
        params["filter"] = f"from-pub-date:{date},until-pub-date:{date}"

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get('message', {}).get('items', []):
        title = item.get('title', [None])[0]  # CrossRef trả về list
        authors = []
        for a in item.get('author', []):
            given = a.get('given', '')
            family = a.get('family', '')
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)
        abstract = item.get('abstract')  
        doi = item.get('DOI')
        pub_date_parts = item.get('published-print') or item.get('published-online')
        pub_date = None
        if pub_date_parts:
            date_parts = pub_date_parts.get('date-parts', [[None]])
            pub_date = "-".join(str(x) for x in date_parts[0] if x is not None)
        journal = item.get('container-title', [None])[0]

        results.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "doi": doi,
            "publication_date": pub_date,
            "journal": journal
        })
    return results

