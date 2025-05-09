import re
from .config import MIN_CONFIDENCE_THRESHOLD, MIN_CHARS_FOR_PARTIAL_MATCHING

def index_text(text_results: list[tuple[str, tuple[int, int, int, int], float]]):
    """Creates in-memory index mapping words to their locations"""
    index = {}  # word -> list of (page_num, bounding_box, confidence, original_text)
    page_num = 1  # assuming single page for now #TODO
    
    for text, box, confidence in text_results:
        # store original text for exact matching
        original_text = text
        
        # clean and normalize text for indexing
        text = text.lower().strip()
        
        # store both full phrases and individual words
        if text:
            if text not in index:
                index[text] = []
            index[text].append((page_num, box, confidence, original_text))
            
        # index individual words
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word and len(word) > 1:  # only index words with 2+ characters
                if word not in index:
                    index[word] = []
                index[word].append((page_num, box, confidence, original_text))
    
    print(f"Indexing complete. Indexed {len(index)} unique words/phrases.")
    return index


def search_text(query, index):
    """Search with various matching strategies including partial word matching"""
    query = query.lower().strip()
    results = []
    min_confidence = MIN_CONFIDENCE_THRESHOLD
    min_chars = MIN_CHARS_FOR_PARTIAL_MATCHING
    
    # strategy 1: exact phrase match
    if query in index:
        for match in index[query]:
            page_num, box, confidence, original_text = match
            if confidence >= min_confidence:
                results.append((page_num, box, confidence, original_text, 1.0))
    
    # strategy 2: all words in query appear in indexed text
    if not results:
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        for indexed_text in index.keys():
            if len(indexed_text) > 3:
                indexed_words = set(re.findall(r'\b\w+\b', indexed_text.lower()))
                if query_words.issubset(indexed_words):
                    for match in index[indexed_text]:
                        page_num, box, confidence, original_text = match
                        if confidence >= min_confidence:
                            match_score = len(query_words) / len(indexed_words)
                            results.append((page_num, box, confidence, original_text, 0.9 * match_score))
                            
    # strategy 3: individual word matches
    if not results:
        query_words = re.findall(r'\b\w+\b', query.lower())
        for word in query_words:
            if len(word) > 2 and word in index:
                for match in index[word]:
                    page_num, box, confidence, original_text = match
                    if confidence >= min_confidence:
                        match_score = 0.7 * (len(word) / len(query))
                        results.append((page_num, box, confidence, original_text, match_score))
    
    # strategy 4: partial word matching
    if not results and len(query) > min_chars:
        for indexed_word in index.keys():
            if query in indexed_word:
                for match in index[indexed_word]:
                    page_num, box, confidence, original_text = match
                    if confidence >= min_confidence:
                        match_score = 0.5 * (len(query) / len(indexed_word))
                        results.append((page_num, box, confidence, original_text, match_score))
    
    # deduplicate and sort results
    if results:
        unique_results = []
        seen_boxes = set()
        
        for result in results:
            page_num, box, confidence, original_text, match_score = result
            box_key = (box[0], box[1], box[2], box[3])
            
            if box_key not in seen_boxes:
                unique_results.append(result)
                seen_boxes.add(box_key)
                
        unique_results.sort(key=lambda x: (x[4], x[2]), reverse=True)
        return [(r[0], r[1], r[2]) for r in unique_results]
    
    print(f"No results found for query: {query}")
    return []
                
                
                            