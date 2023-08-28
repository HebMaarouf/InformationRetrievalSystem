from collections import defaultdict

# get query tokens related documents
def get_candidate_documents(query_tokens,inverted_index):
    candidate_documents = set()
    for term in query_tokens:
        if term in inverted_index:
            candidate_documents.update(inverted_index[term])
    return candidate_documents

def map_doc_id_to_index(candidate_documents,data):
    doc_id_to_index = {doc.doc_id: i for i, doc in enumerate(data)}
    return [doc_id_to_index[doc_id] for doc_id in candidate_documents]

def get_related_docs_and_score(sorted_indices,data,cosine_similarities):
    related_docs_and_score = defaultdict(float)
    for doc_index in sorted_indices:
        doc_id = data[int(doc_index)].doc_id
        relevance_score = cosine_similarities[doc_index]
        related_docs_and_score[doc_id] = relevance_score
    return related_docs_and_score
    