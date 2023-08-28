def calculate_precision(related_docs,candidate_documents):
    return len(related_docs)/len(candidate_documents) if candidate_documents else 0


def calculate_average_precision(query, relevant_documents,inverted_index):
    retrieved_documents = inverted_index.get(query, [])  # Retrieve the documents for the given query

    precision_sum = 0
    num_relevant_documents = len(relevant_documents)

    for i, doc in enumerate(retrieved_documents, 1):
        if doc in relevant_documents:
            precision = len(set(retrieved_documents[:i]) & set(relevant_documents)) / i
            precision_sum += precision

    average_precision = precision_sum / num_relevant_documents if num_relevant_documents != 0 else 0
    return average_precision