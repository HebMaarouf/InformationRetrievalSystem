def vectorize_docs(data,vectorizer):

    # Create a list of document contents
    documents = [doc.text for doc in data]

    # Apply the vectorizer to the documents
    document_vectors = vectorizer.fit_transform(documents)

    #print(document_vectors.toarray())
    
    # return the resulting document vectors
    return document_vectors

def vectorize_query(query,vectorizer):

    # Vectorize the query
    query_vector= vectorizer.transform([query])

    # print(query_vector.toarray())

    # return the resulting query vector
    return query_vector