import ir_datasets
from evaluation import calculate_average_precision, calculate_precision
from query_matching import get_candidate_documents, get_related_docs_and_score, map_doc_id_to_index
from tf_idf import vectorize_docs,vectorize_query
from text_processing import process_arabic_text, process_english_text
from collections import defaultdict
from langdetect import detect
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
import pytrec_eval


# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Function to construct the inverted index for docs tokens
inverted_index_as_list = defaultdict(list)
def build_inverted_index(doc_tokens,doc_id):
    for term in doc_tokens:
        if term not in inverted_index_as_list:
            inverted_index_as_list[term] = [doc_id]
        else:
            inverted_index_as_list[term].append(doc_id)

# load the english dataset and search
def search_english():
    try:
        english_dataset = ir_datasets.load("wikiclir/en-simple")
        for document in english_dataset.docs_iter()[:75000]:
            print(" ------ Document id: "+document.doc_id)
            print(" ------ Document title: "+document.title)
            documentData = document.text
            doc_tokens = process_english_text(documentData)
            build_inverted_index(doc_tokens,document.doc_id)

        inverted_index = dict(inverted_index_as_list)
        
        documents_vector = vectorize_docs(english_dataset.docs_iter()[:75000],vectorizer)

        
        related_docs_to_query = defaultdict()
        query_id = 0
        documentQueries = english_dataset.queries_iter()
        for queryDataset in documentQueries:
            query_id = queryDataset.query_id
            print(" ------ Document query query_id: "+queryDataset.query_id)
            print(" ------ Document query title: "+queryDataset.title)
            print(" ------ Document query first_sent: "+ queryDataset.first_sent)
            query = queryDataset.first_sent
            query_tokens = process_english_text(query)

            # get query tokens related documents
            candidate_documents = get_candidate_documents(query_tokens,inverted_index)

            # Map document IDs to their corresponding indices
            candidate_document_indices = map_doc_id_to_index(candidate_documents,english_dataset.docs_iter()[:75000])

            # Slice the document vectors to get the candidate document vectors
            candidate_document_vectors = documents_vector[candidate_document_indices]

            # Vectorize the query
            query_vector = vectorize_query(query,vectorizer)

            # Compute the cosine similarity between the query vector and document vectors
            # cosine_similarities =cosine_similarity(query_vector, candidate_document_vectors).flatten()
            cosine_similarities = candidate_document_vectors.dot(query_vector.T).toarray().flatten()

            # Get the indices of the documents sorted by relevance
            sorted_indices = cosine_similarities.argsort()[::-1]

            # Get documents with relevance score higher than 0.0
            related_docs = get_related_docs_and_score(sorted_indices,english_dataset.docs_iter()[:75000],cosine_similarities)

            #print the result
            for index, (key, value) in enumerate(related_docs.items()):
                print(f"Rank: {index+1}\tDocument ID: {key}\tRelevance Score: {value}")
                if queryDataset.query_id not in related_docs_to_query:
                    related_docs_to_query[queryDataset.query_id] = {key: value }
                else:
                    related_docs_to_query[queryDataset.query_id].update({key: value })
                print(f"Relevant Documents count: {len(related_docs)}")
            break
        evaluator = pytrec_eval.RelevanceEvaluator(english_dataset.qrels_dict(), pytrec_eval.supported_measures)
            
        metrics = evaluator.evaluate(related_docs_to_query)

        query_results = metrics[query_id]

        # Calculate Precision
        precision_at_10 = query_results['P_10']

        # Calculate Recall
        #recall = query_results['recall']

        # Calculate Mean Average Precision (MAP)
        map_score = query_results['map']

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = query_results['recip_rank']
        # Print the evaluation results
        #print(f"Query ID: {query_id}")
        print(f"Precision@10: {precision_at_10}")
        print(f"Recall: N/A")
        print(f"MAP: {map_score}")
        print(f"MRR: {mrr}")
        print("------")
    except:
        print("NO RESULT")


# load the arabic dataset and search
def search_arabic():
    try:
        arabic_dataset = ir_datasets.load("wikiclir/ar")
        for document in arabic_dataset.docs_iter()[:75000]:
            print(" ------ Document id: "+document.doc_id)
            print(" ------ Document title: "+document.title)
            documentData = document.text
            doc_tokens = process_arabic_text(documentData)
            build_inverted_index(doc_tokens,document.doc_id)

        inverted_index = dict(inverted_index_as_list)
        #print(inverted_index)
        documents_vector = vectorize_docs(arabic_dataset.docs_iter()[:75000],vectorizer)

        related_docs_to_query = defaultdict()
        query_id = 0
        documentQueries = arabic_dataset.queries_iter()
        for queryDataset in documentQueries:
            query_id = queryDataset.query_id
            print(" ------ Document query query_id: "+queryDataset.query_id)
            print(" ------ Document query title: "+queryDataset.title)
            print(" ------ Document query first_sent: "+ queryDataset.first_sent)
            query = queryDataset.title + " " + queryDataset.first_sent

            language = detect(query)
            if(language!='ar'):
                translator = Translator(from_lang=language,to_lang="Arabic")
                query = translator.translate(query)
            
            query_tokens = process_arabic_text(query)

            # get query tokens related documents
            candidate_documents = get_candidate_documents(query_tokens,inverted_index)

            # Map document IDs to their corresponding indices
            candidate_document_indices = map_doc_id_to_index(candidate_documents,arabic_dataset.docs_iter()[:75000])
            
            # Slice the document vectors to get the candidate document vectors
            candidate_document_vectors = documents_vector[candidate_document_indices]

            # Vectorize the query
            query_vector = vectorize_query(query,vectorizer)

            # Compute the cosine similarity between the query vector and document vectors
            #cosine_similarities =cosine_similarity(query_vector, candidate_document_vectors).flatten()
            cosine_similarities = candidate_document_vectors.dot(query_vector.T).toarray().flatten()
            # Get the indices of the documents sorted by relevance
            sorted_indices = cosine_similarities.argsort()[::-1]

            # Get documents with relevance score higher than 0.0
            related_docs = get_related_docs_and_score(sorted_indices,arabic_dataset.docs_iter()[:75000],cosine_similarities)

            #print the result
            for index, (key, value) in enumerate(related_docs.items()):
                print(f"Rank: {index+1}\tDocument ID: {key}\tRelevance Score: {value}")
                if queryDataset.query_id not in related_docs_to_query:
                    related_docs_to_query[queryDataset.query_id] = {key: value }
                else:
                    related_docs_to_query[queryDataset.query_id].update({key: value })
                print(f"Relevant Documents count: {len(related_docs)}")
            break
        evaluator = pytrec_eval.RelevanceEvaluator(arabic_dataset.qrels_dict(), pytrec_eval.supported_measures)
            
        metrics = evaluator.evaluate(related_docs_to_query)

        query_results = metrics[query_id]

        # Calculate Precision
        precision_at_10 = query_results['P_10']

        # Calculate Recall
        #recall = query_results['recall']

        # Calculate Mean Average Precision (MAP)
        map_score = query_results['map']

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = query_results['recip_rank']
        # Print the evaluation results
        print(f"Query ID: {query_id}")
        print(f"Precision@10: {precision_at_10}")
        #print(f"Recall: {recall}")
        print(f"MAP: {map_score}")
        print(f"MRR: {mrr}")
        print("------")
    except:
        print("NO RESULT")


search_english()