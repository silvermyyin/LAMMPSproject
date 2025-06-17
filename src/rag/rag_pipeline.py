from query_translation import translate_query
from query_construction import construct_query
from retriever import retrieve

def rag_pipeline(user_query, translation_method='paraphrase', construction_style='cot', topk=3):
    queries = translate_query(user_query, method=translation_method)
    results = []
    for q in queries:
        search_query = construct_query(q, style=construction_style)
        docs = retrieve(search_query, topk=topk)
        results.extend(docs)
    # 去重
    return list(dict.fromkeys(results))

if __name__ == '__main__':
    user_query = "How to set up an NVT simulation in LAMMPS?"
    context = rag_pipeline(user_query)
    print("\n--- Retrieved Context ---\n")
    for doc in context:
        print(doc[:500], '\n---') 