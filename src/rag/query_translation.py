from transformers import pipeline

# Use a paraphrasing model (can be replaced with a local or API model)
paraphraser = pipeline('text2text-generation', model='Vamsi/T5_Paraphrase_Paws')

def translate_query(query, method='paraphrase', n=3):
    if method == 'paraphrase':
        res = paraphraser(query, num_return_sequences=n, max_length=64)
        return [r['generated_text'] for r in res]
    elif method == 'expand':
        # Simple expansion: add synonyms or related terms (placeholder)
        return [query, query + ' in LAMMPS', 'How to ' + query]
    elif method == 'reformulate':
        # Reformulate as a question
        return [f"What is the best way to {query} in LAMMPS?"]
    else:
        return [query] 