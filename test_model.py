from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

prompt = """
You are an AI assistant that provides clear, detailed answers.
Based on the following document, answer the question in full sentences:

Document:
The Iowa Department of Revenue requires residents to file a 1040 form for annual tax returns. Electronic filing is encouraged. State taxes are due by April 30th each year.

Question: What form do I need to file my taxes?
Answer:
"""

result = qa_pipeline(prompt, max_new_tokens=300, temperature=0.7, do_sample=True, return_full_text=True)
print(result[0]["generated_text"].strip())
