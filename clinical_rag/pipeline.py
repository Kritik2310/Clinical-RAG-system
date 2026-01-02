from pypdf import PdfReader
import numpy as np
import re
import ollama
import numpy as np
#import chunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
#loaded the pdf for reading
pdf_path="data/guidelines.pdf"
reader = PdfReader(pdf_path)

#extract text from pdf
full_text = ""
for p in reader.pages:
    text = p.extract_text()
    if text:
        full_text+=text
        
# print("\n --- first 500 charcaters --- \n")
# print(full_text[:500])
##clean text
full_text = full_text.replace("\n","").strip()
# print("---cleaned text---")
# print(full_text[:500])

##create chunker
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 250)
#now the cleaned text to chunks
#now from these keywords chuncking needs to start before these titles everything is noise not for rag 
keywords = "executive summary"
positions = []
start=0    
lower_text = full_text.lower()
while True:
    pos = lower_text.find(keywords,start)
    if pos == -1:
        break
    positions.append(pos)
    start = pos + len(keywords)
    
if len(positions)<2:
    raise ValueError("second occurence of executive summary isnt there!")

start_index=positions[1]
clean_text = full_text[start_index:]

chunks = text_splitter.split_text(clean_text)

# print("\n Total chunks created:",len(chunks))
# print("\n---CHUNK 0 ---\n")
# print(chunks[0])
# print("\n---CHUNK 1 ---\n")
# print(chunks[1])
# print("\n---CHUNK 2 ---\n")
# print(chunks[2])
# print("\n---CHUNK 3 ---\n")
# print(chunks[3])

#now the embedding phase starts here
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings=model.encode(chunks)

# print("no of chunks",len(chunks))
# print("embedding shape:",embeddings.shape)
# print("first 5 values of first embedding:\n", embeddings[0][:5])

##BIAS AND AGE RISK PATTERNS
AGE_BIAS_PATTERNS = [
    "inevitable",
    "natural consequence",
    "normal part of aging",
    "normal part of ageing",
    "elderly people",
    "old people",
    "due to old age",
    "caused by aging",
    "caused by ageing",
    "just aging",
    "just ageing"
]
NEGATION_TERMS = [
    "not",
    "never",
    "no",
    "isn't",
    "is not",
    "was not",
    "are not"
]

#the detector fn of above
def detect_bias(text):
    flags=[]
    lower_text = text.lower()
    
    for p in AGE_BIAS_PATTERNS:
        if p in lower_text:
            negated = False 
            for n in NEGATION_TERMS:
                if f"{n} {p}" in lower_text:
                    negated = True
                    break
            if not negated:
                flags.append(p)
            
    return flags

#cosine-similarity
def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
#ques
question = "What are the risk factors for cognitive decline?"

q_embeddings = model.encode(question)

##compute similarity with all chunks
scores=[]    ##stores how much similar is each chunk to the question

for i,emb in enumerate(embeddings):
    score = cosine_similarity(q_embeddings,emb)   ##check similarity of ques and current emb vector 
    #high score->same dirn low score->unrelated
    scores.append((i,score))
    
scores.sort(key=lambda x: x[1], reverse=True)
#x[1] meaning use similarity score for sorting reverse=True-> highest score first
#top 3 indexes of similar chunks as llm only needs top ones not everything
# for idx,score in scores[:3]:
#     print("\n---MATCH---")
#     print("chunk index:",idx)
#     print("similarity score:",score)
#     print(chunks[idx][:500])

TOP_K = 2
top_chunks = [chunks[idx] for idx, _ in scores[:TOP_K]]

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+',text)

EVIDENCE_KEYWORDS = [
    "risk factor",
    "risk factors",
    "associated with",
    "increased risk",
    "increase the risk",
    "non-modifiable",
    "modifiable",
    "include",
    "includes"
]

##extracting evidence
def extract_evidence(chunks,keywords):
    evidence=[]
    for c in chunks:
        sentences = split_sentences(c)
        for s in sentences:
            risk_guard = any(p in s.lower() for p in ["risk factor","associated with","increased risk","include"])
            if risk_guard:
                cleaned = s.replace(" "," ").strip()
                cleaned = cleaned.replace("\n"," ").replace("- "," ")
                if len(cleaned) > 40:
                    evidence.append(cleaned)
                    
    seen = set()
    unique_evidence = []
    for e in evidence:
        if e not in seen:
            unique_evidence.append(e)
            seen.add(e)
    return unique_evidence

evidence = extract_evidence(top_chunks,EVIDENCE_KEYWORDS)        
        
##now local llm for text generation
evi_text = "\n".join(f" - {e}" for e in evidence)

SYSTEM_PROMPT = """
        You are a clinical summarization assistant.

        You must ONLY rewrite the provided evidence.
        Do NOT add new information.
        Do NOT use outside knowledge.
        Do NOT infer beyond the evidence.

        If the evidence is insufficient, say:
        "Insufficient evidence to answer the question."

        Your task is to rewrite the evidence into a clear, concise clinical answer.
"""
USER_PROMPT = f""" Evidence : {evi_text} 
              Rewrite the above evidence into consice understandable clinical answer.
              """
              
response = ollama.chat(model="phi3",
                       messages=[{'role':'system','content':SYSTEM_PROMPT},
                                 {'role':'user','content':USER_PROMPT}],
                       options={"temperature":0.1} ##necessary for low creativity 
)


##similarity and cititations about the the answer
def compute_confidence(top):
    if not top:
        return 0.0
    
    k_score = [s for _,s in top]
    
    avg =  np.mean(k_score)
    max_s = max(k_score)
    
    confidence = 0.6*max_s + 0.4*avg
    return round(confidence, 3)

c_score = compute_confidence(scores[:TOP_K])

##biased hiding
for e in evidence:
    flags = detect_bias(e)
    if flags:
        continue
    
bias_flags = detect_bias(response['message']['content'])

def main():
    print("characters extracted", len(full_text))
    print("ques embedding space", q_embeddings.shape)
    print("first 5 values of embedding:\n", q_embeddings[:5])

    print("---evidence-only answer---")
    if not evidence:
        print("insufficient evidence in the retrieved docs.")
    else:
        for i, e in enumerate(evidence, 1):
            print(f"{i}. {e}")

    print("\n ==== FINAL CLINICAL ANSWER FROM THE DOC TO THE QUESTION === \n")
    print(response["message"]["content"])

    print("\nConfidence score:", c_score)

    if not bias_flags:
        print("no age-related bias detected.")
    else:
        print("potential age-related bias detected:")
        for f in bias_flags:
            print("--", f)

if __name__ == "__main__":
    main()