from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import torch
from mongodb_interface import MongoDBManager

# Load model
model = SentenceTransformer('./path/to/model')

# Initialize MongoDBManager
mongo_manager = MongoDBManager()

# Load JSON file
with open('./path/to/json/file.json', 'r') as f:
    faq_data = json.load(f)

articles = []
for topic in faq_data:
    for article in topic['articles']:
        articles.append({
            'id': article['id'],
            'topicId': topic['id'],
            'title': article ['title'],
            'body': article['body']   
        })

questions = [faq['title'] for faq in articles]
answers = [faq['body'] for faq in articles]

question_embeddings = model.encode(questions, convert_to_tensor=True)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def get_most_relevant_questions(query: Query):
    mongo_manager.insert_query(query.question)

    query_embedding = model.encode(query.question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding,question_embeddings)

    top_k = torch.topk(cosine_scores, k=6)

    main_question = {
        'type': 'Main',
        'question': questions[top_k.indices[0][0].item()],
        'answer': answers[top_k.indices[0][0].item()],
        'score': top_k.values[0][0].item()
    }

    related_questions = []
    for idx, score in zip(top_k.indices[0][1:], top_k.values[0][1:]):  # Starting from the second item
        related_questions.append({
            'type': 'Related',
            'question': questions[idx.item()],
            'answer': answers[idx.item()],
            'score': score.item()
        })

    result = [main_question] + related_questions

    return result