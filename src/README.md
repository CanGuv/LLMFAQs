# LLMFAQs

In this section, we integrate the fine-tuned model, creating a python API with a MongoDB database. 

## Installation

You will need to install teh required dependencies.

**Clone the repository**:
```bash
git clone https://github.com/CanGuv/LLMFAQs.git
cd LLMFAQs
```
**Set up virtual environment**
```bash
python -m venv venv
```
Activate virtual environment:

macOS:
```bash
source venv/bin/activate
```

Windows:
```bash
.\venv\Scripts\activate
```

**Install the dependencies**
```bash
cd src
pip install -r requirements.txt
```

**Download MongoDB**

macOS: (Download Homebrew)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew tap mongodb/brew
brew install mongodb-community@6.0
```

Windows: (Download Chocolately)
```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

choco install mongodb
```

## Creating database:

Start MongoDB:
```bash
brew services start mongodb/brew/mongodb-community
```

Open MongoDB shell:
```bash
mongosh
```

Create a database:
```bash
use database_name
```

Create a collection:
```bash
db.createCollection('collectionName')
```

To view database (after inserting):
```bash
db.queries.find().pretty()
```

## Required libraries

- **`sentence_transformers`**:
  - **`SentenceTransformer`**: Used for loading pre-trained models and generating embeddings for text data.
  - **`util`**: Contains utility functions such as cosine similarity, which are useful for comparing text embeddings.
    
- **`fastapi`**:
  - **`FastAPI`**: Used to create and configure API application.
    
- **`pydantic`**:
  - **`BaseModel`**: For creating data models with validation and serialization capabilities.
    
- **`json`**:
  - Provides methods for reading and writing JSON data.
  
- **`torch`**:
  - A deep learning library used for tensor operations, model training, and GPU acceleration.

- **`mongodb_interface`**:
  - **`MongoDBManager`**: To handle database connections and queries.
 
- **`abc`**:
  - **`ABC`**: For defining abstract base classes.
  - **`abstractmethod`**: A decorator to mark methods as abstract, requiring subclasses to implement them.

- **`pymongo`**:
  - **`MongoClient`**: To connect to a MongoDB server.

- **`datetime`**:
  - **`datetime`**: For working with date and time objects.
 
## Usage

### Load the model you have downloaded.

```python
model = SentenceTransformer('./path/to/model')
```

### Initialise MongoDBManager (implementation of interface below).

```python
mongo_manager = MongoDBManager()
```

### Load json file.

```python
with open('./CleanedFullFAQ.json', 'r') as f:
    faq_data = json.load(f)
```

### Extract data from json file

Depending on your json file and its format, extract the required data accordingly. Here each part is extracted, from the json file, with its id, topicId, title (question) and body (answer).

```python
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
```

### Create embeddings

This will be embeddings of the questions in your json file, which will then be used to calculate cosine similarity scores.

```python
question_embeddings = model.encode(questions, convert_to_tensor=True)
```

### Initialise new FastAPI application

```python
app = FastAPI()
```

### Define a Data Model

This model defines the structure of the expected input for the API. It contains a single field, question, which is a string. When a request is made to the API, FastAPI will validate that the incoming data matches this structure.

```python
class Query(BaseModel):
    question: str
```

### API endpoint

A new POST endpoint is defined at the route `/query`. When a POST request is made to this endpoint, the `get_most_relevant_questions` function will be called. 
The `get_most_relevant_questions` function first inserts the query in to the database. It then encodes the query using `model.encode()`, and the cosine similarity scores are calculated, using `util.cos_sim`. `torch.topk()` orders teh scores from highest to lowest,a nd by assigning `k` an integer it will return the top `k` scores. The function separates the main question (top score) with related questions (following top scores), as the main question is expected to deal with the user query. A for loop deals with the related questions;
In the loop:
  - `top_k.indices[0][1:]` contains the indices of the top `k` results (apart from the top score) and `top_k.values[0][1:]` contains        their corresponding scores, the `zip` function combines these two lists.
  - `item()` converts the tensor to a Python type.
  - The question, it's answer,type and score will be appended as a dictionary to the array.

```python
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
```

### MongoDb interface

This interface is for the database, for abstraction and easy implementation of any changes.
An abstract base class is created inheriting from `ABC` - serving as a blueprint for MongoDB operations. In contains the abstract method `insert_query` (you can add more).
The implementation of the interface is done by creating a new class that implements the interface. In this class, when intialising a method of its self, it connects to the MongoDB server and selects the speicif databse and collection. The implementation of the abstarct method contains the insertion method to store the query and a timestamp into the database.

```python
# Abstract Interface for MongoDB operations
class MongoDBInterface(ABC):

    @abstractmethod
    def insert_query(self, query: str) -> None:
        pass

# Concrete Implementation of MongoDBInterface
class MongoDBManager(MongoDBInterface):
    def __init__(self):
        # Initialize the MongoDB client
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['user_queries_db']
        self.queries_collection = self.db['queries']

    # Implementation of insert_query method
    def insert_query(self, query: str) -> None:
        self.queries_collection.insert_one({
            "query": query,
            "timestamp": datetime.now()
        })
```

## To run the API

Make sure your virtual environment is activated.
Then run the FastAPI application:

```bash
uvicorn (NameOfPythonFileWithout.pyExtension):app --reload
```

The server will be running on your local machine: **http://127.0.0.1:8000**

Postman was used to test the API. Download here [https://www.postman.com/downloads/]

  1. Create an account
  2. Launch the Postman application on your computer
  3. Click on the "New" button or the "+" tab to create a new request - Select "HTTP Request"
  4. Change the request type from GET to POST using the dropdown next to the request URL field.
  5. Enter the Request URL - (http://127.0.0.1:8000/query)
  6. Click on the "Body" tab - Select the "raw" option.
  7. In the text area that appears, enter your JSON object. For example:
     {
          "question": "How much do I need in my account to strat trading?"
     }
  8. Click the "Send" button
  9. After a moment, you should see the response from your FastAPI application in the lower section of the Postman window.
