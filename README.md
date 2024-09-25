# LLMFAQs

In this project, a pre-trained LLM is fine-tuned on a specific FAQ dataset (in json format), so a user query can be answered with the most relevant question in the FAQ dataset. 

## Installation

As the code is implemented in Google Colab, use the package manager [!pip] (https://pip.pypa.io/en/stable/) to install sentence-transformers

```bash
!pip install sentence-transformers
!pip install Datasets
```

## Required Libraries

- **`sentence_transformers`**:
  - **`SentenceTransformer`**: Used for loading pre-trained models and generating embeddings for         text data.
  - **`InputExample`**: Helps in formatting the input data during fine-tuning or training.
  - **`losses`**: Contains different loss functions for training the models.
  - **`util`**: Contains utility functions such as cosine similarity, which are useful for comparing     text embeddings.

- **`torch.utils.data`**:
  - **`DataLoader`**: Used for creating efficient data loaders to handle batches of training data.

- **`json`**: 
  - Provides methods for reading and writing JSON data.

- **`pandas`**:
  - A powerful data manipulation library, used here to store the model and it's parameters, with         scores, for easy comparison

- **`torch`**:
  - A deep learning library used for tensor operations, model training, and GPU acceleration.

- **`os`**:
  - A standard Python library for interacting with the operating system, such as file and directory      operations.

- **`itertools`**:
  - A Python module providing functions that create iterators for efficient looping and                  combinatorial processing, used here for model parameters.
 
## Usage

### Create questions

Create an array of questions which will be used later on to for the model to encode and compare to the questions in the json file. You can get these questions from ChatGPT. eg: 'Give me 100 questions that's similar to the questions in my json file but not the same - give it in speeach marks with a comma after each and no index'.

```python
similar_questions = ["How much do I need in my accoun to start trading", ".....",...]
```

### Loading JSON data

To load data from a json file, specify your file path.

```python
json_data='./your/file/path/to/json/file'

with open(json_data, 'r') as f:
    data = json.load(f)
```

### Extract specific data from json file

Depending on your json file and its format, extract the required data accordingly. Here each question is extracted, from the json file, with its id, topicId, title (question) and body (answer).

```python
articles = []
for topic in data:
    for article in topic['articles']:
        articles.append({
            'id': article['id'],
            'topicId': topic['id'],
            'title': article['title'],
            'body': article['body']
        })

questions = [faq['title'] for faq in articles]
answers = [faq['body'] for faq in articles]
```

If you want to view your extraction for a better understanding you can:

```python
df = pd.DataFrame(articles)
df
```

### Create dataframe 

Create a dataframe to store the models. This will be used for comparison and to keep track of results. The columns will be the model's parameters, model name, user query, relevant question and cosine score.

```python
column_names = ['Model Name', 'Num Epochs', 'Batch Size', 'Learning Rate', 'Warmup Steps', 'Weight Decay', 'User Query', 'Relevant Question', 'Cosine Score']
output_df = pd.DataFrame(columns=column_names)
```

### Function to retrieve model's output

This fucntion will take the `user_query` and `k` as an argument, `k` will be defaulted to 1 so the dataframe is not crowded. 
The function will encode the `user_query` (mathematical representation for the query), and this is done so it can be compared to the original `question_embeddings` (which will be seen when fine-tuning the model). 
The `cosine_scores` are calculated using the `util` library and it's function `cos_sim`. You can store the best scores in a variable, using the `torch` library and it's `topk` function, where `k` is the number of scores to return (if `k=5` top 5 scores are returned in order). 
Empty array is intialised to store question, answer and score.
In the loop:
  - `top_k.indices[0]` contains the indices of the top `k` results and `top_k.values[0]` contains        their corresponding scores, the `zip` function combines these two lists.
  - `idx.item()` converts the tensor `idx` to a Python integer.
  - `score.item()` converts the `score` tensor to a Python float.
  - The question, it's answer and score will be appended as a dictionary to the array. 



```python
def retrieve_top_k_answers(user_query, k=1):

  query_embedding = model.encode(user_query, convert_to_tensor=True)

  cosine_scores = util.cos_sim(query_embedding, question_embeddings)
  top_k = torch.topk(cosine_scores, k=k)

  top_k_answers = []
  for idx, score in zip(top_k.indices[0], top_k.values[0]):
        # Convert tensor to integer index
        idx = idx.item()
        score = score.item()

        top_k_answers.append({
            'question': questions[idx],
            'answer': answers[idx],
            'score': score
        })

  return top_k_answers
```

### Function to save models to dataframe

This function will take the columns of the dataframe created earlier as arguments and `user_questions` which was intialised at the start (from ChatGPT). 
Make sure the dataframe is global.
The for loop iterates through every question, uses the `retrieve_top_k_answers()` function to retrieve the top k answers, and then iterates through them, creating a dataframe from each then appending it on the global dataframe uisng `pd.concat()`

```python
def save_model_info_as_df(model_name, num_epochs, batch_size, warmup_steps, weight_decay, learning_rate, user_questions):

  global output_df

  for query in user_questions:
    top_k_answers = retrieve_top_k_answers(query)
    for answer_info in top_k_answers:
        data_to_append = {
                'Model Name': model_name,
                'Num Epochs': num_epochs,
                'Batch Size': batch_size,
                'Warmup Steps': warmup_steps,
                'Weight Decay': weight_decay,
                'Learning Rate': learning_rate,
                'User Query': query,
                'Relevant Question': answer_info['question'],
                'Cosine Score': answer_info['score']
            }

        df_new_data = pd.DataFrame([data_to_append])
        output_df = pd.concat([output_df, df_new_data], ignore_index=True)
```
