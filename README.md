# LLMFAQs

In this project, a pre-trained LLM is fine-tuned on a specific FAQ dataset (in json format), so a user query can be answered with the most relevant question in the FAQ dataset. 

## Installation

As the code is implemented in Google Colab, use the package manager **!pip** to install sentence-transformers

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

Create an array of questions which will be used later on to for the model to encode and compare to the questions in the json file. You can get these questions from ChatGPT. eg: 'Give me 100 questions that's similar to the questions in my json file but not the same - give it in speech marks with a comma after each and no index'.

```python
similar_questions = ["How much do I need in my account to start trading", ".....",...]
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
The dataframe has to be global.
The for loop iterates through every question, uses the `retrieve_top_k_answers()` function to retrieve the top k answers, and then iterates through them, creating a dataframe from each then appending it to the global dataframe uisng `pd.concat()`

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

### Fine-tuning

First the training parameters are intialised, this is a dictionary with each value being an array. You can add or delete keys how you wish (make sure to edit the code of where it might be used). 
`InputExample` is used to represent a pair of texts, in this case our question and it's answer from the json file, that was extracted. This will be used for the model to understand the relationship between both.
`itertools.product()` is used to create different combinations of the training parameters, this ensures every model will be created with different parameters (depending on the values in the arrays, is how many models will be created, in this case it will be 243. The GPU storage might be a problem, so you might want to keep the amount of values in the arrays small).
The `for` loop iterates through the combinations with an index, so each model can have a different name. Within the loop, the pre-trained model is loaded using `SentenceTransformer`, which provides a powerful and efficient way to generate semantic embeddings for text, allowing for meaningful comparison and ranking of sentences based on their semantic similarity. A `DataLoader` is created to load the training examples in batches, which is shuffled to improve model generalisation. A loss function is defined, here `losses.MultipleNegativesRankingLoss()` is used, where the goal is to learn embeddings that rank positive examples higher than negative ones. The objectives are defined with the parameters for the training. The fine-tuned model is then used to encode the questions that was extracted from the json file, uisng `model.enocde()`. First compare the models in the dataframe before downloading (to download you will have to train the model again with its specific parameters). The `save_model_info_as_df()` function is used to compare the question embeddings and the user embeddings, and to insert the relevant information in to the dataframe - when doing this you might want to use just 1 question for each model, so the dataframe is not crowded (instead of `irrelevant_questions` you can put "How much do I need in my account to start trading?") - from the dataframe you can then pick, maybe 5, and feed each 100 questions one at a time, so each model has its own dataframe, and then compare them to pick your model. To view the dataframe you can execute `output_df` in a cell after.

```python
# Configure fine-tuning parameters
training_parameters = {
    'num_epochs': [2,4,8],
    'batch_size': [8,16,32],
    'warmup_steps': [100,500,1000],
    'weight_decay': [0.01,0.05,0.1],
    'learning_rate': [1e-5,3e-5,5e-5]
}

train_examples = [InputExample(texts=[faq['title'], faq['body']]) for faq in articles]

param_combinations = list(itertools.product(*training_parameters.values()))

for i,(num_epochs, batch_size, warmup_steps, weight_decay, learning_rate) in enumerate(param_combinations):
  print(f"Training with parameters: num_epochs={num_epochs}, batch_size={batch_size}, warmup_steps={warmup_steps}, weight_decay={weight_decay}, learning_rate={learning_rate}")

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
  model_name = f"model_{i}"

  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
  train_loss = losses.MultipleNegativesRankingLoss(model)

  # Train the model
  model.fit(
      train_objectives=[(train_dataloader, train_loss)],
      epochs=num_epochs,
      warmup_steps=warmup_steps,
      weight_decay=weight_decay,
      optimizer_params={'lr': learning_rate}  # Learning rate for fine-tuning
  )

  question_embeddings = model.encode(questions, convert_to_tensor=True)

  # model_path = './path/to/save/model'
  # model.save(model_path)

  save_model_info_as_df(model_name, num_epochs, batch_size, warmup_steps, weight_decay, learning_rate, irrelevant_questions)
```

### Downloading 

When you have picked the models you want to use for the 100 questions, you might have to restart the session after each training, as the GPU storage can be a problem in Google Colab - if not, just make sure you initialise the dataframe again by just executing the cell where it was first initialised, so it becomes empty.
When you want you can download the model onto your local computer. After training of all models in one execution, you can download the dataframe on to your local computer.

```python
From google.colab import files

files.download(model)

filepath = ./path/to/excel/file.xlsx
output_df.to_excel(filepath, index=False)
files.download(filepath)
```
