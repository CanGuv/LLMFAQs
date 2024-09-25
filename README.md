# LLMFAQs

In this project, a pre-trained LLM is fine-tuned on a specific FAQ dataset (in json format), so a user query can be answered with the most relevant question in the FAQ dataset. 

## Installation

As the code is implemente in Google Colab, use the package manager [!pip] to install sentence-transformers

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
  - A Python module providing functions that create iterators for efficient looping and                  combinatorial processing, used here for model parameters
