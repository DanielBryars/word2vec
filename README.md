# Word2Vec Examples

Simple examples demonstrating the use of Google News Word2Vec embeddings with Gensim for natural language processing tasks.

## Overview

This project provides basic examples of loading and using pre-trained Word2Vec embeddings from Google's News corpus, which contains 300-dimensional vectors for 3 million words and phrases.

## Features

- Load pre-trained Google News Word2Vec embeddings
- Find most similar words using cosine similarity
- Convert sentences to vector representations
- Word arithmetic and analogies

## Tech Stack

- **Gensim**: Word2Vec model loading and operations
- **NumPy**: Vector computations

## Installation

```bash
pip install -r requirements.txt
```

## Download Model

Download the Google News Word2Vec model:
- URL: https://code.google.com/archive/p/word2vec/
- File: `GoogleNews-vectors-negative300.bin.gz`
- Size: ~1.5 GB

Place the downloaded file in the project directory.

## Usage

```bash
python google-news.py
```

### Example Operations

**Find Similar Words**:
```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print(model.most_similar('king'))
# Output: [('queen', 0.651), ('monarch', 0.602), ...]
```

**Get Word Vector**:
```python
vector = model['king']
print(vector.shape)  # (300,)
```

**Sentence to Vector**:
```python
import numpy as np

sentence = "This is my query"
words = sentence.lower().split()
word_vectors = [model[word] for word in words if word in model]
sentence_vector = np.mean(word_vectors, axis=0)
```

## Word2Vec Embeddings

The Google News model:
- **Dimensionality**: 300
- **Vocabulary**: 3 million words/phrases
- **Training Corpus**: Google News dataset (~100 billion words)
- **Architecture**: Skip-gram with negative sampling

### Properties

- Captures semantic relationships (king - man + woman ≈ queen)
- Cosine similarity measures word relatedness
- Can handle multi-word phrases
- Case-sensitive

## Common Use Cases

- Text similarity comparison
- Document clustering
- Feature extraction for ML models
- Query expansion in search systems
- Recommendation systems

## Limitations

- Large file size (~1.5 GB in memory)
- Fixed vocabulary (no out-of-vocabulary handling)
- No contextual understanding (same vector regardless of context)
- Outdated corpus (pre-2013 news)

## Alternative Models

Consider more modern alternatives:
- **GloVe**: Global Vectors for Word Representation
- **FastText**: Handles out-of-vocabulary words via subword information
- **BERT**: Contextual embeddings
- **Sentence-BERT**: Sentence-level embeddings
- **GPT embeddings**: From OpenAI models

## License

Google News Word2Vec model is provided by Google for research purposes.

## References

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
