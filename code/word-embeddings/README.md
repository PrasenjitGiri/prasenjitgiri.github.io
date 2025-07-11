# Word Embeddings from Scratch - Code Implementation

This folder contains the complete implementation of word embeddings as described in the blog post: [Word Embeddings from Scratch: The Journey from Words to Vectors](../../post.html?id=word-embeddings-from-scratch).

## 🎯 Core Question Answered

**"Does a 20-word sentence create 20 embeddings or one?"**

**Answer: 20 individual word embeddings, one for each word.**

This code proves this mathematically and shows exactly how it works.

## 📁 Files Overview

### Main Implementation
- **`word_embeddings_from_scratch.py`** - Complete Skip-Gram implementation with negative sampling and BPE
- **`Word_Embeddings_From_Scratch.ipynb`** - Interactive Jupyter notebook with step-by-step explanations
- **`demo.py`** - Quick demonstration script showing key concepts

### Visualizations
- **`generate_visualizations.py`** - Creates all charts and diagrams used in the blog post

## 🚀 Quick Start

### Option 1: Run the Demo
```bash
python demo.py
```
This runs a quick demonstration showing:
- Training process
- Individual word embeddings for each word in a sentence
- Word similarities
- Real-world usage examples

### Option 2: Interactive Notebook
```bash
jupyter notebook Word_Embeddings_From_Scratch.ipynb
```
Step-by-step interactive exploration with detailed explanations.

### Option 3: Full Implementation
```python
from word_embeddings_from_scratch import WordEmbeddingTrainer

# Your training corpus
corpus = [
    "Your training sentences here",
    "Machine learning is fascinating",
    # ... more sentences
]

# Train the model
trainer = WordEmbeddingTrainer(
    embedding_dim=300,
    window_size=5,
    negative_samples=5,
    learning_rate=0.025,
    min_count=5,
    epochs=20
)

metrics = trainer.train(corpus)

# Get individual word embeddings
sentence = "Each word gets its own embedding"
words = sentence.lower().split()

for word in words:
    embedding = trainer.get_word_vector(word)
    print(f"'{word}': {embedding.shape if embedding is not None else 'Not in vocab'}")
```

## 🔍 What You'll Learn

1. **Skip-Gram Algorithm**: How words predict their context
2. **Negative Sampling**: Making training computationally feasible
3. **BPE Tokenization**: Handling out-of-vocabulary words
4. **Individual Embeddings**: Why each word gets its own vector
5. **Real Applications**: How search engines, ChatGPT, and recommenders use this

## 📊 Generated Visualizations

The `generate_visualizations.py` script creates:
- Skip-Gram architecture diagram
- Embedding space visualization with semantic clustering
- BPE tokenization process
- Training curves
- Word similarity matrices
- Context window illustration
- Negative sampling demonstration
- t-SNE embeddings visualization

All images are saved to `../../assets/images/word-embeddings/`

## 🧠 Key Insights

### Mathematical Foundation
- **Objective**: Maximize P(context|target) using sigmoid and negative sampling
- **Loss**: -log(σ(v_c · v_w)) for positive pairs, -log(σ(-v_n · v_w)) for negative pairs
- **Updates**: Gradient descent on embedding matrices W1 (input) and W2 (output)

### Real-World Usage
```python
# How ChatGPT processes text:
# 1. Tokenize: "Hello world" → ["Hello", "world"]
# 2. Embed: Each token gets its own vector
# 3. Position: Add positional encodings
# 4. Transform: Process through attention layers
# 5. Generate: Predict next token
```

### From Words to Sentences
```python
sentence = "Machine learning is amazing"
words = sentence.split()

# Individual embeddings (what we built)
embeddings = [get_embedding(word) for word in words]

# Sentence embedding (combination methods)
sentence_emb = np.mean(embeddings, axis=0)  # Simple averaging
# Advanced: Use attention weights, LSTM, or transformer
```

## 🔗 References

The implementation follows these foundational papers:
- Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013): "Distributed Representations of Words and Phrases"
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"

## 🎓 Next Steps

1. **Experiment**: Try different hyperparameters and corpus sizes
2. **Extend**: Add hierarchical softmax, FastText subword embeddings
3. **Apply**: Use these embeddings in classification, clustering, search
4. **Advance**: Explore contextual embeddings (BERT, GPT) that build on these foundations

## 💡 The Bottom Line

Word embeddings are the foundation of modern NLP. Every breakthrough from Word2Vec to ChatGPT starts with the principle demonstrated here: **each word gets its own vector, and meaning emerges from the mathematical relationships between these vectors.**

Understanding this implementation gives you the foundation to understand any modern language model.

---

**Blog Post**: [Word Embeddings from Scratch: The Journey from Words to Vectors](../../post.html?id=word-embeddings-from-scratch)  
**Author**: Prasenjit Giri  
**Date**: January 2023 