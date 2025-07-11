# Word Embeddings from Scratch: The Journey from Words to Vectors

*Ever wondered how computers understand that "king" and "queen" are related, or that "Paris" is closer to "France" than to "banana"? Welcome to the fascinating world of word embeddings—where mathematics meets linguistics in the most elegant way possible.*

Today, we're going on a journey to build word embeddings completely from scratch, understand Byte Pair Encoding (BPE), and answer the fundamental question that confuses many: **When I have a sentence of 20 words, do I get 20 different embeddings or one combined embedding?**

By the end of this post, you'll not only understand the mathematics behind embeddings but also implement them without using any external libraries.

## The Great Embedding Mystery: One Sentence, How Many Vectors?

Let's start with your exact question because it's the foundation of everything we'll discuss:

**Scenario**: You have the sentence "The quick brown fox jumps over the lazy dog" (9 words). You keep adding words: "The quick brown fox jumps over the lazy dog runs" (10 words), then "The quick brown fox jumps over the lazy dog runs fast" (11 words).

**Question**: Do you get 9, 10, and 11 different embeddings respectively, or do you get one embedding for each complete sentence?

**Answer**: **You get individual embeddings for each word!** 

- "The quick brown fox" → 4 separate word embeddings: [vec_the, vec_quick, vec_brown, vec_fox]
- Each word has its own vector representation
- The sentence-level representation comes from **combining** these individual word embeddings

But wait—there's more to this story. Let's dive deep into how this actually works.

## Chapter 1: What Are Word Embeddings, Really?

### The Problem: Computers Don't Understand Words

Imagine you're teaching a robot to understand human language. You show it the word "dog." The robot asks: "What's a dog?"

You could try:
- "It's a four-legged animal" → But so is a cat, horse, and table
- "It's a pet" → But so is a fish or bird
- "It's loyal and friendly" → But that's subjective

The fundamental challenge: **How do we convert the semantic richness of words into numbers that computers can process?**

### The Solution: Vector Space Magic

Word embeddings solve this by mapping each word to a high-dimensional vector (typically 100-300 dimensions) where:
- Similar words are close to each other
- Relationships are preserved through vector arithmetic
- Semantic meaning is captured in geometric space

**Example**: If we have 3-dimensional embeddings:
```
"king"   → [0.2, 0.8, 0.3]
"queen"  → [0.1, 0.7, 0.9]
"man"    → [0.3, 0.1, 0.2]
"woman"  → [0.2, 0.0, 0.8]
```

The famous relationship emerges: **king - man + woman ≈ queen**

## Chapter 2: Building Word Embeddings from Scratch

### The Mathematical Foundation: Skip-Gram Model

Let's implement the Skip-Gram model—one of the most intuitive embedding approaches. The idea: **predict context words given a target word**.

**Scenario**: Given the sentence "The cat sat on the mat"
- Target word: "sat"
- Context words: "cat", "on" (window size = 1)
- Goal: Train a neural network to predict "cat" and "on" when given "sat"

### Step 1: Data Preparation

```python
import numpy as np
import re
from collections import defaultdict, Counter
import random

class WordEmbeddingTrainer:
    def __init__(self, vector_size=100, window_size=2, min_count=1, learning_rate=0.01):
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        
        # Vocabulary mappings
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
        
        # Neural network weights
        self.W1 = None  # Input to hidden layer
        self.W2 = None  # Hidden to output layer
        
        # Training data
        self.training_pairs = []
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        return words
    
    def build_vocabulary(self, corpus):
        """Build vocabulary from corpus"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for sentence in corpus:
            words = self.preprocess_text(sentence)
            self.word_counts.update(words)
        
        # Filter words by minimum count
        vocab_words = [word for word, count in self.word_counts.items() 
                      if count >= self.min_count]
        
        # Create word-to-index mappings
        self.word_to_id = {word: i for i, word in enumerate(vocab_words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        self.vocab_size = len(vocab_words)
        print(f"Vocabulary size: {self.vocab_size}")
        return vocab_words
```

### Step 2: Generate Training Pairs

```python
    def generate_training_data(self, corpus):
        """Generate (target_word, context_word) pairs"""
        print("Generating training pairs...")
        
        for sentence in corpus:
            words = self.preprocess_text(sentence)
            word_ids = [self.word_to_id[word] for word in words 
                       if word in self.word_to_id]
            
            # Generate skip-gram pairs
            for i, target_id in enumerate(word_ids):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(word_ids), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Don't pair word with itself
                        context_id = word_ids[j]
                        self.training_pairs.append((target_id, context_id))
        
        print(f"Generated {len(self.training_pairs)} training pairs")
        return self.training_pairs
```

### Step 3: Initialize Neural Network

```python
    def initialize_weights(self):
        """Initialize neural network weights"""
        # Xavier initialization for better convergence
        limit = np.sqrt(6.0 / (self.vocab_size + self.vector_size))
        
        # Input to hidden weights (vocab_size x vector_size)
        self.W1 = np.random.uniform(-limit, limit, 
                                   (self.vocab_size, self.vector_size))
        
        # Hidden to output weights (vector_size x vocab_size)
        self.W2 = np.random.uniform(-limit, limit, 
                                   (self.vector_size, self.vocab_size))
        
        print(f"Initialized weights: W1 {self.W1.shape}, W2 {self.W2.shape}")
```

### Step 4: Forward Pass and Softmax

```python
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward_pass(self, target_word_id):
        """Forward pass through the network"""
        # One-hot encode input
        input_vector = np.zeros(self.vocab_size)
        input_vector[target_word_id] = 1
        
        # Hidden layer (this becomes our word embedding!)
        hidden = np.dot(input_vector, self.W1)
        
        # Output layer
        output_scores = np.dot(hidden, self.W2)
        output_probs = self.softmax(output_scores)
        
        return hidden, output_probs
```

### Step 5: Backpropagation and Training

```python
    def backward_pass(self, target_id, context_id, hidden, output_probs):
        """Backward pass and weight updates"""
        # Create target vector (one-hot for context word)
        target_vector = np.zeros(self.vocab_size)
        target_vector[context_id] = 1
        
        # Calculate error
        error = output_probs - target_vector
        
        # Gradients
        dW2 = np.outer(hidden, error)
        dW1_hidden = np.dot(error, self.W2.T)
        
        # One-hot vector for target word
        input_vector = np.zeros(self.vocab_size)
        input_vector[target_id] = 1
        dW1 = np.outer(input_vector, dW1_hidden)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1
        
        # Calculate loss (cross-entropy)
        loss = -np.log(output_probs[context_id] + 1e-10)
        return loss
    
    def train(self, corpus, epochs=100):
        """Train the word embedding model"""
        # Build vocabulary and generate training data
        self.build_vocabulary(corpus)
        self.generate_training_data(corpus)
        self.initialize_weights()
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(self.training_pairs)
            
            for target_id, context_id in self.training_pairs:
                hidden, output_probs = self.forward_pass(target_id)
                loss = self.backward_pass(target_id, context_id, hidden, output_probs)
                total_loss += loss
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(self.training_pairs)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
    
    def get_word_embedding(self, word):
        """Get embedding vector for a word"""
        if word not in self.word_to_id:
            return None
        
        word_id = self.word_to_id[word]
        # The embedding is the corresponding row in W1
        return self.W1[word_id]
    
    def find_similar_words(self, word, top_k=5):
        """Find most similar words using cosine similarity"""
        if word not in self.word_to_id:
            return []
        
        word_vec = self.get_word_embedding(word)
        similarities = []
        
        for other_word in self.word_to_id:
            if other_word != word:
                other_vec = self.get_word_embedding(other_word)
                # Cosine similarity
                similarity = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
```

### Step 6: Let's Train Our Model!

```python
# Example usage
corpus = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets",
    "the quick brown fox jumps over the lazy dog",
    "a cat is a small animal",
    "dogs are loyal animals",
    "the mat is on the floor",
    "animals live in the wild",
    "pets need care and love",
    "the brown cat likes to sit"
]

# Create and train the model
trainer = WordEmbeddingTrainer(vector_size=50, window_size=2, learning_rate=0.1)
trainer.train(corpus, epochs=200)

# Test the embeddings
print("Word embedding for 'cat':")
cat_embedding = trainer.get_word_embedding('cat')
print(cat_embedding[:10])  # Show first 10 dimensions

print("\nWords similar to 'cat':")
similar_to_cat = trainer.find_similar_words('cat', top_k=3)
for word, similarity in similar_to_cat:
    print(f"{word}: {similarity:.4f}")
```

**Key Insight**: Each word gets its own unique embedding vector. The word "cat" will always have the same embedding regardless of the sentence it appears in!

## Chapter 3: The Sentence Embedding Question Answered

Now let's address your core question with concrete examples:

### Scenario Analysis

**Sentence 1**: "The cat sat" (3 words)
```python
sentence_1_embeddings = [
    trainer.get_word_embedding('the'),    # Vector 1: [0.1, 0.3, 0.7, ...]
    trainer.get_word_embedding('cat'),    # Vector 2: [0.4, 0.1, 0.9, ...]
    trainer.get_word_embedding('sat')     # Vector 3: [0.2, 0.8, 0.1, ...]
]
# Result: 3 separate embedding vectors
```

**Sentence 2**: "The cat sat on" (4 words)
```python
sentence_2_embeddings = [
    trainer.get_word_embedding('the'),    # Same as before!
    trainer.get_word_embedding('cat'),    # Same as before!
    trainer.get_word_embedding('sat'),    # Same as before!
    trainer.get_word_embedding('on')      # New vector: [0.6, 0.2, 0.4, ...]
]
# Result: 4 separate embedding vectors (first 3 identical to sentence 1)
```

### How to Create Sentence-Level Embeddings

To get ONE embedding for the entire sentence, you need to **combine** the word embeddings:

```python
def create_sentence_embedding(trainer, sentence, method='average'):
    """Create a single embedding for an entire sentence"""
    words = trainer.preprocess_text(sentence)
    word_embeddings = []
    
    for word in words:
        embedding = trainer.get_word_embedding(word)
        if embedding is not None:
            word_embeddings.append(embedding)
    
    if not word_embeddings:
        return None
    
    word_embeddings = np.array(word_embeddings)
    
    if method == 'average':
        # Simple average of all word vectors
        return np.mean(word_embeddings, axis=0)
    
    elif method == 'weighted_average':
        # Weight by inverse frequency (TF-IDF style)
        weights = []
        for word in words:
            if word in trainer.word_counts:
                # Higher weight for rarer words
                weight = 1.0 / trainer.word_counts[word]
                weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        return np.average(word_embeddings, axis=0, weights=weights)
    
    elif method == 'max_pooling':
        # Take maximum value for each dimension
        return np.max(word_embeddings, axis=0)

# Example usage
sentence1 = "The cat sat"
sentence2 = "The cat sat on the mat"

sent1_embedding = create_sentence_embedding(trainer, sentence1, 'average')
sent2_embedding = create_sentence_embedding(trainer, sentence2, 'average')

print(f"Sentence 1 embedding shape: {sent1_embedding.shape}")
print(f"Sentence 2 embedding shape: {sent2_embedding.shape}")

# Calculate similarity between sentences
similarity = np.dot(sent1_embedding, sent2_embedding) / (
    np.linalg.norm(sent1_embedding) * np.linalg.norm(sent2_embedding)
)
print(f"Similarity between sentences: {similarity:.4f}")
```

### The Answer Crystallized

**For your 20-word sentence question**:

1. **Individual word embeddings**: You get 20 separate embedding vectors, one for each word
2. **Sentence embedding**: You get 1 combined embedding vector representing the entire sentence
3. **When you add the 21st word**: You get the same 20 embeddings plus 1 new embedding for the new word

**The key insight**: Word embeddings are **context-independent**. The word "cat" has the same embedding whether it appears in "The cat sat" or "A black cat sleeps."

## Chapter 4: Byte Pair Encoding (BPE) - The Subword Revolution

Now let's tackle BPE, which revolutionizes how we handle vocabulary and unknown words.

### The Problem BPE Solves

Traditional word embeddings have issues:
- **Large vocabularies**: English has 500K+ words
- **Out-of-vocabulary (OOV) words**: What about "ChatGPT" or "COVID-19"?
- **Morphological variations**: "run", "running", "runs" get separate embeddings

**BPE Solution**: Break words into subword units that balance between characters and words.

### BPE Algorithm Implementation

```python
import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.word_freqs = {}
        self.vocab = set()
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
    
    def get_word_tokens(self, word):
        """Convert word to character tokens with end-of-word marker"""
        return list(word) + ['</w>']
    
    def get_pairs(self, word_tokens):
        """Get all adjacent pairs in the word"""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(self, corpus):
        """Train BPE on corpus"""
        print("Training BPE tokenizer...")
        
        # Step 1: Get word frequencies
        word_pattern = re.compile(r'\b\w+\b')
        for text in corpus:
            words = word_pattern.findall(text.lower())
            for word in words:
                self.word_freqs[word] = self.word_freqs.get(word, 0) + 1
        
        # Step 2: Initialize vocabulary with characters
        vocab = defaultdict(int)
        for word, freq in self.word_freqs.items():
            word_tokens = self.get_word_tokens(word)
            vocab[tuple(word_tokens)] += freq
            # Add individual characters to vocabulary
            for token in word_tokens:
                self.vocab.add(token)
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Step 3: Iteratively merge most frequent pairs
        for i in range(self.num_merges):
            # Count all pairs
            pairs = defaultdict(int)
            for word_tokens, freq in vocab.items():
                word_pairs = self.get_pairs(word_tokens)
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            most_frequent_pair = max(pairs, key=pairs.get)
            
            # Merge the most frequent pair
            new_vocab = {}
            pattern = re.escape(' '.join(most_frequent_pair))
            replacement = ''.join(most_frequent_pair)
            
            for word_tokens in vocab:
                word_str = ' '.join(word_tokens)
                new_word_str = re.sub(pattern, replacement, word_str)
                new_word_tokens = tuple(new_word_str.split())
                new_vocab[new_word_tokens] = vocab[word_tokens]
            
            vocab = new_vocab
            self.merges.append(most_frequent_pair)
            self.vocab.add(replacement)
            
            if i % 100 == 0:
                print(f"Merge {i}: {most_frequent_pair} -> {replacement}")
        
        # Step 4: Create token-to-id mappings
        vocab_list = list(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
    
    def tokenize_word(self, word):
        """Tokenize a single word using learned BPE"""
        word_tokens = self.get_word_tokens(word)
        
        # Apply merges in the order they were learned
        for pair in self.merges:
            word_str = ' '.join(word_tokens)
            pattern = re.escape(' '.join(pair))
            replacement = ''.join(pair)
            
            if re.search(pattern, word_str):
                word_str = re.sub(pattern, replacement, word_str)
                word_tokens = word_str.split()
        
        return word_tokens
    
    def encode(self, text):
        """Encode text to token IDs"""
        word_pattern = re.compile(r'\b\w+\b')
        words = word_pattern.findall(text.lower())
        
        token_ids = []
        for word in words:
            word_tokens = self.tokenize_word(word)
            for token in word_tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Handle unknown tokens - split into characters
                    for char in token:
                        if char in self.token_to_id:
                            token_ids.append(self.token_to_id[char])
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = [self.id_to_token[id] for id in token_ids if id in self.id_to_token]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

# Example usage
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps quickly",
    "the fox runs quickly through the forest",
    "running and jumping are good exercises",
    "the runner runs faster than the walker",
    "quickly running foxes jump over logs"
]

# Train BPE
bpe = BPETokenizer(num_merges=20)
bpe.train(corpus)

# Test tokenization
test_word = "running"
tokens = bpe.tokenize_word(test_word)
print(f"'{test_word}' -> {tokens}")

test_word = "jumping"
tokens = bpe.tokenize_word(test_word)
print(f"'{test_word}' -> {tokens}")

# Test encoding/decoding
test_sentence = "the fox is running quickly"
encoded = bpe.encode(test_sentence)
decoded = bpe.decode(encoded)
print(f"Original: {test_sentence}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### BPE + Word Embeddings: The Perfect Marriage

```python
class BPEWordEmbedding:
    def __init__(self, vector_size=100):
        self.bpe_tokenizer = None
        self.embeddings = None
        self.vector_size = vector_size
    
    def train(self, corpus, bpe_merges=1000, embedding_epochs=100):
        """Train both BPE tokenizer and embeddings"""
        # Step 1: Train BPE tokenizer
        self.bpe_tokenizer = BPETokenizer(num_merges=bpe_merges)
        self.bpe_tokenizer.train(corpus)
        
        # Step 2: Create subword corpus for embedding training
        subword_corpus = []
        for text in corpus:
            tokens = self.bpe_tokenizer.encode(text)
            token_words = [self.bpe_tokenizer.id_to_token[id] for id in tokens]
            subword_corpus.append(' '.join(token_words))
        
        # Step 3: Train embeddings on subword tokens
        embedding_trainer = WordEmbeddingTrainer(
            vector_size=self.vector_size,
            window_size=2,
            learning_rate=0.1
        )
        embedding_trainer.train(subword_corpus, epochs=embedding_epochs)
        
        self.embeddings = embedding_trainer
    
    def get_word_embedding(self, word):
        """Get embedding for a word by combining subword embeddings"""
        if not self.bpe_tokenizer or not self.embeddings:
            raise ValueError("Model not trained yet")
        
        # Tokenize word into subwords
        subword_tokens = self.bpe_tokenizer.tokenize_word(word)
        
        # Get embeddings for each subword
        subword_embeddings = []
        for token in subword_tokens:
            embedding = self.embeddings.get_word_embedding(token)
            if embedding is not None:
                subword_embeddings.append(embedding)
        
        if not subword_embeddings:
            return None
        
        # Average the subword embeddings
        return np.mean(subword_embeddings, axis=0)
    
    def handle_unknown_word(self, word):
        """Handle words not seen during training"""
        # BPE can handle any word by breaking it into known subwords
        return self.get_word_embedding(word)

# Example usage
bpe_embedder = BPEWordEmbedding(vector_size=50)
bpe_embedder.train(corpus, bpe_merges=20, embedding_epochs=100)

# Test with known and unknown words
print("Embedding for 'running':")
running_emb = bpe_embedder.get_word_embedding('running')
print(running_emb[:10])

print("\nEmbedding for 'ChatGPT' (unknown word):")
chatgpt_emb = bpe_embedder.handle_unknown_word('ChatGPT')
print(chatgpt_emb[:10] if chatgpt_emb is not None else "Could not generate embedding")
```

## Chapter 5: Advanced Embedding Concepts

### Position-Aware Embeddings

One limitation of basic word embeddings: they ignore word position. "Dog bites man" vs "Man bites dog" have the same word embeddings but different meanings!

```python
class PositionalEmbedding:
    def __init__(self, max_length=100, embedding_dim=100):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.positional_encodings = self._generate_positional_encodings()
    
    def _generate_positional_encodings(self):
        """Generate sinusoidal positional encodings"""
        pe = np.zeros((self.max_length, self.embedding_dim))
        
        for pos in range(self.max_length):
            for i in range(0, self.embedding_dim, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / self.embedding_dim)))
                if i + 1 < self.embedding_dim:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / self.embedding_dim)))
        
        return pe
    
    def add_positional_encoding(self, word_embeddings, positions):
        """Add positional encoding to word embeddings"""
        enhanced_embeddings = []
        
        for word_emb, pos in zip(word_embeddings, positions):
            if pos < self.max_length:
                pos_encoding = self.positional_encodings[pos]
                enhanced_emb = word_emb + pos_encoding
                enhanced_embeddings.append(enhanced_emb)
            else:
                enhanced_embeddings.append(word_emb)
        
        return np.array(enhanced_embeddings)

# Example usage
pos_embedder = PositionalEmbedding(max_length=50, embedding_dim=50)

# Get word embeddings for a sentence
sentence = "the cat sat on the mat"
words = sentence.split()
word_embeddings = []
positions = []

for i, word in enumerate(words):
    emb = trainer.get_word_embedding(word)
    if emb is not None:
        word_embeddings.append(emb)
        positions.append(i)

# Add positional information
enhanced_embeddings = pos_embedder.add_positional_encoding(word_embeddings, positions)
print(f"Enhanced embeddings shape: {enhanced_embeddings.shape}")
```

### Contextual vs Non-Contextual Embeddings

**Key Distinction**:
- **Non-contextual** (Word2Vec, GloVe): "bank" always has the same embedding
- **Contextual** (BERT, GPT): "bank" has different embeddings in "river bank" vs "money bank"

```python
def demonstrate_context_dependency():
    """Show why context matters"""
    sentences = [
        "I went to the bank to deposit money",
        "I sat by the river bank to fish",
        "The bank loan was approved quickly"
    ]
    
    # With non-contextual embeddings, "bank" is always the same
    bank_embedding = trainer.get_word_embedding('bank')
    print("Non-contextual 'bank' embedding (same for all contexts):")
    print(bank_embedding[:10])
    
    # In reality, we need different representations for different meanings
    print("\nIn contextual embeddings, 'bank' would have different vectors:")
    print("Financial bank: [0.1, 0.8, 0.2, ...]")
    print("River bank: [0.7, 0.1, 0.9, ...]")
    print("Bank as institution: [0.3, 0.6, 0.4, ...]")

demonstrate_context_dependency()
```

## Chapter 6: Real-World Applications and Performance

### Embedding Quality Evaluation

```python
def evaluate_embeddings(trainer, test_pairs):
    """Evaluate embedding quality using word similarity tasks"""
    
    similarities = []
    for word1, word2, expected_sim in test_pairs:
        emb1 = trainer.get_word_embedding(word1)
        emb2 = trainer.get_word_embedding(word2)
        
        if emb1 is not None and emb2 is not None:
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append((word1, word2, sim, expected_sim))
    
    return similarities

# Test with word pairs
test_pairs = [
    ("cat", "dog", 0.8),      # Both are pets
    ("cat", "car", 0.1),      # Unrelated
    ("king", "queen", 0.7),   # Related concepts
    ("run", "walk", 0.6),     # Similar actions
]

results = evaluate_embeddings(trainer, test_pairs)
for word1, word2, computed_sim, expected_sim in results:
    print(f"{word1} - {word2}: computed={computed_sim:.3f}, expected={expected_sim:.3f}")
```

### Scaling to Large Vocabularies

```python
class EfficientEmbeddingTrainer:
    """Memory and computationally efficient embedding trainer"""
    
    def __init__(self, vector_size=100, negative_samples=5):
        self.vector_size = vector_size
        self.negative_samples = negative_samples
        # Use hierarchical softmax or negative sampling for efficiency
    
    def negative_sampling_loss(self, target_emb, context_emb, negative_embs):
        """Compute loss using negative sampling"""
        # Positive example
        positive_score = np.dot(target_emb, context_emb)
        positive_loss = -np.log(self.sigmoid(positive_score))
        
        # Negative examples
        negative_loss = 0
        for neg_emb in negative_embs:
            negative_score = np.dot(target_emb, neg_emb)
            negative_loss += -np.log(self.sigmoid(-negative_score))
        
        return positive_loss + negative_loss
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

## Chapter 7: The Complete Picture - Answering All Your Questions

Let's revisit your original questions with complete clarity:

### Question 1: Sentence with 20 words → How many embeddings?

**Answer**: You get **20 individual word embeddings**, not one combined embedding.

```python
def demonstrate_word_vs_sentence_embeddings():
    sentence = "The quick brown fox jumps over the lazy dog and runs fast"
    words = sentence.split()
    
    print(f"Sentence: '{sentence}'")
    print(f"Number of words: {len(words)}")
    print(f"You get: {len(words)} separate word embeddings")
    
    # Individual word embeddings
    word_embeddings = []
    for i, word in enumerate(words):
        emb = trainer.get_word_embedding(word.lower())
        if emb is not None:
            word_embeddings.append(emb)
            print(f"Word {i+1}: '{word}' -> embedding vector of shape {emb.shape}")
    
    # To get ONE sentence embedding, you combine these:
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
        print(f"\nCombined sentence embedding shape: {sentence_embedding.shape}")
        print("This is ONE vector representing the entire sentence")

demonstrate_word_vs_sentence_embeddings()
```

### Question 2: Adding words to the sentence

```python
def demonstrate_incremental_sentence_building():
    """Show what happens when you keep adding words"""
    
    base_sentence = "The cat"
    additions = ["sat", "on", "the", "mat", "quietly"]
    
    for i in range(len(additions) + 1):
        if i == 0:
            current_sentence = base_sentence
        else:
            current_sentence = base_sentence + " " + " ".join(additions[:i])
        
        words = current_sentence.split()
        print(f"\nSentence {i+1}: '{current_sentence}'")
        print(f"Words: {len(words)}")
        print(f"Individual embeddings: {len(words)}")
        
        # Show that previous word embeddings remain the same
        if i > 0:
            print("Note: The embeddings for 'The' and 'cat' are IDENTICAL to previous iterations!")

demonstrate_incremental_sentence_building()
```

### Question 3: BPE Impact on Embeddings

```python
def demonstrate_bpe_vs_word_embeddings():
    """Compare word-level vs BPE embeddings"""
    
    word = "unhappiness"
    
    print(f"Word: '{word}'")
    
    # Traditional word embedding (if word exists in vocabulary)
    word_emb = trainer.get_word_embedding(word)
    if word_emb is not None:
        print(f"Word-level embedding: Single vector of shape {word_emb.shape}")
    else:
        print("Word-level embedding: NOT FOUND (out of vocabulary)")
    
    # BPE tokenization
    if bpe_embedder.bpe_tokenizer:
        bpe_tokens = bpe_embedder.bpe_tokenizer.tokenize_word(word)
        print(f"BPE tokens: {bpe_tokens}")
        print(f"Number of subword embeddings: {len(bpe_tokens)}")
        
        bpe_emb = bpe_embedder.get_word_embedding(word)
        if bpe_emb is not None:
            print(f"Combined BPE embedding: Single vector of shape {bpe_emb.shape}")
            print("This is created by averaging the subword embeddings")

demonstrate_bpe_vs_word_embeddings()
```

## Conclusion: The Embedding Journey Complete

We've traveled from the basic question of "how many embeddings?" to building complete embedding systems from scratch. Here are the key takeaways:

### Core Principles:
1. **Each word gets its own embedding vector** - this is fundamental
2. **Sentence embeddings are combinations of word embeddings** - averaging, weighted averaging, or more complex methods
3. **BPE enables handling of unknown words** by breaking them into known subword pieces
4. **Context independence** in traditional embeddings vs **context dependence** in modern models

### Implementation Insights:
- Word embeddings learn semantic relationships through co-occurrence patterns
- The Skip-Gram model predicts context from target words
- BPE solves the vocabulary explosion problem elegantly
- Position encodings add order awareness to embeddings

### The Answer to Your Original Question:
**A 20-word sentence gives you 20 individual word embeddings.** To get one sentence-level embedding, you must explicitly combine these 20 vectors using methods like averaging, max pooling, or learned attention mechanisms.

When you add the 21st word, you get the same 20 embeddings plus one new embedding for the additional word. The beauty is in the compositionality—complex meanings emerge from combining simple word vectors.

**Next time you see a sentence, remember**: it's not just text—it's a collection of high-dimensional vectors dancing together in semantic space, each word contributing its own mathematical signature to the overall meaning.

---

*The journey from words to vectors reveals the elegant mathematical foundation underlying natural language understanding. Every word is a point in space, every sentence a constellation of meaning, and every embedding model a bridge between human language and machine comprehension.* 