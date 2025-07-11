# Word Embeddings from Scratch: The Journey from Words to Vectors

*Published: January 20, 2023*

With ChatGPT revolutionizing how we interact with AI, understanding the fundamental building blocks of natural language processing has never been more crucial. At the heart of modern NLP systems lie **word embeddings** - dense vector representations that capture semantic meaning in numerical form.

But here's the question that sparked this deep dive: **When you have a 20-word sentence, do you get 20 different word embeddings or one combined embedding?** 

**The answer is definitive: You get 20 individual word embeddings, one for each word.** This might seem obvious, but it's a crucial distinction that shapes how we process and understand language computationally.

## The Foundation: What Are Word Embeddings?

Word embeddings transform discrete words into continuous vector spaces where semantic similarity translates to mathematical proximity. Unlike traditional one-hot encoding where each word is an isolated dimension, embeddings capture rich semantic relationships.

![Embedding Space Visualization](/assets/images/word-embeddings/embedding_space_visualization.png)

The groundbreaking work by Mikolov et al. in 2013 ([Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)) introduced Skip-Gram and Continuous Bag of Words (CBOW) models that revolutionized how we represent language computationally.

## Skip-Gram Architecture: Predicting Context from Target

The Skip-Gram model, detailed in Mikolov et al.'s follow-up paper ([Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)), learns embeddings by predicting context words given a target word.

![Skip-Gram Architecture](/assets/images/word-embeddings/skipgram_architecture.png)

### Mathematical Foundation

For a target word `w` and context word `c`, the Skip-Gram objective maximizes:

```
P(c|w) = exp(v_c · v_w) / Σ_{c'∈V} exp(v_c' · v_w)
```

Where:
- `v_w` is the target word embedding
- `v_c` is the context word embedding  
- `V` is the vocabulary

### Context Window: The Learning Mechanism

![Context Window Illustration](/assets/images/word-embeddings/context_window_illustration.png)

The context window defines which words influence each other during training. A window size of 2 means we consider 2 words on each side of the target word as context.

## Implementation from Scratch: No External Libraries

Let me walk you through a complete implementation that builds word embeddings from the ground up, explaining each component clearly.

### 1. Vocabulary Building and Preprocessing

```python
import numpy as np
import re
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import pickle
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WordEmbeddingTrainer:
    """
    Complete word embedding trainer implementing Skip-Gram with negative sampling.
    
    This implementation follows the mathematical foundations from:
    - Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
    - Mikolov et al. (2013): "Distributed Representations of Words and Phrases"
    """
    
    def __init__(self, embedding_dim: int = 300, window_size: int = 5, 
                 negative_samples: int = 5, learning_rate: float = 0.025,
                 min_count: int = 5, epochs: int = 10):
        """
        Initialize the word embedding trainer.
        
        Args:
            embedding_dim: Dimensionality of word vectors (typically 100-300)
            window_size: Context window size around target word
            negative_samples: Number of negative samples per positive sample
            learning_rate: Learning rate for gradient descent
            min_count: Minimum word frequency to include in vocabulary
            epochs: Number of training epochs
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.epochs = epochs
        
        # Vocabulary and mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        
        # Embedding matrices
        self.W1 = None  # Input embeddings (what we ultimately use)
        self.W2 = None  # Output embeddings (for prediction)
        
        # Training data
        self.training_pairs = []
        
        logger.info(f"Initialized WordEmbeddingTrainer with dim={embedding_dim}, "
                   f"window={window_size}, negative_samples={negative_samples}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by cleaning and tokenizing.
        
        Args:
            text: Raw input text
            
        Returns:
            List of cleaned tokens
        """
        try:
            # Convert to lowercase and remove extra whitespace
            text = text.lower().strip()
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
            
            # Split into tokens
            tokens = text.split()
            
            # Filter out very short tokens
            tokens = [token for token in tokens if len(token) > 1]
            
            logger.info(f"Preprocessed text: {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return []
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from input texts with frequency filtering.
        
        Args:
            texts: List of input text documents
        """
        try:
            logger.info("Building vocabulary...")
            
            # Count word frequencies across all texts
            for text in texts:
                tokens = self.preprocess_text(text)
                self.word_counts.update(tokens)
            
            # Filter by minimum count
            filtered_words = {word: count for word, count in self.word_counts.items() 
                            if count >= self.min_count}
            
            # Create word-to-index mappings
            self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words.keys())}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            self.vocab_size = len(self.word_to_idx)
            
            logger.info(f"Built vocabulary: {self.vocab_size} words "
                       f"(filtered from {len(self.word_counts)} total)")
            
            if self.vocab_size == 0:
                raise ValueError("Empty vocabulary after filtering. Lower min_count or provide more text.")
                
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            raise
```

### 2. Training Data Generation

```python
    def generate_training_data(self, texts: List[str]) -> None:
        """
        Generate (target, context) pairs for Skip-Gram training.
        
        For each word in the corpus, we create training pairs with surrounding words
        within the context window. This implements the core Skip-Gram idea of 
        predicting context from target.
        
        Args:
            texts: List of input text documents
        """
        try:
            logger.info("Generating training data...")
            self.training_pairs = []
            
            for text in texts:
                tokens = self.preprocess_text(text)
                
                # Convert to indices, skip unknown words
                indices = []
                for token in tokens:
                    if token in self.word_to_idx:
                        indices.append(self.word_to_idx[token])
                
                # Generate Skip-Gram pairs
                for i, target_idx in enumerate(indices):
                    # Define context window boundaries
                    start = max(0, i - self.window_size)
                    end = min(len(indices), i + self.window_size + 1)
                    
                    # Create pairs with all context words
                    for j in range(start, end):
                        if i != j:  # Skip the target word itself
                            context_idx = indices[j]
                            self.training_pairs.append((target_idx, context_idx))
            
            logger.info(f"Generated {len(self.training_pairs)} training pairs")
            
            if len(self.training_pairs) == 0:
                raise ValueError("No training pairs generated. Check vocabulary and input texts.")
                
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            raise
```

### 3. Negative Sampling Implementation

Negative sampling, introduced in Mikolov et al. (2013), makes training computationally feasible by approximating the softmax with a few negative examples.

![Negative Sampling Illustration](/assets/images/word-embeddings/negative_sampling_illustration.png)

```python
    def negative_sampling(self, target_idx: int, positive_context: int) -> List[int]:
        """
        Generate negative samples using unigram distribution.
        
        Following Mikolov et al. (2013), we sample negative examples from a unigram
        distribution raised to the 3/4 power, which gives better results than
        uniform sampling.
        
        Args:
            target_idx: Index of target word
            positive_context: Index of positive context word
            
        Returns:
            List of negative sample indices
        """
        try:
            # Create probability distribution (unigram^0.75)
            if not hasattr(self, '_negative_sampling_probs'):
                word_freqs = np.array([self.word_counts[self.idx_to_word[i]] 
                                     for i in range(self.vocab_size)])
                word_freqs = np.power(word_freqs, 0.75)
                self._negative_sampling_probs = word_freqs / np.sum(word_freqs)
            
            negative_samples = []
            attempts = 0
            max_attempts = self.negative_samples * 10  # Prevent infinite loops
            
            while len(negative_samples) < self.negative_samples and attempts < max_attempts:
                # Sample from the distribution
                candidate = np.random.choice(self.vocab_size, p=self._negative_sampling_probs)
                
                # Ensure it's not the target or positive context
                if candidate != target_idx and candidate != positive_context:
                    negative_samples.append(candidate)
                
                attempts += 1
            
            return negative_samples
            
        except Exception as e:
            logger.error(f"Error in negative sampling: {e}")
            return list(range(min(self.negative_samples, self.vocab_size)))
```

### 4. Neural Network Training with Gradient Descent

```python
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid function to prevent overflow."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def train_step(self, target_idx: int, context_idx: int) -> float:
        """
        Perform one training step using Skip-Gram with negative sampling.
        
        This implements the core learning algorithm:
        1. Forward pass: compute predictions for positive and negative samples
        2. Backward pass: compute gradients and update embeddings
        
        Args:
            target_idx: Index of target word
            context_idx: Index of context word
            
        Returns:
            Loss for this training step
        """
        try:
            # Get target word embedding
            target_embedding = self.W1[target_idx]  # Shape: (embedding_dim,)
            
            # POSITIVE SAMPLE
            # Compute positive score
            positive_score = np.dot(target_embedding, self.W2[context_idx])
            positive_prob = self.sigmoid(positive_score)
            
            # Compute positive loss: -log(σ(v_c · v_w))
            positive_loss = -np.log(positive_prob + 1e-10)  # Add epsilon for stability
            
            # Positive gradient for context embedding
            positive_error = positive_prob - 1  # d/dx of -log(sigmoid(x))
            context_grad = positive_error * target_embedding
            
            # Positive gradient for target embedding
            target_grad = positive_error * self.W2[context_idx]
            
            # NEGATIVE SAMPLES
            negative_samples = self.negative_sampling(target_idx, context_idx)
            negative_loss = 0
            
            for neg_idx in negative_samples:
                # Compute negative score
                negative_score = np.dot(target_embedding, self.W2[neg_idx])
                negative_prob = self.sigmoid(-negative_score)  # Note the negative sign
                
                # Compute negative loss: -log(σ(-v_n · v_w))
                negative_loss += -np.log(negative_prob + 1e-10)
                
                # Negative gradient for negative word embedding
                negative_error = -(1 - negative_prob)  # d/dx of -log(sigmoid(-x))
                self.W2[neg_idx] += self.learning_rate * negative_error * target_embedding
                
                # Add to target gradient
                target_grad += negative_error * self.W2[neg_idx]
            
            # UPDATE EMBEDDINGS
            # Update context word embedding
            self.W2[context_idx] += self.learning_rate * context_grad
            
            # Update target word embedding
            self.W1[target_idx] += self.learning_rate * target_grad
            
            total_loss = positive_loss + negative_loss
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return float('inf')
    
    def train(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        Train word embeddings on the provided texts.
        
        Args:
            texts: List of input text documents
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Build vocabulary and generate training data
            self.build_vocabulary(texts)
            self.generate_training_data(texts)
            
            # Initialize embedding matrices
            # Xavier initialization for better convergence
            std_dev = np.sqrt(2.0 / (self.vocab_size + self.embedding_dim))
            self.W1 = np.random.normal(0, std_dev, (self.vocab_size, self.embedding_dim))
            self.W2 = np.random.normal(0, std_dev, (self.vocab_size, self.embedding_dim))
            
            logger.info(f"Starting training for {self.epochs} epochs...")
            
            # Training metrics
            epoch_losses = []
            
            for epoch in range(self.epochs):
                epoch_loss = 0
                
                # Shuffle training pairs for better convergence
                random.shuffle(self.training_pairs)
                
                # Process each training pair
                for i, (target_idx, context_idx) in enumerate(self.training_pairs):
                    loss = self.train_step(target_idx, context_idx)
                    epoch_loss += loss
                    
                    # Log progress periodically
                    if i % 10000 == 0 and i > 0:
                        avg_loss = epoch_loss / (i + 1)
                        logger.info(f"Epoch {epoch+1}/{self.epochs}, "
                                   f"Step {i}/{len(self.training_pairs)}, "
                                   f"Avg Loss: {avg_loss:.4f}")
                
                # Calculate average loss for epoch
                avg_epoch_loss = epoch_loss / len(self.training_pairs)
                epoch_losses.append(avg_epoch_loss)
                
                # Decay learning rate
                self.learning_rate *= 0.95
                
                logger.info(f"Completed epoch {epoch+1}/{self.epochs}, "
                           f"Average Loss: {avg_epoch_loss:.4f}")
            
            logger.info("Training completed successfully!")
            
            return {
                'epoch_losses': epoch_losses,
                'final_vocab_size': self.vocab_size,
                'training_pairs': len(self.training_pairs)
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
```

![Training Curves](/assets/images/word-embeddings/training_curves.png)

## Byte Pair Encoding (BPE): Handling Subword Units

Traditional word-level tokenization struggles with out-of-vocabulary words and morphologically rich languages. BPE, introduced by Sennrich et al. (2016) ([Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)), provides an elegant solution through subword tokenization.

![BPE Tokenization Process](/assets/images/word-embeddings/bpe_tokenization_process.png)

The algorithm, originally developed by Gage (1994) for data compression ([A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)), iteratively merges the most frequent character pairs.

### BPE Implementation from Scratch

```python
class BytePairEncoder:
    """
    Byte Pair Encoding implementation for subword tokenization.
    
    Based on Sennrich et al. (2016): "Neural Machine Translation of Rare Words 
    with Subword Units" and the original BPE algorithm by Gage (1994).
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize BPE encoder.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.merges = {}  # (pair) -> merged_token
        self.vocab = set()  # Final vocabulary
        
        logger.info(f"Initialized BPE with target vocab size: {vocab_size}")
    
    def get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to character-level tokens with end-of-word marker.
        
        Args:
            word: Input word
            
        Returns:
            List of character tokens
        """
        try:
            # Add end-of-word marker to distinguish word boundaries
            return list(word) + ['</w>']
        except Exception as e:
            logger.error(f"Error tokenizing word '{word}': {e}")
            return ['</w>']
    
    def get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs in word tokens.
        
        Args:
            word_tokens: List of tokens
            
        Returns:
            Set of adjacent pairs
        """
        try:
            pairs = set()
            prev_char = word_tokens[0]
            
            for char in word_tokens[1:]:
                pairs.add((prev_char, char))
                prev_char = char
                
            return pairs
            
        except Exception as e:
            logger.error(f"Error getting pairs: {e}")
            return set()
    
    def train(self, corpus: List[str]) -> Dict[str, int]:
        """
        Train BPE on the provided corpus.
        
        Args:
            corpus: List of words to train on
            
        Returns:
            Final vocabulary with frequencies
        """
        try:
            logger.info("Starting BPE training...")
            
            # Initialize word frequency dictionary
            word_freqs = Counter(corpus)
            
            # Convert words to character tokens
            word_tokens = {}
            for word in word_freqs.keys():
                word_tokens[word] = self.get_word_tokens(word)
            
            # Initial vocabulary (all characters)
            self.vocab = set()
            for tokens in word_tokens.values():
                self.vocab.update(tokens)
            
            logger.info(f"Initial character vocabulary size: {len(self.vocab)}")
            
            # Iteratively merge most frequent pairs
            merge_count = 0
            while len(self.vocab) < self.vocab_size:
                # Count all pairs
                pair_counts = defaultdict(int)
                
                for word, freq in word_freqs.items():
                    word_token_list = word_tokens[word]
                    pairs = self.get_pairs(word_token_list)
                    
                    for pair in pairs:
                        pair_counts[pair] += freq
                
                # Find most frequent pair
                if not pair_counts:
                    logger.warning("No more pairs to merge")
                    break
                
                best_pair = max(pair_counts, key=pair_counts.get)
                best_freq = pair_counts[best_pair]
                
                # Merge the best pair
                new_token = best_pair[0] + best_pair[1]
                self.merges[best_pair] = new_token
                self.vocab.add(new_token)
                
                # Update word tokens
                for word in word_tokens:
                    word_tokens[word] = self.merge_tokens(word_tokens[word], best_pair, new_token)
                
                merge_count += 1
                
                if merge_count % 100 == 0:
                    logger.info(f"Merge {merge_count}: '{best_pair[0]}' + '{best_pair[1]}' -> "
                               f"'{new_token}' (freq: {best_freq})")
            
            logger.info(f"BPE training completed. Final vocabulary size: {len(self.vocab)}")
            
            # Return vocabulary with frequencies
            vocab_freqs = {}
            for word, freq in word_freqs.items():
                tokens = self.encode_word(word)
                for token in tokens:
                    vocab_freqs[token] = vocab_freqs.get(token, 0) + freq
            
            return vocab_freqs
            
        except Exception as e:
            logger.error(f"Error during BPE training: {e}")
            raise
    
    def merge_tokens(self, tokens: List[str], pair: Tuple[str, str], 
                    new_token: str) -> List[str]:
        """
        Merge specified pair in token list.
        
        Args:
            tokens: List of tokens
            pair: Pair to merge
            new_token: Replacement token
            
        Returns:
            Updated token list
        """
        try:
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == pair[0] and tokens[i + 1] == pair[1]):
                    # Merge the pair
                    new_tokens.append(new_token)
                    i += 2  # Skip both tokens
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            return new_tokens
            
        except Exception as e:
            logger.error(f"Error merging tokens: {e}")
            return tokens
    
    def encode_word(self, word: str) -> List[str]:
        """
        Encode a word using learned BPE merges.
        
        Args:
            word: Word to encode
            
        Returns:
            List of subword tokens
        """
        try:
            tokens = self.get_word_tokens(word)
            
            # Apply merges in order
            for pair, new_token in self.merges.items():
                tokens = self.merge_tokens(tokens, pair, new_token)
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            return [word, '</w>']
    
    def encode_text(self, text: str) -> List[str]:
        """
        Encode entire text using BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        try:
            words = text.lower().split()
            all_tokens = []
            
            for word in words:
                # Remove punctuation for this example
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    tokens = self.encode_word(clean_word)
                    all_tokens.extend(tokens)
            
            return all_tokens
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []
```

## Word Similarity and Clustering Analysis

Once we have trained embeddings, we can analyze semantic relationships through various mathematical operations.

![Similarity Matrix](/assets/images/word-embeddings/similarity_matrix.png)

### Similarity Computation and Evaluation

```python
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a specific word.
        
        Args:
            word: Target word
            
        Returns:
            Word embedding vector or None if word not in vocabulary
        """
        try:
            if word in self.word_to_idx:
                return self.W1[self.word_to_idx[word]]
            else:
                logger.warning(f"Word '{word}' not in vocabulary")
                return None
        except Exception as e:
            logger.error(f"Error getting vector for '{word}': {e}")
            return None
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Cosine similarity is the standard metric for word embedding similarity:
        cos(θ) = (A · B) / (||A|| * ||B||)
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        try:
            vec1 = self.get_word_vector(word1)
            vec2 = self.get_word_vector(word2)
            
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity between '{word1}' and '{word2}': {e}")
            return 0.0
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to a given word.
        
        Args:
            word: Target word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            target_vector = self.get_word_vector(word)
            if target_vector is None:
                return []
            
            similarities = []
            
            for vocab_word in self.word_to_idx.keys():
                if vocab_word != word:
                    similarity = self.cosine_similarity(word, vocab_word)
                    similarities.append((vocab_word, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar words for '{word}': {e}")
            return []
    
    def word_analogy(self, word_a: str, word_b: str, word_c: str) -> str:
        """
        Solve word analogies: A is to B as C is to ?
        
        Using vector arithmetic: embedding(B) - embedding(A) + embedding(C)
        This implements the famous "king - man + woman = queen" relationship.
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy  
            word_c: Third word in analogy
            
        Returns:
            Best candidate for the fourth word
        """
        try:
            vec_a = self.get_word_vector(word_a)
            vec_b = self.get_word_vector(word_b)
            vec_c = self.get_word_vector(word_c)
            
            if any(v is None for v in [vec_a, vec_b, vec_c]):
                logger.warning("One or more words not in vocabulary")
                return ""
            
            # Calculate target vector: B - A + C
            target_vector = vec_b - vec_a + vec_c
            
            # Find most similar word to target vector
            best_similarity = -1
            best_word = ""
            
            for word in self.word_to_idx.keys():
                if word in [word_a, word_b, word_c]:
                    continue
                
                word_vector = self.get_word_vector(word)
                if word_vector is not None:
                    similarity = np.dot(target_vector, word_vector) / (
                        np.linalg.norm(target_vector) * np.linalg.norm(word_vector)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_word = word
            
            return best_word
            
        except Exception as e:
            logger.error(f"Error solving analogy {word_a}:{word_b}::{word_c}:?: {e}")
            return ""
```

## Visualization and Analysis

![t-SNE Embeddings](/assets/images/word-embeddings/tsne_embeddings.png)

The t-SNE visualization shows how semantically related words cluster together in the high-dimensional embedding space. This dimensionality reduction technique, introduced by van der Maaten and Hinton (2008) ([Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)), helps us understand the learned semantic structure.

## Back to the Original Question: 20 Words = 20 Embeddings

```python
def demonstrate_individual_embeddings():
    """
    Demonstrate that each word gets its own embedding, even in a sentence.
    """
    # Example sentence
    sentence = "The quick brown fox jumps over the lazy dog and runs fast"
    words = sentence.lower().split()
    
    print(f"Sentence: '{sentence}'")
    print(f"Number of words: {len(words)}")
    print(f"Words: {words}")
    
    # Each word gets its own embedding
    embeddings = []
    for i, word in enumerate(words):
        embedding = trainer.get_word_vector(word)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Word {i+1}: '{word}' -> {embedding.shape} embedding")
        else:
            print(f"Word {i+1}: '{word}' -> Not in vocabulary")
    
    print(f"\nTotal embeddings created: {len(embeddings)}")
    print("Each word maintains its individual vector representation!")
    
    # To get a sentence-level representation, you would typically:
    # 1. Average the word embeddings (mean pooling)
    # 2. Use weighted averaging (TF-IDF weights)
    # 3. Use more sophisticated methods (attention mechanisms)
    
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
        print(f"\nSentence embedding (mean pooling): {sentence_embedding.shape}")

# Run the demonstration
demonstrate_individual_embeddings()
```

**Key Insight**: Each word in a sentence produces its own embedding vector. For sentence-level representations, we combine these individual embeddings through techniques like:

1. **Mean Pooling**: Average all word embeddings
2. **Weighted Averaging**: Use TF-IDF or attention weights
3. **LSTM/Transformer Encoders**: Learn contextual combinations

## Practical Usage Example

```python
def complete_example():
    """Complete example showing the entire pipeline."""
    
    # Sample corpus (in practice, use much larger datasets)
    corpus = [
        "The cat sat on the mat and looked around",
        "A dog ran quickly through the park",
        "Machine learning algorithms process data efficiently",
        "Neural networks learn complex patterns from examples",
        "Natural language processing enables computer understanding",
        "Deep learning models require large amounts of training data",
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence will transform many industries",
        "Python programming language is popular for data science",
        "Word embeddings capture semantic relationships between words"
    ]
    
    # Initialize and train
    trainer = WordEmbeddingTrainer(
        embedding_dim=100,  # Smaller for demo
        window_size=3,
        negative_samples=3,
        learning_rate=0.01,
        min_count=2,
        epochs=20
    )
    
    # Train the model
    metrics = trainer.train(corpus)
    
    # Test similarity
    print("\n=== Word Similarities ===")
    test_words = ['learning', 'data', 'quick']
    for word in test_words:
        similar = trainer.find_similar_words(word, top_k=3)
        print(f"Words similar to '{word}': {similar}")
    
    # Test analogies
    print("\n=== Word Analogies ===")
    analogies = [
        ('machine', 'learning', 'natural'),
        ('cat', 'dog', 'quick'),
    ]
    
    for a, b, c in analogies:
        result = trainer.word_analogy(a, b, c)
        print(f"{a} : {b} :: {c} : {result}")
    
    # Save the model
    trainer.save_model('word_embeddings_model.pkl')
    print("\nModel saved successfully!")

# Run complete example
if __name__ == "__main__":
    complete_example()
```

## Computational Complexity and Optimization

The training complexity is O(|C| × d × k) where:
- |C| = corpus size
- d = embedding dimension  
- k = negative samples

For large vocabularies, hierarchical softmax (Morin & Bengio, 2005) can reduce complexity from O(V) to O(log V) per training example.

## Modern Extensions and Applications

1. **Contextual Embeddings**: ELMo (Peters et al., 2018), BERT (Devlin et al., 2018)
2. **Subword Models**: FastText (Bojanowski et al., 2017)
3. **Cross-lingual Embeddings**: MUSE (Conneau et al., 2017)
4. **Domain-specific Embeddings**: BioBERT, FinBERT, etc.

## References and Further Reading

1. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013)**. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). *arXiv preprint arXiv:1301.3781*.

2. **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013)**. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). *Advances in Neural Information Processing Systems*.

3. **Sennrich, R., Haddow, B., & Birch, A. (2016)**. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). *Proceedings of ACL*.

4. **Gage, P. (1994)**. [A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf). *C Users Journal*.

5. **Pennington, J., Socher, R., & Manning, C. D. (2014)**. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). *Proceedings of EMNLP*.

6. **Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017)**. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606). *Transactions of ACL*.

## Conclusion

Word embeddings form the foundation of modern NLP, transforming discrete symbols into meaningful numerical representations. Understanding their mechanics - from Skip-Gram training to BPE tokenization - provides crucial insights into how AI systems like ChatGPT process and understand language.

**Remember**: Each word gets its own embedding, but the magic happens when we combine them intelligently to capture the meaning of larger linguistic units. This principle scales from simple averaging to the sophisticated attention mechanisms powering today's transformer models.

The journey from words to vectors is more than a technical exercise - it's a bridge between human language and machine understanding, enabling the AI revolution we're witnessing today. 