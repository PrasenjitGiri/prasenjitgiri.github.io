"""
Word Embeddings from Scratch Implementation
==========================================

A complete implementation of Skip-Gram word embeddings with negative sampling
and Byte Pair Encoding (BPE) tokenization.

This implementation demonstrates the mathematical foundations of word embeddings
and shows how models like ChatGPT process individual words.

Author: Prasenjit Giri
Date: January 2023
Blog: https://prasenjitgiri.github.io
"""

import numpy as np
import re
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
import pickle
import random
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WordEmbeddingTrainer:
    """
    Complete word embedding trainer implementing Skip-Gram with negative sampling.
    
    This implementation follows the mathematical foundations from:
    - Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
    - Mikolov et al. (2013): "Distributed Representations of Words and Phrases"
    
    The key insight: Each word gets its own embedding vector, even in a sentence.
    For a 20-word sentence, you get 20 individual embeddings, not one combined.
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
        
        # Embedding matrices - the heart of the model
        self.W1 = None  # Input embeddings (what we ultimately use)
        self.W2 = None  # Output embeddings (for prediction)
        
        # Training data
        self.training_pairs = []
        
        # Training history
        self.training_history = {
            'epoch_losses': [],
            'timestamps': [],
            'vocab_size': 0,
            'total_pairs': 0
        }
        
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
            
            # Update training history
            self.training_history['vocab_size'] = self.vocab_size
            
            logger.info(f"Built vocabulary: {self.vocab_size} words "
                       f"(filtered from {len(self.word_counts)} total)")
            
            if self.vocab_size == 0:
                raise ValueError("Empty vocabulary after filtering. Lower min_count or provide more text.")
                
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            raise
    
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
            
            # Update training history
            self.training_history['total_pairs'] = len(self.training_pairs)
            
            logger.info(f"Generated {len(self.training_pairs)} training pairs")
            
            if len(self.training_pairs) == 0:
                raise ValueError("No training pairs generated. Check vocabulary and input texts.")
                
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            raise
    
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
            target_embedding = self.W1[target_idx].copy()  # Shape: (embedding_dim,)
            
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
                epoch_start_time = datetime.now()
                
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
                
                # Update training history
                self.training_history['epoch_losses'].append(avg_epoch_loss)
                self.training_history['timestamps'].append(datetime.now().isoformat())
                
                # Decay learning rate
                self.learning_rate *= 0.95
                
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                logger.info(f"Completed epoch {epoch+1}/{self.epochs}, "
                           f"Average Loss: {avg_epoch_loss:.4f}, "
                           f"Time: {epoch_time:.2f}s")
            
            logger.info("Training completed successfully!")
            
            return {
                'epoch_losses': epoch_losses,
                'final_vocab_size': self.vocab_size,
                'training_pairs': len(self.training_pairs),
                'final_learning_rate': self.learning_rate
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
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
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        try:
            model_data = {
                'W1': self.W1,
                'W2': self.W2,
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'word_counts': dict(self.word_counts),
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'training_history': self.training_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.W1 = model_data['W1']
            self.W2 = model_data['W2']
            self.word_to_idx = model_data['word_to_idx']
            self.idx_to_word = model_data['idx_to_word']
            self.word_counts = Counter(model_data['word_counts'])
            self.vocab_size = model_data['vocab_size']
            self.embedding_dim = model_data['embedding_dim']
            self.training_history = model_data.get('training_history', {})
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


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


def demonstrate_individual_embeddings(trainer: WordEmbeddingTrainer):
    """
    Demonstrate that each word gets its own embedding, even in a sentence.
    This answers the core question: 20 words = 20 individual embeddings.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING: 20 WORDS = 20 INDIVIDUAL EMBEDDINGS")
    print("="*60)
    
    # Example sentence
    sentence = "The quick brown fox jumps over the lazy dog and runs fast"
    words = sentence.lower().split()
    
    print(f"Sentence: '{sentence}'")
    print(f"Number of words: {len(words)}")
    print(f"Words: {words}")
    print("\n" + "-"*40)
    
    # Each word gets its own embedding
    embeddings = []
    for i, word in enumerate(words):
        embedding = trainer.get_word_vector(word)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Word {i+1:2d}: '{word:6s}' -> {embedding.shape} embedding (first 3 dims: {embedding[:3]})")
        else:
            print(f"Word {i+1:2d}: '{word:6s}' -> Not in vocabulary")
    
    print("\n" + "-"*40)
    print(f"Total embeddings created: {len(embeddings)}")
    print("✓ Each word maintains its individual vector representation!")
    
    # To get a sentence-level representation, you would typically:
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
        print(f"\nSentence embedding (mean pooling): {sentence_embedding.shape}")
        print(f"First 5 dimensions: {sentence_embedding[:5]}")
        
        print("\nMethods to combine word embeddings into sentence embeddings:")
        print("1. Mean Pooling (averaging) - what we just did")
        print("2. Weighted averaging (TF-IDF weights)")
        print("3. Max pooling (element-wise maximum)")
        print("4. LSTM/GRU encoders (like in older models)")
        print("5. Attention mechanisms (like in Transformers/ChatGPT)")


def demonstrate_real_world_usage():
    """
    Show how word embeddings are used in real applications.
    """
    print("\n" + "="*60)
    print("REAL-WORLD USAGE: HOW MODELS LIKE CHATGPT USE EMBEDDINGS")
    print("="*60)
    
    print("""
1. TEXT CLASSIFICATION:
   Input: "This movie is amazing!"
   Process: [This, movie, is, amazing] -> [emb1, emb2, emb3, emb4] -> average -> classifier
   Output: Positive sentiment
   
2. MACHINE TRANSLATION:
   Input: "Hello world"
   Process: [Hello, world] -> [emb1, emb2] -> encoder -> decoder -> [Hola, mundo]
   
3. QUESTION ANSWERING:
   Question: "What is the capital of France?"
   Context: "Paris is the capital and largest city of France..."
   Process: Both get converted to embeddings, similarity computed
   
4. CHATGPT-STYLE MODELS:
   Input: "Explain quantum physics"
   Process: 
   - Tokenize: ["Explain", "quantum", "physics"]
   - Convert to embeddings: [emb1, emb2, emb3]
   - Add positional encodings
   - Pass through transformer layers
   - Generate response token by token
   
5. SEARCH ENGINES:
   Query: "best restaurants near me"
   Process: Convert to embeddings, compare with document embeddings
   Return: Most similar restaurant reviews/websites
""")


def complete_example():
    """Complete example showing the entire pipeline."""
    
    print("="*80)
    print("WORD EMBEDDINGS FROM SCRATCH: COMPLETE EXAMPLE")
    print("="*80)
    
    # Sample corpus (in practice, use much larger datasets)
    corpus = [
        "The cat sat on the mat and looked around the room",
        "A dog ran quickly through the park and played with children",
        "Machine learning algorithms process data efficiently and accurately",
        "Neural networks learn complex patterns from examples and training data",
        "Natural language processing enables computer understanding of human text",
        "Deep learning models require large amounts of training data to work well",
        "The quick brown fox jumps over the lazy dog in the field",
        "Artificial intelligence will transform many industries in the coming years",
        "Python programming language is popular for data science and machine learning",
        "Word embeddings capture semantic relationships between words and concepts",
        "ChatGPT represents a breakthrough in conversational artificial intelligence systems",
        "Language models use attention mechanisms to understand context and meaning",
        "Transformers have revolutionized natural language processing applications worldwide",
        "Researchers continue advancing the field of artificial intelligence rapidly",
        "Data scientists use various tools and techniques for analysis and modeling"
    ]
    
    print(f"Training corpus: {len(corpus)} sentences")
    
    # Initialize and train
    trainer = WordEmbeddingTrainer(
        embedding_dim=100,  # Smaller for demo
        window_size=3,
        negative_samples=3,
        learning_rate=0.01,
        min_count=2,
        epochs=20
    )
    
    print("\nStarting training...")
    # Train the model
    metrics = trainer.train(corpus)
    
    print(f"\nTraining completed!")
    print(f"Final vocabulary size: {metrics['final_vocab_size']}")
    print(f"Training pairs: {metrics['training_pairs']}")
    
    # Demonstrate individual embeddings
    demonstrate_individual_embeddings(trainer)
    
    # Test similarity
    print("\n" + "="*40)
    print("WORD SIMILARITIES")
    print("="*40)
    test_words = ['learning', 'data', 'quick', 'intelligence']
    for word in test_words:
        similar = trainer.find_similar_words(word, top_k=3)
        if similar:
            print(f"Words similar to '{word}': {similar}")
        else:
            print(f"'{word}' not in vocabulary or no similar words found")
    
    # Test analogies
    print("\n" + "="*40)
    print("WORD ANALOGIES")
    print("="*40)
    analogies = [
        ('machine', 'learning', 'natural'),
        ('data', 'science', 'artificial'),
        ('quick', 'brown', 'lazy'),
    ]
    
    for a, b, c in analogies:
        result = trainer.word_analogy(a, b, c)
        print(f"{a} : {b} :: {c} : {result if result else 'not found'}")
    
    # Show real-world usage
    demonstrate_real_world_usage()
    
    # Save the model
    model_path = 'code/word-embeddings/trained_model.pkl'
    trainer.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return trainer


# Example usage with BPE
def demonstrate_bpe():
    """Demonstrate BPE tokenization."""
    print("\n" + "="*60)
    print("BYTE PAIR ENCODING (BPE) DEMONSTRATION")
    print("="*60)
    
    # Sample corpus for BPE
    bpe_corpus = [
        "lower", "lowest", "newer", "newest", "wider", "widest",
        "faster", "fastest", "slower", "slowest", "better", "best",
        "running", "runner", "walking", "walker", "jumping", "jumper"
    ]
    
    print(f"BPE training corpus: {bpe_corpus}")
    
    # Initialize and train BPE
    bpe = BytePairEncoder(vocab_size=50)
    vocab_freqs = bpe.train(bpe_corpus)
    
    print(f"\nFinal BPE vocabulary size: {len(vocab_freqs)}")
    print("Sample BPE vocabulary:", list(vocab_freqs.keys())[:10])
    
    # Test encoding
    test_words = ["running", "fastest", "unknown"]
    print(f"\nBPE encoding examples:")
    for word in test_words:
        encoded = bpe.encode_word(word)
        print(f"'{word}' -> {encoded}")
    
    return bpe


if __name__ == "__main__":
    print("Starting Word Embeddings from Scratch demonstration...")
    
    # Run complete example
    trainer = complete_example()
    
    # Demonstrate BPE
    bpe = demonstrate_bpe()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("""
Key Takeaways:
1. Each word gets its own embedding vector (20 words = 20 embeddings)
2. Embeddings capture semantic relationships through vector arithmetic  
3. Models like ChatGPT use these principles but with more sophisticated architectures
4. BPE helps handle out-of-vocabulary words through subword tokenization
5. This is the foundation that enables modern language understanding

Next steps:
- Try training on larger datasets
- Experiment with different hyperparameters
- Explore pre-trained embeddings like Word2Vec, GloVe
- Learn about contextual embeddings (BERT, GPT)
""") 