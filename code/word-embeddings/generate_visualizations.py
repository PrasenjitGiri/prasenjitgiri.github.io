"""
Word Embeddings Visualization Generator
======================================

This script generates all visualizations for the word embeddings blog post.
It creates professional charts and diagrams to illustrate key concepts.

Author: Prasenjit Giri
Date: January 2023
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for professional-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
os.makedirs('../../assets/images/word-embeddings', exist_ok=True)

def create_skipgram_architecture():
    """Create Skip-Gram architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 6), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6.5, 'Input\nWord\n"learning"', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # One-hot encoding
    onehot_box = FancyBboxPatch((0.5, 4), 1.5, 1, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(onehot_box)
    ax.text(1.25, 4.5, 'One-Hot\nEncoding\n[0,0,1,0,0...]', ha='center', va='center', fontsize=9)
    
    # Hidden layer (embedding)
    hidden_box = FancyBboxPatch((3.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(4.5, 5.75, 'Hidden Layer\n(Word Embedding)\n300 dimensions', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output layer
    output_boxes = []
    output_words = ['machine', 'deep', 'algorithm', 'neural']
    for i, word in enumerate(output_words):
        y_pos = 6.5 - i * 0.8
        output_box = FancyBboxPatch((7, y_pos - 0.3), 1.8, 0.6, boxstyle="round,pad=0.1",
                                    facecolor='lightyellow', edgecolor='black', linewidth=1)
        ax.add_patch(output_box)
        ax.text(7.9, y_pos, f'P("{word}")', ha='center', va='center', fontsize=9)
    
    # Weight matrices
    ax.text(2.5, 3, 'W₁\n(V × D)', ha='center', va='center', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    ax.text(6, 3, 'W₂\n(D × V)', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    # Arrows
    ax.annotate('', xy=(3.5, 5.75), xytext=(2, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, 5.5), xytext=(5.5, 5.75), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(1.25, 6), xytext=(1.25, 5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Context window illustration
    ax.text(1.25, 7.5, 'Context Window: "machine learning algorithm"', ha='center', va='center', 
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Title and labels
    ax.text(5, 0.5, 'Skip-Gram Architecture: Predicting Context from Target Word', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(0.5, 2, 'V = Vocabulary size\nD = Embedding dimension', 
            ha='left', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue'))
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/skipgram_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Skip-Gram architecture diagram created")

def create_embedding_space_visualization():
    """Create 2D visualization of word embedding space"""
    # Generate synthetic embeddings for demonstration
    np.random.seed(42)
    
    # Create semantic clusters
    tech_words = ['algorithm', 'neural', 'network', 'machine', 'learning', 'deep', 'artificial', 'intelligence']
    animal_words = ['cat', 'dog', 'bird', 'fish', 'lion', 'elephant', 'tiger', 'bear']
    color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown']
    
    # Generate embeddings with semantic clustering
    tech_embeddings = np.random.normal([2, 2], 0.5, (len(tech_words), 2))
    animal_embeddings = np.random.normal([-1, 1], 0.5, (len(animal_words), 2))
    color_embeddings = np.random.normal([0, -2], 0.5, (len(color_words), 2))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Semantic clusters
    ax1.scatter(tech_embeddings[:, 0], tech_embeddings[:, 1], 
               c='red', s=100, alpha=0.7, label='Technology')
    ax1.scatter(animal_embeddings[:, 0], animal_embeddings[:, 1], 
               c='blue', s=100, alpha=0.7, label='Animals')
    ax1.scatter(color_embeddings[:, 0], color_embeddings[:, 1], 
               c='green', s=100, alpha=0.7, label='Colors')
    
    # Add word labels
    for i, word in enumerate(tech_words):
        ax1.annotate(word, (tech_embeddings[i, 0], tech_embeddings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, word in enumerate(animal_words):
        ax1.annotate(word, (animal_embeddings[i, 0], animal_embeddings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, word in enumerate(color_words):
        ax1.annotate(word, (color_embeddings[i, 0], color_embeddings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax1.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax1.set_title('Word Embeddings: Semantic Clustering in 2D Space', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Similarity relationships
    # Create word analogy visualization: king - man + woman = queen
    king = np.array([1, 1])
    man = np.array([0.5, 0.8])
    woman = np.array([0.3, 0.2])
    queen = king - man + woman
    
    words_analogy = ['king', 'man', 'woman', 'queen']
    positions = [king, man, woman, queen]
    colors = ['gold', 'lightblue', 'pink', 'purple']
    
    for i, (word, pos, color) in enumerate(zip(words_analogy, positions, colors)):
        ax2.scatter(pos[0], pos[1], c=color, s=200, alpha=0.8)
        ax2.annotate(word, pos, xytext=(10, 10), textcoords='offset points', 
                    fontsize=12, fontweight='bold')
    
    # Draw vectors
    ax2.arrow(king[0], king[1], -man[0], -man[1], head_width=0.05, 
             head_length=0.05, fc='red', ec='red', alpha=0.7)
    ax2.arrow(king[0]-man[0], king[1]-man[1], woman[0], woman[1], 
             head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.7)
    
    ax2.text(0.5, -0.3, 'king - man + woman = queen\nVector Arithmetic in Embedding Space', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax2.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax2.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax2.set_title('Word Analogy: Vector Arithmetic', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/embedding_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Embedding space visualization created")

def create_bpe_tokenization_process():
    """Visualize BPE tokenization process"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Original text
    ax.text(5, 7.5, 'BPE Tokenization Process', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Step 1: Character-level
    ax.text(1, 6.5, 'Step 1: Initial Character Vocabulary', ha='left', va='center', 
            fontsize=12, fontweight='bold')
    chars = "l o w e r </w> n e w e s t </w> w i d e s t </w>"
    ax.text(1, 6, f'Text: "lower newest widest"', ha='left', va='center', fontsize=10)
    ax.text(1, 5.7, f'Chars: {chars}', ha='left', va='center', fontsize=10, 
            family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # Step 2: Most frequent pairs
    ax.text(1, 5, 'Step 2: Find Most Frequent Pairs', ha='left', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1, 4.7, 'Pair frequencies: (e,s): 6, (e,r): 2, (n,e): 2...', ha='left', va='center', fontsize=10)
    ax.text(1, 4.4, 'Merge: e + s → es', ha='left', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Step 3: After merging
    ax.text(1, 3.7, 'Step 3: After Merging "es"', ha='left', va='center', 
            fontsize=12, fontweight='bold')
    merged1 = "l o w e r </w> n e w es t </w> w i d es t </w>"
    ax.text(1, 3.4, f'Result: {merged1}', ha='left', va='center', fontsize=10, 
            family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Step 4: Continue merging
    ax.text(1, 2.7, 'Step 4: Continue Process', ha='left', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1, 2.4, 'Next frequent pair: (es, t) → est', ha='left', va='center', fontsize=10)
    merged2 = "l o w e r </w> n e w est </w> w i d est </w>"
    ax.text(1, 2.1, f'Result: {merged2}', ha='left', va='center', fontsize=10, 
            family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # Final vocabulary
    ax.text(1, 1.4, 'Final BPE Vocabulary:', ha='left', va='center', 
            fontsize=12, fontweight='bold')
    vocab = "l, o, w, e, r, n, i, d, </w>, es, est, new, low, wide"
    ax.text(1, 1.1, f'{vocab}', ha='left', va='center', fontsize=10, 
            family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue'))
    
    # Benefits box
    benefits_text = """Benefits of BPE:
• Handles OOV words
• Subword granularity
• Compact vocabulary
• Language agnostic"""
    
    ax.text(7.5, 4, benefits_text, ha='left', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', edgecolor='brown'))
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/bpe_tokenization_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ BPE tokenization process created")

def create_training_curves():
    """Create training loss curves"""
    # Simulate training data
    epochs = np.arange(1, 101)
    
    # Skip-gram loss (decreasing with some noise)
    skipgram_loss = 8 * np.exp(-epochs/20) + 0.5 + 0.1 * np.random.normal(0, 1, len(epochs))
    skipgram_loss = np.maximum(skipgram_loss, 0.3)  # Ensure positive
    
    # Negative sampling loss
    neg_sampling_loss = 6 * np.exp(-epochs/15) + 0.3 + 0.08 * np.random.normal(0, 1, len(epochs))
    neg_sampling_loss = np.maximum(neg_sampling_loss, 0.2)
    
    # Hierarchical softmax loss
    hierarchical_loss = 7 * np.exp(-epochs/18) + 0.4 + 0.12 * np.random.normal(0, 1, len(epochs))
    hierarchical_loss = np.maximum(hierarchical_loss, 0.25)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training loss comparison
    ax1.plot(epochs, skipgram_loss, label='Skip-gram (Basic)', linewidth=2, color='blue')
    ax1.plot(epochs, neg_sampling_loss, label='Negative Sampling', linewidth=2, color='red')
    ax1.plot(epochs, hierarchical_loss, label='Hierarchical Softmax', linewidth=2, color='green')
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison: Different Skip-Gram Variants', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 9)
    
    # Learning rate effects
    lr_high = 8 * np.exp(-epochs/25) + 0.5 + 0.2 * np.random.normal(0, 1, len(epochs))
    lr_medium = 7 * np.exp(-epochs/20) + 0.4 + 0.1 * np.random.normal(0, 1, len(epochs))
    lr_low = 6.5 * np.exp(-epochs/15) + 0.6 + 0.05 * np.random.normal(0, 1, len(epochs))
    
    lr_high = np.maximum(lr_high, 0.3)
    lr_medium = np.maximum(lr_medium, 0.2)
    lr_low = np.maximum(lr_low, 0.4)
    
    ax2.plot(epochs, lr_high, label='LR = 0.1 (High)', linewidth=2, color='orange')
    ax2.plot(epochs, lr_medium, label='LR = 0.01 (Medium)', linewidth=2, color='purple')
    ax2.plot(epochs, lr_low, label='LR = 0.001 (Low)', linewidth=2, color='brown')
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Impact of Learning Rate on Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 9)
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Training curves created")

def create_similarity_matrix():
    """Create word similarity matrix"""
    words = ['king', 'queen', 'man', 'woman', 'car', 'vehicle', 'cat', 'dog', 'happy', 'joy']
    
    # Create synthetic similarity matrix with semantic relationships
    np.random.seed(42)
    similarity_matrix = np.random.rand(len(words), len(words))
    
    # Make it symmetric
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    
    # Set diagonal to 1 (perfect self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Add semantic relationships
    # Royal words
    similarity_matrix[0, 1] = 0.85  # king-queen
    similarity_matrix[1, 0] = 0.85
    
    # Gender pairs
    similarity_matrix[0, 2] = 0.75  # king-man
    similarity_matrix[2, 0] = 0.75
    similarity_matrix[1, 3] = 0.75  # queen-woman
    similarity_matrix[3, 1] = 0.75
    
    # Vehicle category
    similarity_matrix[4, 5] = 0.9   # car-vehicle
    similarity_matrix[5, 4] = 0.9
    
    # Animals
    similarity_matrix[6, 7] = 0.8   # cat-dog
    similarity_matrix[7, 6] = 0.8
    
    # Emotions
    similarity_matrix[8, 9] = 0.9   # happy-joy
    similarity_matrix[9, 8] = 0.9
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Similarity matrix heatmap
    im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(words)))
    ax1.set_yticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45)
    ax1.set_yticklabels(words)
    ax1.set_title('Word Similarity Matrix\n(Cosine Similarity)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(words)):
        for j in range(len(words)):
            text = ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Create distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Hierarchical clustering visualization
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Convert to condensed distance matrix
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Create dendrogram
    dendrogram(linkage_matrix, labels=words, ax=ax2, orientation='top')
    ax2.set_title('Hierarchical Clustering of Words\n(Based on Embedding Similarity)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Distance', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Similarity matrix created")

def create_context_window_illustration():
    """Illustrate context window concept"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    words = sentence.split()
    
    # Draw words
    word_positions = []
    for i, word in enumerate(words):
        x = 1 + i * 1.2
        y = 4
        word_positions.append((x, y))
        
        # Highlight target word (jumps)
        if word == "jumps":
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.1",
                                  facecolor='lightcoral', edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, word, ha='center', va='center', fontsize=11, fontweight='bold')
            target_pos = (x, y)
        else:
            ax.text(x, y, word, ha='center', va='center', fontsize=11)
    
    # Context window illustration
    target_idx = words.index("jumps")
    window_size = 2
    
    # Draw context window
    for i in range(max(0, target_idx - window_size), min(len(words), target_idx + window_size + 1)):
        if i != target_idx:
            x, y = word_positions[i]
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='blue', linewidth=1, alpha=0.7)
            ax.add_patch(rect)
            
            # Draw arrow to target
            ax.annotate('', xy=target_pos, xytext=(x, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.7))
    
    # Labels and explanations
    ax.text(6, 5.5, 'Skip-Gram Context Window (size=2)', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    ax.text(6, 2.5, 'Target Word: "jumps" (red)', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax.text(6, 2, 'Context Words: "fox", "over", "the", "lazy" (blue)', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    ax.text(6, 1.5, 'Task: Predict context words given target word', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Window size examples
    ax.text(1, 0.5, 'Window size = 1: fox, over', ha='left', va='center', fontsize=10)
    ax.text(6, 0.5, 'Window size = 2: brown, fox, over, the', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/context_window_illustration.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Context window illustration created")

def create_negative_sampling_illustration():
    """Illustrate negative sampling concept"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Negative Sampling in Skip-Gram', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Target word
    target_box = FancyBboxPatch((4, 5.5), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(target_box)
    ax.text(5, 6, 'Target: "learning"', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Positive samples
    pos_words = ['machine', 'deep', 'algorithm']
    for i, word in enumerate(pos_words):
        x = 1 + i * 2.5
        y = 4
        pos_box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(pos_box)
        ax.text(x, y, word, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow from target
        ax.annotate('', xy=(x, y+0.3), xytext=(5, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Negative samples
    neg_words = ['banana', 'guitar', 'ocean', 'purple', 'running']
    for i, word in enumerate(neg_words):
        x = 0.5 + i * 1.8
        y = 2
        neg_box = FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.1",
                                 facecolor='lightpink', edgecolor='red', linewidth=1)
        ax.add_patch(neg_box)
        ax.text(x, y, word, ha='center', va='center', fontsize=9)
        
        # Dashed arrow from target
        ax.annotate('', xy=(x, y+0.25), xytext=(5, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='dashed', alpha=0.7))
    
    # Labels
    ax.text(4.5, 3.5, 'Positive Samples (k=3)\nActual context words', ha='center', va='center', 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.text(5, 1, 'Negative Samples (k=5)\nRandomly sampled non-context words', ha='center', va='center', 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink', alpha=0.7))
    
    # Objective function
    objective_text = """Objective: Maximize P(positive | target) 
           Minimize P(negative | target)
           
σ(v_c · v_w) for positive pairs
σ(-v_n · v_w) for negative pairs"""
    
    ax.text(8.5, 4, objective_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.4", facecolor='wheat', edgecolor='brown'),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/negative_sampling_illustration.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Negative sampling illustration created")

def create_tsne_embeddings():
    """Create t-SNE visualization of learned embeddings"""
    # Generate synthetic embeddings for realistic word clusters
    np.random.seed(42)
    
    # Word categories
    animals = ['cat', 'dog', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'rabbit']
    tech = ['computer', 'algorithm', 'neural', 'machine', 'artificial', 'learning', 'data', 'model']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown']
    emotions = ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'excited']
    
    all_words = animals + tech + colors + emotions
    n_words = len(all_words)
    
    # Generate high-dimensional embeddings (300D)
    embeddings_300d = np.random.normal(0, 1, (n_words, 300))
    
    # Add semantic structure
    # Animals cluster
    animal_center = np.random.normal(0, 1, 300)
    for i in range(len(animals)):
        embeddings_300d[i] = animal_center + np.random.normal(0, 0.3, 300)
    
    # Tech cluster
    tech_center = np.random.normal(2, 1, 300)
    for i in range(len(animals), len(animals) + len(tech)):
        embeddings_300d[i] = tech_center + np.random.normal(0, 0.3, 300)
    
    # Colors cluster
    color_center = np.random.normal(-1, 1, 300)
    for i in range(len(animals) + len(tech), len(animals) + len(tech) + len(colors)):
        embeddings_300d[i] = color_center + np.random.normal(0, 0.3, 300)
    
    # Emotions cluster
    emotion_center = np.random.normal(0, 2, 300)
    for i in range(len(animals) + len(tech) + len(colors), n_words):
        embeddings_300d[i] = emotion_center + np.random.normal(0, 0.3, 300)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_300d)
    
    # Apply PCA for comparison
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_300d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # t-SNE plot
    category_colors = ['red', 'blue', 'green', 'orange']
    category_labels = ['Animals', 'Technology', 'Colors', 'Emotions']
    
    start_idx = 0
    for cat_idx, (category, cat_color, cat_label) in enumerate(zip([animals, tech, colors, emotions], 
                                                                   category_colors, category_labels)):
        end_idx = start_idx + len(category)
        ax1.scatter(embeddings_2d[start_idx:end_idx, 0], embeddings_2d[start_idx:end_idx, 1], 
                   c=cat_color, s=100, alpha=0.7, label=cat_label)
        
        # Add word labels
        for i, word in enumerate(category):
            ax1.annotate(word, (embeddings_2d[start_idx + i, 0], embeddings_2d[start_idx + i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        start_idx = end_idx
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('t-SNE Visualization of Word Embeddings\n(300D → 2D)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCA plot
    start_idx = 0
    for cat_idx, (category, cat_color, cat_label) in enumerate(zip([animals, tech, colors, emotions], 
                                                                   category_colors, category_labels)):
        end_idx = start_idx + len(category)
        ax2.scatter(embeddings_pca[start_idx:end_idx, 0], embeddings_pca[start_idx:end_idx, 1], 
                   c=cat_color, s=100, alpha=0.7, label=cat_label)
        
        # Add word labels
        for i, word in enumerate(category):
            ax2.annotate(word, (embeddings_pca[start_idx + i, 0], embeddings_pca[start_idx + i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        start_idx = end_idx
    
    ax2.set_xlabel('First Principal Component', fontsize=12)
    ax2.set_ylabel('Second Principal Component', fontsize=12)
    ax2.set_title('PCA Visualization of Word Embeddings\n(300D → 2D)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../assets/images/word-embeddings/tsne_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ t-SNE embeddings visualization created")

# Main execution
if __name__ == "__main__":
    print("Generating word embeddings visualizations...")
    
    create_skipgram_architecture()
    create_embedding_space_visualization()
    create_bpe_tokenization_process()
    create_training_curves()
    create_similarity_matrix()
    create_context_window_illustration()
    create_negative_sampling_illustration()
    create_tsne_embeddings()
    
    print("\nAll visualizations generated successfully!")
    print("Files saved in: ../../assets/images/word-embeddings/") 