"""
Word Embeddings Demo
===================

A simple demo script showing how to use our word embeddings implementation.
Run this script to see word embeddings in action!

Usage: python demo.py
"""

from word_embeddings_from_scratch import WordEmbeddingTrainer, demonstrate_individual_embeddings, demonstrate_real_world_usage
import numpy as np

def main():
    print("=" * 80)
    print("WORD EMBEDDINGS FROM SCRATCH - DEMO")
    print("=" * 80)
    print("This demo proves: 20 words = 20 individual embeddings")
    print("Showing how ChatGPT and other AI models process text")
    print()
    
    # Quick training corpus
    corpus = [
        "The cat sat on the mat and looked around the room",
        "Machine learning algorithms process data efficiently and accurately",
        "Neural networks learn complex patterns from training examples",
        "Natural language processing enables computer understanding of text",
        "ChatGPT represents a breakthrough in conversational AI systems",
        "Deep learning models require large amounts of training data",
        "Artificial intelligence will transform many industries rapidly",
        "Python programming language is popular for data science",
        "Word embeddings capture semantic relationships between words",
        "Language models use attention mechanisms for understanding"
    ]
    
    print(f"Training on {len(corpus)} sentences...")
    
    # Train quickly with smaller parameters
    trainer = WordEmbeddingTrainer(
        embedding_dim=50,   # Smaller for faster demo
        window_size=2,
        negative_samples=3,
        learning_rate=0.02,
        min_count=1,
        epochs=15
    )
    
    metrics = trainer.train(corpus)
    
    print(f"\n✅ Training completed!")
    print(f"   Vocabulary size: {metrics['final_vocab_size']}")
    print(f"   Training pairs: {metrics['training_pairs']}")
    
    # Demonstrate the core concept
    demonstrate_individual_embeddings(trainer)
    
    # Show real-world usage
    demonstrate_real_world_usage()
    
    # Test word similarities
    print("\n" + "=" * 40)
    print("TESTING WORD SIMILARITIES")
    print("=" * 40)
    
    test_pairs = [
        ('machine', 'learning'),
        ('neural', 'networks'),
        ('language', 'processing'),
        ('data', 'training')
    ]
    
    for word1, word2 in test_pairs:
        similarity = trainer.cosine_similarity(word1, word2)
        print(f"'{word1}' ↔ '{word2}': {similarity:.4f}")
    
    # Show individual word vectors
    print("\n" + "=" * 40)
    print("EXAMINING WORD VECTORS")
    print("=" * 40)
    
    sample_words = ['machine', 'learning', 'neural', 'data']
    for word in sample_words:
        vector = trainer.get_word_vector(word)
        if vector is not None:
            print(f"'{word}': shape={vector.shape}, norm={np.linalg.norm(vector):.3f}")
            print(f"  First 5 dims: {vector[:5]}")
        else:
            print(f"'{word}': Not in vocabulary")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Key takeaways:")
    print("✓ Each word gets its own unique embedding vector")
    print("✓ 20-word sentence = 20 individual embeddings")
    print("✓ Semantic similarity emerges from context training")
    print("✓ This is the foundation of ChatGPT and other AI models")
    print("\nTry the full implementation in:")
    print("📓 Jupyter notebook: Word_Embeddings_From_Scratch.ipynb")
    print("🐍 Python script: word_embeddings_from_scratch.py")
    print("📊 Visualizations: generate_visualizations.py")

if __name__ == "__main__":
    main() 