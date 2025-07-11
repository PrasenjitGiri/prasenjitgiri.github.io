#!/usr/bin/env python3
"""
Recommender Systems Demo Script
==============================

This script demonstrates the usage of different recommender system algorithms
on sample e-commerce data. It showcases collaborative filtering, matrix 
factorization, content-based filtering, and hybrid approaches.

Author: Prasenjit Giri
Date: January 2025
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for demo
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RecommenderSystemDemo:
    """Demo class for showcasing recommender system capabilities"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        np.random.seed(random_state)
        
        # Data containers
        self.users = None
        self.items = None
        self.interactions = None
        self.train_data = None
        self.test_data = None
        
        # Models (would import from the main implementation)
        self.models = {}
        
    def generate_synthetic_data(self) -> None:
        """Generate synthetic e-commerce data for demonstration"""
        try:
            self.logger.info("Generating synthetic e-commerce data...")
            
            # Generate users
            n_users = 1000
            self.users = pd.DataFrame({
                'user_id': range(n_users),
                'age': np.random.randint(18, 70, n_users),
                'gender': np.random.choice(['M', 'F'], n_users),
                'income_bracket': np.random.choice(['Low', 'Medium', 'High'], n_users),
                'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_users)
            })
            
            # Generate items (electronic products)
            n_items = 500
            categories = ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras', 'Smartwatches']
            brands = ['Apple', 'Samsung', 'Sony', 'Dell', 'HP', 'Canon', 'Bose', 'Microsoft']
            
            self.items = pd.DataFrame({
                'item_id': range(n_items),
                'category': np.random.choice(categories, n_items),
                'brand': np.random.choice(brands, n_items),
                'price': np.random.uniform(50, 3000, n_items),
                'avg_rating': np.random.uniform(3.0, 5.0, n_items),
                'num_reviews': np.random.randint(10, 1000, n_items),
                'description': [f"High-quality {cat.lower()} from {brand}" 
                              for cat, brand in zip(np.random.choice(categories, n_items), 
                                                  np.random.choice(brands, n_items))]
            })
            
            # Generate interactions with realistic patterns
            n_interactions = 15000
            
            # Create preference patterns based on user demographics
            user_category_preferences = {}
            for user_id in range(n_users):
                user = self.users.iloc[user_id]
                preferences = {}
                
                # Age-based preferences
                if user['age'] < 30:
                    preferences = {'Smartphones': 0.3, 'Tablets': 0.2, 'Headphones': 0.25, 
                                 'Smartwatches': 0.15, 'Laptops': 0.08, 'Cameras': 0.02}
                elif user['age'] < 50:
                    preferences = {'Laptops': 0.25, 'Smartphones': 0.2, 'Cameras': 0.2,
                                 'Tablets': 0.15, 'Headphones': 0.15, 'Smartwatches': 0.05}
                else:
                    preferences = {'Cameras': 0.3, 'Laptops': 0.25, 'Tablets': 0.2,
                                 'Smartphones': 0.15, 'Headphones': 0.08, 'Smartwatches': 0.02}
                
                user_category_preferences[user_id] = preferences
            
            # Generate interactions based on preferences
            interactions_list = []
            for _ in range(n_interactions):
                user_id = np.random.randint(0, n_users)
                
                # Select category based on user preferences
                preferences = user_category_preferences[user_id]
                category = np.random.choice(list(preferences.keys()), p=list(preferences.values()))
                
                # Select item from preferred category
                category_items = self.items[self.items['category'] == category]['item_id'].values
                if len(category_items) > 0:
                    item_id = np.random.choice(category_items)
                    
                    # Generate rating based on category preference and item quality
                    base_rating = preferences[category] * 5  # Base rating from preference
                    item_quality = self.items.loc[self.items['item_id'] == item_id, 'avg_rating'].iloc[0]
                    
                    # Combine preference and quality with some noise
                    rating = 0.7 * base_rating + 0.3 * item_quality + np.random.normal(0, 0.3)
                    rating = max(1, min(5, round(rating)))  # Clamp to 1-5 range
                    
                    interactions_list.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating,
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
                    })
            
            self.interactions = pd.DataFrame(interactions_list)
            
            # Remove duplicates (keep highest rating)
            self.interactions = self.interactions.sort_values('rating', ascending=False)
            self.interactions = self.interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
            
            self.logger.info(f"Generated {len(self.users)} users, {len(self.items)} items, "
                           f"{len(self.interactions)} interactions")
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def explore_data(self) -> None:
        """Explore and visualize the generated data"""
        try:
            self.logger.info("Exploring generated data...")
            
            print("\n" + "="*60)
            print("DATA EXPLORATION SUMMARY")
            print("="*60)
            
            # User statistics
            print(f"\nUSER STATISTICS:")
            print(f"Total users: {len(self.users)}")
            print(f"Age distribution: {self.users['age'].describe()}")
            print(f"Gender distribution:\n{self.users['gender'].value_counts()}")
            
            # Item statistics
            print(f"\nITEM STATISTICS:")
            print(f"Total items: {len(self.items)}")
            print(f"Categories:\n{self.items['category'].value_counts()}")
            print(f"Price range: ${self.items['price'].min():.2f} - ${self.items['price'].max():.2f}")
            
            # Interaction statistics
            print(f"\nINTERACTION STATISTICS:")
            print(f"Total interactions: {len(self.interactions)}")
            print(f"Rating distribution:\n{self.interactions['rating'].value_counts().sort_index()}")
            print(f"Sparsity: {1 - len(self.interactions) / (len(self.users) * len(self.items)):.4f}")
            
            # User activity distribution
            user_activity = self.interactions.groupby('user_id').size()
            print(f"\nUser activity (interactions per user):")
            print(f"Mean: {user_activity.mean():.2f}")
            print(f"Median: {user_activity.median():.2f}")
            print(f"Max: {user_activity.max()}")
            
            # Item popularity distribution
            item_popularity = self.interactions.groupby('item_id').size()
            print(f"\nItem popularity (interactions per item):")
            print(f"Mean: {item_popularity.mean():.2f}")
            print(f"Median: {item_popularity.median():.2f}")
            print(f"Max: {item_popularity.max()}")
            
        except Exception as e:
            self.logger.error(f"Error exploring data: {e}")
            raise
    
    def split_data(self, test_ratio: float = 0.2) -> None:
        """Split data into train and test sets"""
        try:
            self.logger.info(f"Splitting data with test ratio: {test_ratio}")
            
            # Group by user to ensure each user has data in both sets
            user_groups = self.interactions.groupby('user_id')
            train_data = []
            test_data = []
            
            for user_id, user_interactions in user_groups:
                if len(user_interactions) >= 2:  # Need at least 2 interactions
                    user_interactions = user_interactions.sample(frac=1, random_state=self.random_state)
                    n_test = max(1, int(len(user_interactions) * test_ratio))
                    
                    test_data.append(user_interactions.head(n_test))
                    train_data.append(user_interactions.tail(len(user_interactions) - n_test))
                else:
                    # Put single interaction in training
                    train_data.append(user_interactions)
            
            self.train_data = pd.concat(train_data, ignore_index=True)
            self.test_data = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
            
            self.logger.info(f"Split complete: {len(self.train_data)} train, {len(self.test_data)} test")
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
    
    def demonstrate_collaborative_filtering(self) -> None:
        """Demonstrate collaborative filtering approach"""
        try:
            self.logger.info("Demonstrating Collaborative Filtering...")
            
            print("\n" + "="*60)
            print("COLLABORATIVE FILTERING DEMONSTRATION")
            print("="*60)
            
            # Create user-item matrix
            user_item_matrix = self.train_data.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating', 
                fill_value=0
            )
            
            print(f"User-item matrix shape: {user_item_matrix.shape}")
            print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / user_item_matrix.size:.4f}")
            
            # Simple item-based collaborative filtering
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute item similarity matrix
            item_similarity = cosine_similarity(user_item_matrix.T)
            item_similarity_df = pd.DataFrame(
                item_similarity,
                index=user_item_matrix.columns,
                columns=user_item_matrix.columns
            )
            
            # Find most similar items to a sample item
            sample_item = user_item_matrix.columns[0]
            similar_items = item_similarity_df[sample_item].sort_values(ascending=False)
            
            print(f"\nTop 5 items similar to item {sample_item}:")
            for i, (item_id, similarity) in enumerate(similar_items.head(6).items()):
                if item_id != sample_item:  # Skip self
                    item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                    print(f"{i}. Item {item_id} ({item_info['category']} - {item_info['brand']}) - "
                          f"Similarity: {similarity:.3f}")
            
            # Generate recommendations for a sample user
            sample_user = user_item_matrix.index[0]
            user_ratings = user_item_matrix.loc[sample_user]
            unrated_items = user_ratings[user_ratings == 0].index
            
            if len(unrated_items) > 0:
                # Predict ratings for unrated items
                predictions = []
                for item in unrated_items[:20]:  # Limit for demo
                    # Find items rated by user that are similar to this item
                    rated_items = user_ratings[user_ratings > 0].index
                    similarities = item_similarity_df[item][rated_items]
                    
                    if similarities.sum() > 0:
                        predicted_rating = (similarities * user_ratings[rated_items]).sum() / similarities.sum()
                        predictions.append((item, predicted_rating))
                
                # Sort by predicted rating
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\nTop 5 recommendations for user {sample_user}:")
                for i, (item_id, pred_rating) in enumerate(predictions[:5]):
                    item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                    print(f"{i+1}. Item {item_id} ({item_info['category']} - {item_info['brand']}) - "
                          f"Predicted Rating: {pred_rating:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in collaborative filtering demo: {e}")
            raise
    
    def demonstrate_content_based_filtering(self) -> None:
        """Demonstrate content-based filtering"""
        try:
            self.logger.info("Demonstrating Content-Based Filtering...")
            
            print("\n" + "="*60)
            print("CONTENT-BASED FILTERING DEMONSTRATION")
            print("="*60)
            
            # Create item content matrix
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler
            
            # Process categorical features
            item_features = pd.get_dummies(self.items[['category', 'brand']], prefix=['cat', 'brand'])
            
            # Process numerical features
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(self.items[['price', 'avg_rating', 'num_reviews']])
            numerical_df = pd.DataFrame(
                numerical_features, 
                columns=['price_scaled', 'rating_scaled', 'reviews_scaled']
            )
            
            # Process text features
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            text_features = vectorizer.fit_transform(self.items['description']).toarray()
            text_df = pd.DataFrame(text_features, columns=[f'text_{i}' for i in range(text_features.shape[1])])
            
            # Combine all features
            content_matrix = pd.concat([
                item_features.reset_index(drop=True),
                numerical_df.reset_index(drop=True),
                text_df.reset_index(drop=True)
            ], axis=1)
            
            content_matrix.index = self.items['item_id']
            
            print(f"Content matrix shape: {content_matrix.shape}")
            
            # Compute content similarity
            content_similarity = cosine_similarity(content_matrix)
            content_similarity_df = pd.DataFrame(
                content_similarity,
                index=content_matrix.index,
                columns=content_matrix.index
            )
            
            # Find content-similar items
            sample_item = content_matrix.index[0]
            similar_items = content_similarity_df[sample_item].sort_values(ascending=False)
            
            print(f"\nTop 5 content-similar items to item {sample_item}:")
            sample_item_info = self.items[self.items['item_id'] == sample_item].iloc[0]
            print(f"Reference: {sample_item_info['category']} - {sample_item_info['brand']} - ${sample_item_info['price']:.2f}")
            
            for i, (item_id, similarity) in enumerate(similar_items.head(6).items()):
                if item_id != sample_item:
                    item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                    print(f"{i}. Item {item_id} ({item_info['category']} - {item_info['brand']} - "
                          f"${item_info['price']:.2f}) - Similarity: {similarity:.3f}")
            
            # Generate content-based recommendations for a user
            sample_user = self.train_data['user_id'].iloc[0]
            user_items = self.train_data[self.train_data['user_id'] == sample_user]
            
            if len(user_items) > 0:
                # Create user profile as average of liked items' features
                liked_items = user_items[user_items['rating'] >= 4]['item_id'].values
                
                if len(liked_items) > 0:
                    user_profile = content_matrix.loc[liked_items].mean()
                    
                    # Compute similarity with all items
                    item_scores = content_matrix.apply(
                        lambda x: cosine_similarity([user_profile], [x])[0][0], axis=1
                    )
                    
                    # Remove already rated items
                    rated_items = user_items['item_id'].values
                    unrated_scores = item_scores[~item_scores.index.isin(rated_items)]
                    
                    recommendations = unrated_scores.sort_values(ascending=False).head(5)
                    
                    print(f"\nTop 5 content-based recommendations for user {sample_user}:")
                    print(f"Based on {len(liked_items)} liked items")
                    
                    for i, (item_id, score) in enumerate(recommendations.items()):
                        item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                        print(f"{i+1}. Item {item_id} ({item_info['category']} - {item_info['brand']}) - "
                              f"Score: {score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in content-based filtering demo: {e}")
            raise
    
    def demonstrate_matrix_factorization(self) -> None:
        """Demonstrate matrix factorization approach"""
        try:
            self.logger.info("Demonstrating Matrix Factorization...")
            
            print("\n" + "="*60)
            print("MATRIX FACTORIZATION DEMONSTRATION")
            print("="*60)
            
            from sklearn.decomposition import NMF
            
            # Create user-item matrix
            user_item_matrix = self.train_data.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating', 
                fill_value=0
            )
            
            # Apply NMF
            n_components = 20
            model = NMF(n_components=n_components, random_state=self.random_state, max_iter=100)
            
            user_factors = model.fit_transform(user_item_matrix.values)
            item_factors = model.components_
            
            print(f"Matrix factorization complete:")
            print(f"Original matrix shape: {user_item_matrix.shape}")
            print(f"User factors shape: {user_factors.shape}")
            print(f"Item factors shape: {item_factors.shape}")
            print(f"Reconstruction error: {model.reconstruction_err_:.4f}")
            
            # Generate predictions
            reconstructed_matrix = user_factors @ item_factors
            reconstructed_df = pd.DataFrame(
                reconstructed_matrix,
                index=user_item_matrix.index,
                columns=user_item_matrix.columns
            )
            
            # Generate recommendations for sample user
            sample_user_idx = 0
            sample_user = user_item_matrix.index[sample_user_idx]
            
            user_ratings = user_item_matrix.iloc[sample_user_idx]
            predicted_ratings = reconstructed_df.iloc[sample_user_idx]
            
            # Get unrated items
            unrated_mask = user_ratings == 0
            unrated_predictions = predicted_ratings[unrated_mask].sort_values(ascending=False)
            
            print(f"\nTop 5 matrix factorization recommendations for user {sample_user}:")
            for i, (item_id, pred_rating) in enumerate(unrated_predictions.head(5).items()):
                item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                print(f"{i+1}. Item {item_id} ({item_info['category']} - {item_info['brand']}) - "
                      f"Predicted Rating: {pred_rating:.2f}")
            
            # Show factor interpretation
            print(f"\nUser factor analysis for user {sample_user}:")
            user_factor_values = user_factors[sample_user_idx]
            top_factors = np.argsort(user_factor_values)[-3:][::-1]
            
            for i, factor_idx in enumerate(top_factors):
                factor_value = user_factor_values[factor_idx]
                print(f"Factor {factor_idx}: {factor_value:.3f}")
                
                # Find items most associated with this factor
                item_factor_values = item_factors[factor_idx]
                top_items = np.argsort(item_factor_values)[-3:][::-1]
                
                print(f"  Top items for this factor:")
                for item_idx in top_items:
                    item_id = user_item_matrix.columns[item_idx]
                    item_info = self.items[self.items['item_id'] == item_id].iloc[0]
                    print(f"    Item {item_id} ({item_info['category']} - {item_info['brand']})")
            
        except Exception as e:
            self.logger.error(f"Error in matrix factorization demo: {e}")
            raise
    
    def evaluate_recommendations(self) -> None:
        """Evaluate recommendation quality using test data"""
        try:
            self.logger.info("Evaluating recommendation quality...")
            
            print("\n" + "="*60)
            print("RECOMMENDATION EVALUATION")
            print("="*60)
            
            if self.test_data.empty:
                print("No test data available for evaluation")
                return
            
            # Simple evaluation: precision and recall for high-rated items
            test_users = self.test_data['user_id'].unique()[:20]  # Sample for demo
            
            precision_scores = []
            recall_scores = []
            
            for user_id in test_users:
                try:
                    # Get user's test items with high ratings (4-5)
                    user_test = self.test_data[
                        (self.test_data['user_id'] == user_id) & 
                        (self.test_data['rating'] >= 4)
                    ]
                    
                    if len(user_test) == 0:
                        continue
                    
                    relevant_items = set(user_test['item_id'].values)
                    
                    # Generate recommendations using simple popularity-based approach
                    # (In practice, you'd use the trained models)
                    item_popularity = self.train_data.groupby('item_id').size()
                    
                    # Filter out items user has already rated
                    user_train_items = set(self.train_data[
                        self.train_data['user_id'] == user_id
                    ]['item_id'].values)
                    
                    available_items = item_popularity[~item_popularity.index.isin(user_train_items)]
                    recommendations = available_items.sort_values(ascending=False).head(10).index.tolist()
                    
                    # Calculate metrics
                    recommended_relevant = len(set(recommendations) & relevant_items)
                    
                    precision = recommended_relevant / len(recommendations) if recommendations else 0
                    recall = recommended_relevant / len(relevant_items) if relevant_items else 0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating user {user_id}: {e}")
                    continue
            
            if precision_scores and recall_scores:
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
                
                print(f"Evaluation Results (Popularity-based baseline):")
                print(f"Average Precision@10: {avg_precision:.3f}")
                print(f"Average Recall@10: {avg_recall:.3f}")
                print(f"Average F1@10: {avg_f1:.3f}")
                print(f"Evaluated on {len(precision_scores)} users")
            else:
                print("No valid evaluations could be performed")
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            raise
    
    def run_complete_demo(self) -> None:
        """Run the complete demonstration"""
        try:
            print("="*80)
            print("RECOMMENDER SYSTEMS MACHINE LEARNING DEMO")
            print("E-commerce Product Recommendations")
            print("="*80)
            
            # Generate and explore data
            self.generate_synthetic_data()
            self.explore_data()
            
            # Split data
            self.split_data()
            
            # Demonstrate different approaches
            self.demonstrate_collaborative_filtering()
            self.demonstrate_content_based_filtering()
            self.demonstrate_matrix_factorization()
            
            # Evaluate
            self.evaluate_recommendations()
            
            print("\n" + "="*80)
            print("DEMO COMPLETE!")
            print("="*80)
            print("\nKey Takeaways:")
            print("1. Collaborative filtering leverages user behavior patterns")
            print("2. Content-based filtering uses item features for recommendations")
            print("3. Matrix factorization discovers latent factors in user preferences")
            print("4. Each approach has strengths and limitations")
            print("5. Hybrid systems can combine multiple approaches")
            print("6. Proper evaluation is crucial for production systems")
            
        except Exception as e:
            self.logger.error(f"Error in complete demo: {e}")
            raise

def main():
    """Main function to run the demo"""
    try:
        demo = RecommenderSystemDemo(random_state=42)
        demo.run_complete_demo()
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 