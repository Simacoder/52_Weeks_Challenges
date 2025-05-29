# Advanced Recommender System Analysis
# Additional techniques and deep-dive analysis

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
from scipy.stats import pearsonr
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedRecommenderMetrics:
    """
    Advanced metrics for evaluating recommender systems beyond basic accuracy
    """
    
    def __init__(self, user_item_matrix, item_features=None):
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        
    def calculate_ndcg(self, recommended_items, relevant_items, k=10):
        """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
        def dcg(scores):
            return np.sum([score / np.log2(i + 2) for i, score in enumerate(scores)])
        
        # Get relevance scores for recommended items
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
        
        # Calculate DCG
        dcg_score = dcg(relevance_scores)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevance = sorted([1] * min(len(relevant_items), k), reverse=True)
        idcg_score = dcg(ideal_relevance)
        
        # Calculate NDCG
        if idcg_score == 0:
            return 0
        return dcg_score / idcg_score
    
    def calculate_map(self, all_recommendations, all_relevant_items, k=10):
        """Calculate Mean Average Precision (MAP)"""
        average_precisions = []
        
        for recommendations, relevant_items in zip(all_recommendations, all_relevant_items):
            if not relevant_items:
                continue
                
            precisions = []
            relevant_count = 0
            
            for i, item in enumerate(recommendations[:k]):
                if item in relevant_items:
                    relevant_count += 1
                    precision = relevant_count / (i + 1)
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
        
        return np.mean(average_precisions) if average_precisions else 0
    
    def calculate_diversity(self, recommendations, similarity_matrix):
        """Calculate intra-list diversity of recommendations"""
        if len(recommendations) < 2:
            return 0
        
        total_similarity = 0
        count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                    total_similarity += similarity_matrix[i, j]
                    count += 1
        
        avg_similarity = total_similarity / count if count > 0 else 0
        return 1 - avg_similarity  # Diversity is inverse of similarity
    
    def calculate_coverage(self, all_recommendations, total_items):
        """Calculate catalog coverage"""
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        
        return len(unique_recommended) / total_items
    
    def calculate_novelty(self, recommendations, item_popularity):
        """Calculate novelty based on item popularity"""
        novelty_scores = []
        for item in recommendations:
            if item in item_popularity:
                # Novelty is inverse of popularity
                novelty = -np.log2(item_popularity[item] + 1e-10)
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0

class MatrixFactorizationRecommender:
    """
    Matrix Factorization based recommender using SVD and NMF
    """
    
    def __init__(self, n_components=50, method='svd'):
        self.n_components = n_components
        self.method = method
        self.model = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, user_item_matrix):
        """Fit the matrix factorization model"""
        if self.method == 'svd':
            self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        elif self.method == 'nmf':
            self.model = NMF(n_components=self.n_components, random_state=42, max_iter=200)
        
        # Fit the model
        user_factors = self.model.fit_transform(user_item_matrix)
        
        if self.method == 'svd':
            item_factors = self.model.components_
        else:  # NMF
            item_factors = self.model.components_
        
        self.user_factors = user_factors
        self.item_factors = item_factors
        
        return self
    
    def predict(self, user_idx, item_idx):
        """Predict rating for user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted")
        
        return np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])
    
    def recommend(self, user_idx, n_recommendations=10, exclude_seen=True):
        """Generate recommendations for a user"""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted")
        
        # Calculate scores for all items
        user_vector = self.user_factors[user_idx]
        scores = np.dot(user_vector, self.item_factors)
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1]
        
        return top_items[:n_recommendations]

class GraphBasedRecommender:
    """
    Graph-based recommender using network analysis
    """
    
    def __init__(self):
        self.graph = None
        self.user_nodes = set()
        self.item_nodes = set()
    
    def build_graph(self, interactions_df, user_col, item_col, rating_col):
        """Build bipartite graph from user-item interactions"""
        self.graph = nx.Graph()
        
        for _, row in interactions_df.iterrows():
            user = f"u_{row[user_col]}"
            item = f"i_{row[item_col]}"
            rating = row[rating_col]
            
            self.graph.add_edge(user, item, weight=rating)
            self.user_nodes.add(user)
            self.item_nodes.add(item)
    
    def get_recommendations_via_random_walk(self, user_id, n_recommendations=10, walk_length=4):
        """Get recommendations using random walk with restart"""
        user_node = f"u_{user_id}"
        
        if user_node not in self.graph:
            return []
        
        # Perform random walks
        item_scores = defaultdict(float)
        n_walks = 100
        
        for _ in range(n_walks):
            current_node = user_node
            for _ in range(walk_length):
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Choose next node based on edge weights
                weights = [self.graph[current_node][neighbor]['weight'] for neighbor in neighbors]
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                current_node = np.random.choice(neighbors, p=weights)
                
                # If we land on an item node, increment its score
                if current_node.startswith('i_'):
                    item_scores[current_node] += 1
        
        # Sort items by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item[0].replace('i_', '') for item in sorted_items[:n_recommendations]]
        
        return recommended_items

def comprehensive_evaluation_suite(user_item_matrix, test_interactions, item_popularity=None):
    """
    Comprehensive evaluation of recommender system performance
    """
    print("="*60)
    print("COMPREHENSIVE RECOMMENDER SYSTEM EVALUATION")
    print("="*60)
    
    # Initialize metrics calculator
    metrics_calc = AdvancedRecommenderMetrics(user_item_matrix)
    
    # Example evaluation (would need actual recommendations and ground truth)
    n_users = min(100, user_item_matrix.shape[0])  # Evaluate on subset for efficiency
    
    # Simulate recommendations and ground truth for demonstration
    all_recommendations = []
    all_relevant_items = []
    
    for user_idx in range(n_users):
        # Simulate recommendations (in practice, these come from your model)
        n_items = user_item_matrix.shape[1]
        recommended_items = np.random.choice(n_items, size=10, replace=False)
        
        # Simulate ground truth (items the user actually interacted with)
        user_interactions = user_item_matrix[user_idx].nonzero()[1]
        relevant_items = user_interactions[:5] if len(user_interactions) > 5 else user_interactions
        
        all_recommendations.append(recommended_items.tolist())
        all_relevant_items.append(relevant_items.tolist())
    
    # Calculate metrics
    print("\nRANKING METRICS:")
    print("-" * 30)
    
    # NDCG
    ndcg_scores = []
    for recs, rel in zip(all_recommendations, all_relevant_items):
        ndcg = metrics_calc.calculate_ndcg(recs, rel, k=10)
        ndcg_scores.append(ndcg)
    
    print(f"NDCG@10: {np.mean(ndcg_scores):.4f} ± {np.std(ndcg_scores):.4f}")
    
    # MAP
    map_score = metrics_calc.calculate_map(all_recommendations, all_relevant_items, k=10)
    print(f"MAP@10: {map_score:.4f}")
    
    print("\nBEYOND-ACCURACY METRICS:")
    print("-" * 30)
    
    # Coverage
    total_items = user_item_matrix.shape[1]
    coverage = metrics_calc.calculate_coverage(all_recommendations, total_items)
    print(f"Catalog Coverage: {coverage:.4f}")
    
    # Diversity (using random similarity matrix for demonstration)
    similarity_matrix = np.random.rand(10, 10)
    diversity_scores = []
    for recs in all_recommendations:
        diversity = metrics_calc.calculate_diversity(recs[:10], similarity_matrix)
        diversity_scores.append(diversity)
    
    print(f"Average Diversity: {np.mean(diversity_scores):.4f} ± {np.std(diversity_scores):.4f}")
    
    # Novelty
    if item_popularity is None:
        # Create synthetic popularity scores
        item_popularity = {i: np.random.exponential(0.1) for i in range(total_items)}
    
    novelty_scores = []
    for recs in all_recommendations:
        novelty = metrics_calc.calculate_novelty(recs, item_popularity)
        novelty_scores.append(novelty)
    
    print(f"Average Novelty: {np.mean(novelty_scores):.4f} ± {np.std(novelty_scores):.4f}")
    
    return {
        'ndcg': np.mean(ndcg_scores),
        'map': map_score,
        'coverage': coverage,
        'diversity': np.mean(diversity_scores),
        'novelty': np.mean(novelty_scores)
    }

def compare_recommender_algorithms(user_item_matrix, test_data=None):
    """
    Compare different recommender algorithms
    """
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    
    algorithms = {}
    results = {}
    
    # Matrix Factorization with SVD
    print("\nTraining SVD-based Matrix Factorization...")
    svd_recommender = MatrixFactorizationRecommender(n_components=50, method='svd')
    svd_recommender.fit(user_item_matrix)
    algorithms['SVD'] = svd_recommender
    
    # Matrix Factorization with NMF
    print("Training NMF-based Matrix Factorization...")
    nmf_recommender = MatrixFactorizationRecommender(n_components=50, method='nmf')
    nmf_recommender.fit(user_item_matrix)
    algorithms['NMF'] = nmf_recommender
    
    # Evaluate algorithms
    print("\nEvaluating algorithms...")
    
    for name, algorithm in algorithms.items():
        print(f"\nEvaluating {name}:")
        
        # Generate predictions for a sample of users
        n_test_users = min(50, user_item_matrix.shape[0])
        predictions = []
        actuals = []
        
        for user_idx in range(n_test_users):
            # Get user's actual ratings
            user_ratings = user_item_matrix[user_idx].toarray()[0]
            non_zero_items = np.nonzero(user_ratings)[0]
            
            if len(non_zero_items) > 0:
                # Sample some items for testing
                test_items = np.random.choice(non_zero_items, 
                                            size=min(5, len(non_zero_items)), 
                                            replace=False)
                
                for item_idx in test_items:
                    pred_rating = algorithm.predict(user_idx, item_idx)
                    actual_rating = user_ratings[item_idx]
                    
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)
        
        if predictions:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            results[name] = {'rmse': rmse, 'mae': mae}
    
    return algorithms, results

def analyze_user_behavior_patterns(df):
    """
    Analyze user behavior patterns for better recommendations
    """
    print("\n" + "="*60)
    print("USER BEHAVIOR PATTERN ANALYSIS")
    print("="*60)
    
    # Time-based analysis
    if 'int_date' in df.columns:
        print("\nTemporal Patterns:")
        df['int_date'] = pd.to_datetime(df['int_date'], errors='coerce')
        df['hour'] = df['int_date'].dt.hour
        df['day_of_week'] = df['int_date'].dt.dayofweek
        
        # Peak activity hours
        hourly_activity = df.groupby('hour').size()
        peak_hours = hourly_activity.nlargest(3).index.tolist()
        print(f"Peak activity hours: {peak_hours}")
        
        # Weekly patterns
        weekly_activity = df.groupby('day_of_week').size()
        peak_days = weekly_activity.nlargest(3).index.tolist()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        peak_day_names = [day_names[i] for i in peak_days]
        print(f"Peak activity days: {peak_day_names}")
    
    # Segment analysis
    if 'segment' in df.columns and 'beh_segment' in df.columns:
        print("\nUser Segmentation Analysis:")
        segment_stats = df.groupby(['segment', 'beh_segment']).agg({
            'rating': ['mean', 'count', 'std']
        }).round(3)
        print(segment_stats.head(10))
    
    # Item type preferences
    if 'item_type' in df.columns:
        print("\nItem Type Preferences:")
        item_type_prefs = df.groupby('item_type')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(item_type_prefs.head(10))
    
    # Page interaction analysis
    if 'page' in df.columns:
        print("\nPage Interaction Analysis:")
        page_effectiveness = df.groupby('page')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(page_effectiveness.head(10))

def create_recommendation_explanation_system():
    """
    Create an explainable recommendation system
    """
    print("\n" + "="*60)
    print("EXPLAINABLE RECOMMENDATIONS")
    print("="*60)
    
    class ExplainableRecommender:
        def __init__(self):
            self.feature_importance = {}
            self.user_profiles = {}
            self.item_profiles = {}
        
        def explain_recommendation(self, user_id, item_id, recommendation_score):
            """Generate explanation for why an item was recommended"""
            explanations = []
            
            # Content-based explanations
            explanations.append(f"Recommended because you previously liked similar items")
            explanations.append(f"This item matches your interest in {self._get_user_interests(user_id)}")
            
            # Collaborative filtering explanations
            explanations.append(f"Users with similar preferences also liked this item")
            explanations.append(f"High rating from users in your demographic segment")
            
            # Contextual explanations
            explanations.append(f"Popular choice for your time of day preference")
            explanations.append(f"Trending in your behavioral segment")
            
            return {
                'item_id': item_id,
                'score': recommendation_score,
                'explanations': explanations[:3],  # Top 3 explanations
                'confidence': min(recommendation_score, 1.0)
            }
        
        def _get_user_interests(self, user_id):
            # Placeholder for user interest extraction
            return "technology and lifestyle products"
    
    explainer = ExplainableRecommender()
    
    # Example explanation
    sample_explanation = explainer.explain_recommendation(
        user_id=12345, 
        item_id=67890, 
        recommendation_score=0.85
    )
    
    print("Sample Recommendation Explanation:")
    print(f"Item: {sample_explanation['item_id']}")
    print(f"Confidence: {sample_explanation['confidence']:.2f}")
    print("Reasons:")
    for i, explanation in enumerate(sample_explanation['explanations'], 1):
        print(f"  {i}. {explanation}")
    
    return explainer

def production_monitoring_framework():
    """
    Framework for monitoring recommender systems in production
    """
    print("\n" + "="*60)
    print("PRODUCTION MONITORING FRAMEWORK")
    print("="*60)
    
    monitoring_metrics = {
        "Business Metrics": {
            "Click-Through Rate (CTR)": "Percentage of recommendations clicked",
            "Conversion Rate": "Percentage of clicks that lead to purchases",
            "Revenue per Recommendation": "Average revenue generated per recommendation",
            "User Engagement": "Time spent interacting with recommended items",
            "Customer Lifetime Value": "Long-term value impact of recommendations"
        },
        
        "Technical Metrics": {
            "Response Time": "Average time to generate recommendations",
            "Throughput": "Number of recommendations served per second",
            "Model Accuracy": "Ongoing validation of prediction accuracy",
            "Coverage": "Percentage of catalog being recommended",
            "Diversity": "Variety in recommendation types"
        },
        
        "Data Quality Metrics": {
            "Data Freshness": "Age of training data",
            "Feature Drift": "Changes in input feature distributions",
            "Prediction Drift": "Changes in model output patterns",
            "Bias Detection": "Monitoring for unfair recommendations"
        },
        
        "User Experience Metrics": {
            "Recommendation Relevance": "User feedback on recommendation quality",
            "Novelty Score": "How surprising recommendations are to users",
            "Serendipity": "Discovery of unexpected but appreciated items",
            "User Satisfaction": "Overall satisfaction with recommendations"
        }
    }
    
    for category, metrics in monitoring_metrics.items():
        print(f"\n{category}:")
        for metric, description in metrics.items():
            print(f"  • {metric}: {description}")
    
    # Alert thresholds example
    print(f"\nRecommended Alert Thresholds:")
    print(f"  • CTR drops below 2%")
    print(f"  • Response time exceeds 200ms")
    print(f"  • Coverage drops below 70%")
    print(f"  • User satisfaction score below 3.5/5")
    print(f"  • Model accuracy drops by >5% from baseline")

def advanced_techniques_summary():
    """
    Summary of advanced recommender system techniques
    """
    print("\n" + "="*80)
    print("ADVANCED RECOMMENDER SYSTEM TECHNIQUES")
    print("="*80)
    
    techniques = {
        "Deep Learning Approaches": [
            "Neural Collaborative Filtering (NCF)",
            "Autoencoders for Collaborative Filtering", 
            "Recurrent Neural Networks for Sequential Recommendations",
            "Convolutional Neural Networks for Content-Based Filtering",
            "Graph Neural Networks for Social Recommendations"
        ],
        
        "Multi-Armed Bandit Algorithms": [
            "Epsilon-Greedy for Exploration vs Exploitation",
            "Upper Confidence Bound (UCB) algorithms",
            "Thompson Sampling for Bayesian optimization",
            "LinUCB for contextual recommendations"
        ],
        
        "Reinforcement Learning": [
            "Q-Learning for long-term user satisfaction",
            "Actor-Critic methods for recommendation policies",
            "Multi-Agent RL for marketplace recommendations",
            "Reward modeling for user engagement optimization"
        ],
        
        "Advanced Matrix Factorization": [
            "Bayesian Personalized Ranking (BPR)",
            "Factorization Machines for feature interactions",
            "Field-aware Factorization Machines",
            "Neural Matrix Factorization"
        ],
        
        "Context-Aware Recommendations": [
            "Tensor Factorization for multi-dimensional data",
            "Time-aware collaborative filtering",
            "Location-based recommendations",
            "Social network integration"
        ],
        
        "Ensemble Methods": [
            "Weighted combination of multiple algorithms",
            "Stacking with meta-learners",
            "Dynamic algorithm selection",
            "Multi-objective optimization"
        ]
    }
    
    for category, methods in techniques.items():
        print(f"\n{category}:")
        for method in methods:
            print(f"  • {method}")

# Main execution for advanced analysis
def run_advanced_analysis():
    """
    Run the complete advanced analysis pipeline
    """
    print("ADVANCED RECOMMENDER SYSTEM ANALYSIS")
    print("=" * 80)
    
    # Create synthetic data for demonstration
    n_users, n_items = 1000, 500
    density = 0.1  # 10% of user-item pairs have interactions
    
    # Generate synthetic user-item matrix
    user_item_matrix = sp.random(n_users, n_items, density=density, format='csr')
    user_item_matrix.data = np.random.choice([1, 2, 3, 4, 5], size=len(user_item_matrix.data))
    
    print(f"Synthetic dataset: {n_users} users, {n_items} items, {density*100}% density")
    
    # Run comprehensive evaluation
    eval_results = comprehensive_evaluation_suite(user_item_matrix)
    
    # Compare algorithms
    algorithms, comparison_results = compare_recommender_algorithms(user_item_matrix)
    
    # Create explainable system
    explainer = create_recommendation_explanation_system()
    
    # Show production monitoring framework
    production_monitoring_framework()
    
    # Show advanced techniques
    advanced_techniques_summary()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return eval_results, comparison_results, algorithms, explainer

if __name__ == "__main__":
    results = run_advanced_analysis()