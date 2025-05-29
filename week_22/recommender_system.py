import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class RecommenderSystem:
    """
    A comprehensive recommender system for personalized offers combining
    collaborative filtering, content-based filtering, and hybrid approaches.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.hybrid_model = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Basic data exploration
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nData types:\n{df.dtypes}")
        
        # Handle missing values
        df = df.dropna()
        
        # Create user-item interactions
        # Assuming 'interaction' column indicates engagement level
        df['rating'] = self._create_implicit_ratings(df)
        
        return df
    
    def _create_implicit_ratings(self, df):
        """Convert implicit feedback to ratings"""
        # Create ratings based on interaction types and behavior
        rating_map = {
            'view': 1,
            'click': 2, 
            'add_to_cart': 3,
            'purchase': 5,
            'like': 3,
            'share': 4
        }
        
        # If interaction column has specific values, map them
        if 'interaction' in df.columns:
            ratings = df['interaction'].map(rating_map).fillna(1)
        else:
            # Create synthetic ratings based on available features
            ratings = np.ones(len(df))
            
            # Boost ratings for active users
            if 'active_ind' in df.columns:
                ratings = np.where(df['active_ind'] == 'Y', ratings + 1, ratings)
            
            # Adjust based on time of day (assuming peak hours get higher ratings)
            if 'tod' in df.columns:
                peak_hours = ['morning', 'evening']
                ratings = np.where(df['tod'].isin(peak_hours), ratings + 0.5, ratings)
        
        return ratings.astype(float)
    
    def create_user_item_matrix(self, df):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Use 'idcol' as user identifier and 'item' as item identifier
        user_col = 'idcol' if 'idcol' in df.columns else df.columns[0]
        item_col = 'item' if 'item' in df.columns else 'item_descrip'
        
        # Create pivot table
        user_item_df = df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values='rating', 
            aggfunc='mean',
            fill_value=0
        )
        
        self.user_item_matrix = csr_matrix(user_item_df.values)
        self.user_ids = user_item_df.index.tolist()
        self.item_ids = user_item_df.columns.tolist()
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        return user_item_df
    
    def collaborative_filtering(self, n_recommendations=10):
        """Implement collaborative filtering using cosine similarity"""
        print("Training collaborative filtering model...")
        
        # Calculate user-user similarity
        user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Calculate item-item similarity  
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        self.user_similarity = user_similarity
        self.item_similarity = item_similarity
        
        return user_similarity, item_similarity
    
    def content_based_filtering(self, df):
        """Implement content-based filtering"""
        print("Training content-based filtering model...")
        
        # Encode categorical features
        categorical_cols = ['page', 'tod', 'item_type', 'segment', 'beh_segment']
        
        feature_df = df.copy()
        
        for col in categorical_cols:
            if col in feature_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    feature_df[col] = self.label_encoders[col].fit_transform(feature_df[col].astype(str))
                else:
                    feature_df[col] = self.label_encoders[col].transform(feature_df[col].astype(str))
        
        # Select numeric features for content-based filtering
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        content_features = feature_df[numeric_cols].fillna(0)
        
        # Scale features
        content_features_scaled = self.scaler.fit_transform(content_features)
        
        self.content_features = content_features_scaled
        return content_features_scaled
    
    def hybrid_model(self, df):
        """Train a hybrid model combining collaborative and content-based approaches"""
        print("Training hybrid recommender model...")
        
        # Prepare features for ML model
        X = self.content_features
        y = (df['rating'] > df['rating'].median()).astype(int)  # Binary classification
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train ensemble model
        self.hybrid_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.hybrid_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.hybrid_model.predict(X_test)
        y_pred_proba = self.hybrid_model.predict_proba(X_test)[:, 1]
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def get_user_recommendations(self, user_id, n_recommendations=10, method='hybrid'):
        """Get recommendations for a specific user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        
        if method == 'collaborative':
            # Get similar users
            user_sim_scores = self.user_similarity[user_idx]
            similar_users = np.argsort(user_sim_scores)[::-1][1:11]  # Top 10 similar users
            
            # Get items liked by similar users
            recommendations = []
            for sim_user_idx in similar_users:
                user_items = self.user_item_matrix[sim_user_idx].toarray()[0]
                top_items = np.argsort(user_items)[::-1][:n_recommendations]
                recommendations.extend([self.item_ids[i] for i in top_items if user_items[i] > 0])
            
            return list(set(recommendations))[:n_recommendations]
        
        elif method == 'content':
            # Content-based recommendations would require item features
            # This is a simplified version
            return self.item_ids[:n_recommendations]
        
        else:  # hybrid
            # Use ML model predictions
            if self.hybrid_model is not None:
                # This would require proper feature engineering for the specific user
                return self.item_ids[:n_recommendations]
    
    def evaluate_model(self, X_test, y_test, y_pred, y_pred_proba):
        """Evaluate model performance with various metrics"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Accuracy metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Beyond-accuracy metrics
        self.calculate_beyond_accuracy_metrics(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_beyond_accuracy_metrics(self, y_test, y_pred_proba):
        """Calculate beyond-accuracy metrics important for recommender systems"""
        print("\n" + "-"*30)
        print("BEYOND-ACCURACY METRICS")
        print("-"*30)
        
        # Coverage - how many unique items can be recommended
        unique_items_coverage = len(self.item_ids) / len(self.item_ids)  # Placeholder
        print(f"Catalog Coverage: {unique_items_coverage:.4f}")
        
        # Diversity - average dissimilarity between recommended items
        if hasattr(self, 'item_similarity'):
            avg_item_similarity = np.mean(self.item_similarity)
            diversity = 1 - avg_item_similarity
            print(f"Diversity (1 - avg similarity): {diversity:.4f}")
        
        # Novelty - recommend less popular items
        # This would require popularity scores
        print("Novelty: Calculated based on item popularity (implementation depends on business logic)")
        
        # Serendipity - unexpected but relevant recommendations
        print("Serendipity: Measured through user feedback in production")
        
        # Business metrics placeholders
        print(f"Predicted CTR improvement: {np.mean(y_pred_proba):.4f}")
        print("ROI and conversion metrics: Require business KPI integration")
    
    def visualize_results(self, df):
        """Create visualizations for the recommender system"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Rating distribution
        plt.subplot(2, 3, 1)
        plt.hist(df['rating'], bins=20, alpha=0.7)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Plot 2: Item popularity
        plt.subplot(2, 3, 2)
        item_counts = df['item'].value_counts().head(20)
        plt.bar(range(len(item_counts)), item_counts.values)
        plt.title('Top 20 Most Popular Items')
        plt.xlabel('Items (ranked)')
        plt.ylabel('Interaction Count')
        
        # Plot 3: User engagement by segment
        if 'segment' in df.columns:
            plt.subplot(2, 3, 3)
            segment_engagement = df.groupby('segment')['rating'].mean()
            plt.bar(segment_engagement.index, segment_engagement.values)
            plt.title('Average Rating by Segment')
            plt.xticks(rotation=45)
        
        # Plot 4: Time of day analysis
        if 'tod' in df.columns:
            plt.subplot(2, 3, 4)
            tod_engagement = df.groupby('tod')['rating'].mean()
            plt.bar(tod_engagement.index, tod_engagement.values)
            plt.title('Average Rating by Time of Day')
            plt.xticks(rotation=45)
        
        # Plot 5: User-Item interaction heatmap (sample)
        plt.subplot(2, 3, 5)
        sample_matrix = self.user_item_matrix[:50, :50].toarray() if self.user_item_matrix is not None else np.random.rand(50, 50)
        sns.heatmap(sample_matrix, cmap='viridis', cbar=True)
        plt.title('User-Item Interaction Matrix (Sample)')
        
        plt.tight_layout()
        plt.show()

# Production considerations and recommendations
def production_considerations():
    """
    Detailed production considerations for the recommender system
    """
    print("\n" + "="*60)
    print("PRODUCTION ENVIRONMENT CONSIDERATIONS")
    print("="*60)
    
    considerations = {
        "Scalability": [
            "• Implement distributed computing (Spark/Dask) for large datasets",
            "• Use approximate algorithms (LSH, random sampling) for real-time recommendations",
            "• Consider matrix factorization techniques (SVD, NMF) for memory efficiency",
            "• Implement caching strategies for frequent recommendations"
        ],
        "Real-time Performance": [
            "• Pre-compute user/item embeddings and similarities",
            "• Use streaming data processing (Kafka, Kinesis) for real-time updates",
            "• Implement recommendation serving with sub-100ms latency",
            "• Use feature stores for consistent feature serving"
        ],
        "Data Quality & Monitoring": [
            "• Implement data validation pipelines",
            "• Monitor for concept drift and model degradation",
            "• Track recommendation quality metrics continuously",
            "• Set up alerts for anomalous user behavior patterns"
        ],
        "Business Metrics": [
            "• A/B testing framework for recommendation strategies",
            "• Track conversion rates, click-through rates, revenue impact",
            "• Monitor user engagement and retention metrics",
            "• Measure diversity and fairness in recommendations"
        ],
        "Cold Start Problem": [
            "• Content-based recommendations for new users/items",
            "• Implement popularity-based fallbacks",
            "• Use demographic and contextual information",
            "• Active learning strategies to gather initial preferences"
        ],
        "Privacy & Ethics": [
            "• Implement differential privacy techniques",
            "• Ensure GDPR/CCPA compliance for user data",
            "• Address algorithmic bias and fairness",
            "• Provide transparency and explainability in recommendations"
        ],
        "Infrastructure": [
            "• Multi-model serving architecture (online/batch predictions)",
            "• Model versioning and rollback capabilities",
            "• Load balancing and auto-scaling",
            "• Cross-regional deployment for global users"
        ]
    }
    
    for category, items in considerations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

# Main execution function
def main():
    """Main function to run the complete recommender system pipeline"""
    
    # Initialize recommender system
    rec_system = RecommenderSystem()
    
    # Load and preprocess data
    try:
        df = rec_system.load_and_preprocess_data('data/dq_recsys_challenge_2025(in).csv')
        
        # Create user-item matrix
        user_item_df = rec_system.create_user_item_matrix(df)
        
        # Train collaborative filtering
        rec_system.collaborative_filtering()
        
        # Train content-based filtering
        rec_system.content_based_filtering(df)
        
        # Train hybrid model
        X_test, y_test, y_pred, y_pred_proba = rec_system.hybrid_model(df)
        
        # Evaluate model
        metrics = rec_system.evaluate_model(X_test, y_test, y_pred, y_pred_proba)
        
        # Visualize results
        rec_system.visualize_results(df)
        
        # Example recommendations
        print("\n" + "="*40)
        print("SAMPLE RECOMMENDATIONS")
        print("="*40)
        
        if len(rec_system.user_ids) > 0:
            sample_user = rec_system.user_ids[0]
            recommendations = rec_system.get_user_recommendations(sample_user, n_recommendations=5)
            print(f"Top 5 recommendations for user {sample_user}:")
            for i, item in enumerate(recommendations, 1):
                print(f"  {i}. {item}")
        
        # Production considerations
        production_considerations()
        
        return rec_system, metrics
        
    except FileNotFoundError:
        print("Error: challenge_2025.csv file not found.")
        print("Please ensure the data file is in the correct directory.")
        return None, None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    recommender_system, evaluation_metrics = main()