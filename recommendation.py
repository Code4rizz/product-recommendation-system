import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import faiss

class RecommendationEngine:
    def __init__(self, data_path='DMart_cleaned.csv', embeddings_path='product_embeddings.npy'):
        """Initialize the recommendation engine with dual search modes"""
        # Load data
        self.data = pd.read_csv(data_path)
        
        # Ensure Price is numeric
        self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce').fillna(0.0)
        
        # Load and validate embeddings
        self.embeddings = np.load(embeddings_path).astype('float32')
        
        # Validate row/embedding match
        if len(self.data) != self.embeddings.shape[0]:
            raise ValueError(
                f"Row mismatch: {len(self.data)} products but {self.embeddings.shape[0]} embeddings"
            )
        
        # Verify normalization
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("⚠ Embeddings not normalized, normalizing now...")
            faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index for deep search
        self.dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(self.embeddings)
        
        # Load model
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        print(f"✓ Loaded {len(self.data)} products with {self.dimension}-dim embeddings")
        print(f"✓ Built FAISS index with {self.faiss_index.ntotal} vectors")
        print(f"✓ Dual search mode: Standard (NumPy) & Deep (FAISS)")
    
    def get_user_profile(self, purchase_history):
        """Generate user profile from purchase history"""
        if not purchase_history:
            return {
                'total_purchases': 0,
                'categories': {},
                'subcategories': {},
                'avg_price': 0,
                'price_range': (0, 0)
            }
        
        categories = [p['category'] for p in purchase_history]
        subcategories = [p['subcategory'] for p in purchase_history]
        prices = [p['price'] for p in purchase_history]
        
        return {
            'total_purchases': len(purchase_history),
            'categories': dict(Counter(categories)),
            'subcategories': dict(Counter(subcategories)),
            'avg_price': np.mean(prices) if prices else 0,
            'price_range': (min(prices), max(prices)) if prices else (0, 0)
        }
    
    def get_recommendations(self, user_query, top_n=5, category_filter=None, 
                          min_similarity=0, user_purchase_history=None, deep_search=False):
        """
        Get product recommendations with dual search modes
        
        Args:
            deep_search: If True, uses FAISS for faster approximate search.
                        If False, uses NumPy for exhaustive exact search.
        """
        try:
            # Encode query with BGE prefix
            query_text = f"query: {user_query.strip()}"
            query_embedding = self.model.encode(
                [query_text], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].astype('float32')
            
            # ========================================
            # DUAL SEARCH MODE
            # ========================================
            
            if deep_search:
                # FAISS MODE: Fast approximate search
                query_faiss = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_faiss)
                
                # Over-fetch to account for filtering
                fetch_k = min(top_n * 10, len(self.data))
                scores, indices = self.faiss_index.search(query_faiss, fetch_k)
                
                scores = (scores[0] * 100).round(2)
                indices = indices[0]
                
                # Build recommendations from FAISS results
                recommendations = self.data.iloc[indices].copy()
                recommendations['similarity_score'] = scores
                
            else:
                # NUMPY MODE: Exhaustive exact search (all products)
                similarities = (self.embeddings @ query_embedding) * 100
                
                recommendations = self.data.copy()
                recommendations['similarity_score'] = similarities.round(2)
            
            # ========================================
            # PERSONALIZATION (same for both modes)
            # ========================================
            
            if user_purchase_history and user_purchase_history.get('categories'):
                purchased_categories = set(user_purchase_history['categories'])
                purchased_subcategories = set(user_purchase_history.get('subcategories', []))
                user_avg_price = user_purchase_history.get('avg_price', 0)
                
                # Vectorized boost calculation
                boost = np.ones(len(recommendations))
                
                # Subcategory match = 15% boost
                subcategory_match = recommendations['SubCategory'].isin(purchased_subcategories)
                boost[subcategory_match] = 1.15
                
                # Category match = 10% boost
                category_match = recommendations['Category'].isin(purchased_categories) & ~subcategory_match
                boost[category_match] = 1.10
                
                # Price-aware: reduce boost for expensive items
                if user_avg_price > 0:
                    price_ratio = recommendations['Price'] / user_avg_price
                    boost[price_ratio > 3.0] = 1.0
                    expensive_mask = (price_ratio > 2.0) & (price_ratio <= 3.0)
                    boost[expensive_mask] *= 0.7
                
                recommendations['personalized_score'] = (recommendations['similarity_score'] * boost).round(2)
                recommendations['is_personalized'] = subcategory_match | category_match
            else:
                recommendations['personalized_score'] = recommendations['similarity_score']
                recommendations['is_personalized'] = False
            
            # Sort and filter
            recommendations = recommendations.sort_values('personalized_score', ascending=False)
            recommendations = recommendations[recommendations['similarity_score'] >= min_similarity]
            
            if category_filter and category_filter != "All":
                recommendations = recommendations[recommendations['Category'] == category_filter]
            
            recommendations = recommendations.reset_index(drop=True)
            
            return recommendations.head(top_n), None, None
            
        except Exception as e:
            return pd.DataFrame(), None, str(e)
    
    def get_categories(self):
        """Get list of unique categories with 'All' option"""
        categories = sorted(self.data['Category'].unique().tolist())
        return ["All"] + categories