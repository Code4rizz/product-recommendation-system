import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import faiss
import re


class RecommendationEngine:
    def __init__(self, data_path='DMart_cleaned.csv', embeddings_path='product_embeddings.npy'):
        
        # Load

        self.data = pd.read_csv(data_path)
        self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce').fillna(0.0)
        self.embeddings = np.load(embeddings_path).astype('float32')
        
        # Validate

        if len(self.data) != self.embeddings.shape[0]:
            raise ValueError(
                f"Row mismatch: {len(self.data)} products but {self.embeddings.shape[0]} embeddings"
            )
        
        # Normalize

        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("⚠ Embeddings not normalized, normalizing now...")
            faiss.normalize_L2(self.embeddings)
        
        # FAISS

        self.dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(self.embeddings)
        
        # Model

        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        print(f" Loaded {len(self.data)} products with {self.dimension}-dim embeddings")
        print(f" Built FAISS index with {self.faiss_index.ntotal} vectors")
        print(f" Dual search mode: Standard (NumPy) & Deep (FAISS)")
    
    def extract_price_filter(self, query):

        # Patterns

        patterns = [
            (r'under (?:rs\.?|₹)?\s*(\d+)', 'max'),
            (r'below (?:rs\.?|₹)?\s*(\d+)', 'max'),
            (r'less than (?:rs\.?|₹)?\s*(\d+)', 'max'),
            (r'above (?:rs\.?|₹)?\s*(\d+)', 'min'),
            (r'over (?:rs\.?|₹)?\s*(\d+)', 'min'),
            (r'more than (?:rs\.?|₹)?\s*(\d+)', 'min'),
            (r'between (?:rs\.?|₹)?\s*(\d+)\s*(?:and|to|-)\s*(?:rs\.?|₹)?\s*(\d+)', 'range'),
        ]
        
        # Match

        for pattern, filter_type in patterns:
            match = re.search(pattern, query.lower())
            if match:
                if filter_type == 'range':
                    return {'min': int(match.group(1)), 'max': int(match.group(2))}
                elif filter_type == 'max':
                    return {'max': int(match.group(1))}
                else:
                    return {'min': int(match.group(1))}
        return None
    
    def get_user_profile(self, purchase_history):

        if not purchase_history:
            return {
                'total_purchases': 0,
                'categories': {},
                'subcategories': {},
                'avg_price': 0,
                'price_range': (0, 0)
            }
        
        # Extract

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
       
        try:
            # Price

            price_filter = self.extract_price_filter(user_query)
            
            # Clean

            clean_query = re.sub(
                r'(under|below|above|over|less than|more than|between)\s+(?:rs\.?|₹)?\s*\d+(?:\s*(?:and|to|-)\s*(?:rs\.?|₹)?\s*\d+)?',
                '', 
                user_query, 
                flags=re.IGNORECASE
            ).strip()
            
            if not clean_query:
                clean_query = user_query
            
            # Encode

            query_text = f"query: {clean_query.strip()}"
            query_embedding = self.model.encode(
                [query_text], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].astype('float32')
            
            # Search

            if deep_search:

                # FAISS

                query_faiss = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_faiss)
                
                fetch_k = min(top_n * 10, len(self.data))
                scores, indices = self.faiss_index.search(query_faiss, fetch_k)
                
                scores = (scores[0] * 100).round(2)
                indices = indices[0]
                
                recommendations = self.data.iloc[indices].copy()
                recommendations['similarity_score'] = scores
                
            else:

                # NumPy

                similarities = (self.embeddings @ query_embedding) * 100
                
                recommendations = self.data.copy()
                recommendations['similarity_score'] = similarities.round(2)
            
            # Personalization

            if user_purchase_history and user_purchase_history.get('categories'):
                purchased_categories = set(user_purchase_history['categories'])
                purchased_subcategories = set(user_purchase_history.get('subcategories', []))
                
                boost = np.ones(len(recommendations))
                
                # Threshold

                high_similarity = recommendations['similarity_score'] > 60
                
                # Boost

                subcategory_match = recommendations['SubCategory'].isin(purchased_subcategories) & high_similarity
                boost[subcategory_match] = 1.15
                
                category_match = recommendations['Category'].isin(purchased_categories) & ~subcategory_match & high_similarity
                boost[category_match] = 1.10
                
                recommendations['personalized_score'] = (recommendations['similarity_score'] * boost).round(2)
                recommendations['is_personalized'] = subcategory_match | category_match
            else:
                recommendations['personalized_score'] = recommendations['similarity_score']
                recommendations['is_personalized'] = False
            
            # Filter

            if price_filter:
                if 'min' in price_filter:
                    recommendations = recommendations[recommendations['Price'] >= price_filter['min']]
                if 'max' in price_filter:
                    recommendations = recommendations[recommendations['Price'] <= price_filter['max']]
            
            # Sort

            recommendations = recommendations.sort_values('personalized_score', ascending=False)
            recommendations = recommendations[recommendations['similarity_score'] >= min_similarity]
            
            if category_filter and category_filter != "All":
                recommendations = recommendations[recommendations['Category'] == category_filter]
            
            recommendations = recommendations.reset_index(drop=True)
            
            return recommendations.head(top_n), price_filter, None
            
        except Exception as e:
            return pd.DataFrame(), None, str(e)
    
    def get_categories(self):
        
        # Exclude

        exclude_categories = {
            'butterfly', 'geep', 'joyo plastics', 'kitchen aprons', 
            'motorbike helmet', 'pigeon', 'plant container', 'raincoat', 
            'specials', 'syska', 'wonderchef','zebronics'
        }
        
        # Filter
        
        all_categories = self.data['Category'].unique().tolist()
        filtered_categories = [
            cat for cat in all_categories 
            if cat.lower() not in exclude_categories
        ]
        
        return ["All"] + sorted(filtered_categories)