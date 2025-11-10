# 🛒 Smart Product Recommender

AI-powered product recommendation system with semantic search and price filtering.

## Features
- 🔍 Semantic product search with FAISS
- 💰 Natural language price filtering (e.g., "chips under 200")
- 🌟 Personalized recommendations based on purchase history
- 🚀 Dual search modes: Exact & Deep Search
- 🛒 Shopping cart functionality

## Tech Stack
- **Frontend**: Streamlit
- **ML**: Sentence Transformers (BGE), FAISS
- **Data**: Pandas, NumPy

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have these files:
   - `DMart_cleaned.csv` (product data)
   - `product_embeddings.npy` (pre-computed embeddings)

3. Run the app:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## Project Structure
```
├── app.py                      # Main Streamlit UI
├── recommendation.py           # Recommendation engine
├── DMart_cleaned.csv          # Product dataset
├── product_embeddings.npy     # Pre-computed embeddings
├── requirements.txt           # Python dependencies
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## Usage Examples
- "chips under 200"
- "rice below 500"
- "snacks between 50 and 150"
- "oil above 100"

## License
MIT