import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
data = pd.read_csv('DMart_cleaned.csv')
print(f"Loaded {data.shape[0]} products")

# Create combined text for embeddings with BGE prefix
data['combined_text'] = (
    'passage: ' +  # BGE best practice
    data['Name'] + ' ' + 
    data['Category'] + ' ' + 
    data['SubCategory'] + ' ' + 
    data['Description']
).str.strip()

print("\nLoading BGE-Base model (this may take a moment on first run)...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
print(f"✓ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

print("\nGenerating embeddings for all products...")
print("This will take a few minutes depending on your hardware...")

# Generate embeddings with built-in normalization
embeddings = model.encode(
    data['combined_text'].tolist(), 
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # Built-in L2 normalization
    batch_size=32
)

print(f"\n✓ Generated embeddings with shape: {embeddings.shape}")

# Convert to float32 for efficiency
embeddings = embeddings.astype(np.float32)
print(f"✓ Converted to float32")

# Save embeddings
np.save('product_embeddings.npy', embeddings)
print(f"\n✓ Embeddings saved to 'product_embeddings.npy'")
print(f"  File size: {embeddings.nbytes / (1024*1024):.2f} MB")

# Display statistics
print("\n" + "="*50)
print("EMBEDDING STATISTICS")
print("="*50)
print(f"Total products: {len(data)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Model: BAAI/bge-base-en-v1.5")
print(f"Total parameters: ~109M")
print(f"Normalization: L2 (unit vectors)")
print("="*50)

print("\n✅ Done! You can now use these embeddings with your recommendation system.")
print("⚠️  Make sure to use 'query: ' prefix in recommendation.py queries!")