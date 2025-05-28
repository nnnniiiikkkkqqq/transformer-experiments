import numpy as np
import plotly.graph_objects as go
import gensim.downloader as api
from sklearn.decomposition import PCA

# Load pre-trained Word2Vec model
print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")  # Cached from previous run

# Words for the semantic shift
words = ["big", "large", "small", "tiny"]

# Get embeddings
embeddings = np.array([model[word] for word in words])

# Perform vector operation: big - large + small ≈ small or tiny
shift_vector = embeddings[0] - embeddings[1] + embeddings[2]  # big - large + small
# Verify closest word
closest_word = model.most_similar(positive=["big", "small"], negative=["large"], topn=1)
print(f"Closest word to shift: {closest_word}")

# Dimensionality reduction to 3D using PCA
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)
shift_vector_3d = pca.transform([shift_vector])[0]

# Create 3D scatter plot
fig = go.Figure()

# Plot original word embeddings
fig.add_trace(go.Scatter3d(
    x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
    mode='markers+text',
    text=words,
    marker=dict(size=8, color='blue'),
    textposition='top center'
))

# Plot shift result
fig.add_trace(go.Scatter3d(
    x=[shift_vector_3d[0]], y=[shift_vector_3d[1]], z=[shift_vector_3d[2]],
    mode='markers+text',
    text=['Shift Result'],
    marker=dict(size=8, color='red'),
    textposition='top center'
))

# Add lines to show relationships
# big -> large (synonym)
fig.add_trace(go.Scatter3d(
    x=[embeddings_3d[0, 0], embeddings_3d[1, 0]],
    y=[embeddings_3d[0, 1], embeddings_3d[1, 1]],
    z=[embeddings_3d[0, 2], embeddings_3d[1, 2]],
    mode='lines',
    line=dict(color='green', width=2),
    name='big -> large (synonym)'
))

# big -> small (antonym)
fig.add_trace(go.Scatter3d(
    x=[embeddings_3d[0, 0], embeddings_3d[2, 0]],
    y=[embeddings_3d[0, 1], embeddings_3d[2, 1]],
    z=[embeddings_3d[0, 2], embeddings_3d[2, 2]],
    mode='lines',
    line=dict(color='purple', width=2),
    name='big -> small (antonym)'
))

# Update layout
fig.update_layout(
    title="3D Visualization of Word Embeddings: Semantic Shift from 'big' to 'small' or 'tiny' via Synonym-Antonym Transformation",
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    showlegend=True
)

# Show plot
fig.show()
