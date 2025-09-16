import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

from datetime import datetime
from brand_similarity_analysis import run_brand_similarity
from review_similarity_analysis import run_review_similarity

st.set_page_config(page_title="Amazon Brand & Review Analyzer", layout="wide")
st.title("Amazon Brand & Review Analyzer")

st.markdown("""
<style>
/* center headers and cells for all Streamlit dataframes */
thead tr th, tbody tr td {
    text-align: center !important;
    vertical-align: middle !important;
}
/* center Plotly/Matplotlib titles (just in case) */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Controls")
search_url = st.sidebar.text_input("Amazon search URL", value="https://www.amazon.in/s?k=trimmer")
num_products = st.sidebar.slider("Number of Products", 5, 30, 15)
run_brand = st.sidebar.button("Run Brand Similarity")
run_review = st.sidebar.button("Run Review Similarity")

OUT_DIR = "outputs"
REV_OUT = "outputs_reviews"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REV_OUT, exist_ok=True)

if run_brand:
    st.info("Running brand similarity...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUT_DIR, ts)
    os.makedirs(outdir, exist_ok=True)
    try:
        run_brand_similarity(st, search_url, num_products=num_products, out_dir=outdir)
        st.success(f"Results saved in {outdir}")
    except Exception as e:
        st.error(f"Error: {e}")

if run_review:
    product_csv = st.sidebar.text_input("Path to product_data.xlsx", value="")
    if not product_csv:
        # auto-detect latest
        found = None
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if f == "product_data.xlsx":
                    found = os.path.join(root, f)
        product_csv = found
    if not product_csv or not os.path.exists(product_csv):
        st.error("No product_data.xlsx found. Run Brand Similarity first.")
    else:
        st.info("Running review similarity...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(REV_OUT, ts)
        os.makedirs(outdir, exist_ok=True)
        try:
            run_review_similarity(product_csv, out_dir=outdir)
            st.success(f"Review results saved in {outdir}")
        except Exception as e:
            st.error(f"Error: {e}")
            
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict, Counter
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from brand_similarity_analysis import run_brand_similarity
# import warnings
# warnings.filterwarnings('ignore')

# # Configure page
# st.set_page_config(
#     page_title="🚀 Brand Analytics Dashboard", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#         margin: 0.5rem 0;
#     }
#     .insight-box {
#         background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 5px solid #ff6b6b;
#         margin: 1rem 0;
#     }
#     .stSelectbox > div > div > select {
#         background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Enhanced header
# st.markdown("""
# <div class="main-header">
#     <h1>🚀 Advanced Brand Analytics Dashboard</h1>
#     <p>Deep Brand Similarity Analysis with Multi-Modal Intelligence</p>
# </div>
# """, unsafe_allow_html=True)

# class EnhancedBrandAnalyzer:
#     def __init__(self):
#         self.colors = {
#             'primary': '#667eea',
#             'secondary': '#764ba2', 
#             'accent': '#ff6b6b',
#             'success': '#51cf66',
#             'warning': '#ffd43b',
#             'info': '#339af0'
#         }
        
#     def create_enhanced_similarity_matrix(self, similarity_matrix, brand_names, title, color_scale="viridis"):
#         """Create an interactive similarity heatmap with enhanced features"""
#         fig = go.Figure(data=go.Heatmap(
#             z=similarity_matrix,
#             x=brand_names,
#             y=brand_names,
#             colorscale=color_scale,
#             text=np.round(similarity_matrix, 3),
#             texttemplate="%{text}",
#             textfont={"size": 10},
#             hoverongaps=False,
#             hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title={
#                 'text': f"<b>{title}</b>",
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'font': {'size': 18, 'color': self.colors['primary']}
#             },
#             xaxis_title="Brands",
#             yaxis_title="Brands",
#             width=600,
#             height=500,
#             template="plotly_white"
#         )
        
#         return fig

#     def create_similarity_network(self, similarity_matrix, brand_names, threshold=0.7):
#         """Create a network graph showing brand relationships"""
#         # Create edges based on similarity threshold
#         edges = []
#         edge_weights = []
        
#         for i in range(len(brand_names)):
#             for j in range(i+1, len(brand_names)):
#                 if similarity_matrix[i][j] > threshold:
#                     edges.append((i, j))
#                     edge_weights.append(similarity_matrix[i][j])
        
#         # Create network layout (simple circular for now)
#         n = len(brand_names)
#         angles = np.linspace(0, 2*np.pi, n, endpoint=False)
#         x_pos = np.cos(angles)
#         y_pos = np.sin(angles)
        
#         # Create traces for edges
#         edge_trace = []
#         for i, (start, end) in enumerate(edges):
#             edge_trace.append(go.Scatter(
#                 x=[x_pos[start], x_pos[end], None],
#                 y=[y_pos[start], y_pos[end], None],
#                 mode='lines',
#                 line=dict(width=edge_weights[i]*5, color=f'rgba(100,100,100,{edge_weights[i]})'),
#                 hoverinfo='none',
#                 showlegend=False
#             ))
        
#         # Create trace for nodes
#         node_trace = go.Scatter(
#             x=x_pos,
#             y=y_pos,
#             mode='markers+text',
#             marker=dict(size=30, color=self.colors['primary'], line=dict(width=2, color='white')),
#             text=brand_names,
#             textposition="middle center",
#             textfont=dict(size=12, color='white'),
#             hovertemplate='<b>%{text}</b><extra></extra>',
#             name="Brands"
#         )
        
#         fig = go.Figure(data=edge_trace + [node_trace])
#         fig.update_layout(
#             title=f"<b>Brand Similarity Network (Threshold: {threshold})</b>",
#             showlegend=False,
#             hovermode='closest',
#             margin=dict(b=20,l=5,r=5,t=40),
#             annotations=[
#                 dict(text="Node size represents brand importance<br>Edge thickness shows similarity strength",
#                      showarrow=False, xref="paper", yref="paper",
#                      x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
#                      font=dict(color='gray', size=10))
#             ],
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             template="plotly_white"
#         )
        
#         return fig

#     def create_brand_positioning_map(self, text_embeddings, brand_names):
#         """Create 2D positioning map using t-SNE"""
#         # Use t-SNE for dimensionality reduction
#         tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(brand_names)-1))
#         coords_2d = tsne.fit_transform(text_embeddings)
        
#         fig = go.Figure()
        
#         # Add scatter points for brands
#         fig.add_trace(go.Scatter(
#             x=coords_2d[:, 0],
#             y=coords_2d[:, 1],
#             mode='markers+text',
#             marker=dict(
#                 size=15,
#                 color=np.arange(len(brand_names)),
#                 colorscale='viridis',
#                 opacity=0.8,
#                 line=dict(width=2, color='white')
#             ),
#             text=brand_names,
#             textposition="top center",
#             textfont=dict(size=12),
#             hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
#             name="Brands"
#         ))
        
#         fig.update_layout(
#             title="<b>Brand Positioning Map (t-SNE)</b>",
#             xaxis_title="Dimension 1",
#             yaxis_title="Dimension 2",
#             template="plotly_white",
#             width=800,
#             height=600
#         )
        
#         return fig

#     def create_tone_radar_chart(self, tone_data):
#         """Create radar chart for tone analysis"""
#         categories = [col for col in tone_data.columns if col not in ['Sl No', 'Brand']]
        
#         fig = go.Figure()
        
#         colors = px.colors.qualitative.Set3
        
#         for i, row in tone_data.iterrows():
#             brand = row['Brand']
#             values = [row[cat] for cat in categories]
            
#             fig.add_trace(go.Scatterpolar(
#                 r=values,
#                 theta=categories,
#                 fill='toself',
#                 name=brand,
#                 line_color=colors[i % len(colors)],
#                 opacity=0.7
#             ))
        
#         fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[0, max(tone_data[categories].max())]
#                 )),
#             showlegend=True,
#             title="<b>Brand Tone Profile Comparison</b>",
#             template="plotly_white",
#             width=700,
#             height=600
#         )
        
#         return fig

#     def create_color_analysis_chart(self, color_data):
#         """Enhanced color analysis with treemap and distribution"""
#         # Create color frequency analysis
#         all_colors = []
#         brand_color_counts = []
        
#         for _, row in color_data.iterrows():
#             brand = row['Brand']
#             colors_str = row.get('Top_10_Colors', '')
#             if colors_str:
#                 colors_list = [c.strip() for c in colors_str.split(',') if c.strip()]
#                 all_colors.extend(colors_list)
#                 brand_color_counts.append({
#                     'Brand': brand,
#                     'Color_Count': len(colors_list),
#                     'Dominant': row.get('Dominant_Color', 'Unknown')
#                 })
        
#         # Color frequency chart
#         color_counts = Counter(all_colors)
#         top_colors = color_counts.most_common(10)
        
#         fig1 = go.Figure(data=[
#             go.Bar(
#                 x=[color for color, _ in top_colors],
#                 y=[count for _, count in top_colors],
#                 marker_color=self.colors['accent'],
#                 text=[count for _, count in top_colors],
#                 textposition='auto',
#             )
#         ])
        
#         fig1.update_layout(
#             title="<b>Most Popular Colors Across All Brands</b>",
#             xaxis_title="Colors",
#             yaxis_title="Frequency",
#             template="plotly_white"
#         )
        
#         # Brand color diversity
#         diversity_df = pd.DataFrame(brand_color_counts)
#         fig2 = px.bar(
#             diversity_df, 
#             x='Brand', 
#             y='Color_Count',
#             color='Color_Count',
#             color_continuous_scale='viridis',
#             title="<b>Color Diversity by Brand</b>"
#         )
        
#         return fig1, fig2

#     def create_comprehensive_metrics_dashboard(self, tone_df, visual_df, logo_df):
#         """Create a comprehensive metrics overview"""
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             avg_emotional = tone_df['emotional'].mean()
#             st.markdown(f"""
#             <div class="metric-card">
#                 <h3>😊 Emotional Tone</h3>
#                 <h2>{avg_emotional:.1f}%</h2>
#                 <p>Average across brands</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             avg_premium = tone_df['premium'].mean()
#             st.markdown(f"""
#             <div class="metric-card">
#                 <h3>💎 Premium Score</h3>
#                 <h2>{avg_premium:.1f}%</h2>
#                 <p>Luxury positioning</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col3:
#             avg_logo_presence = logo_df['%With_Logo'].mean()
#             st.markdown(f"""
#             <div class="metric-card">
#                 <h3>🏷️ Logo Presence</h3>
#                 <h2>{avg_logo_presence:.1f}%</h2>
#                 <p>Average logo visibility</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col4:
#             innovation_leader = tone_df.loc[tone_df['innovation'].idxmax(), 'Brand']
#             st.markdown(f"""
#             <div class="metric-card">
#                 <h3>🚀 Innovation Leader</h3>
#                 <h2>{innovation_leader}</h2>
#                 <p>Highest tech score</p>
#             </div>
#             """, unsafe_allow_html=True)

#     def create_cluster_analysis_chart(self, clusters, brand_names, text_embeddings):
#         """Enhanced cluster visualization with PCA"""
#         # Use PCA for better 2D representation
#         pca = PCA(n_components=2, random_state=42)
#         coords_2d = pca.fit_transform(text_embeddings)
        
#         cluster_df = pd.DataFrame({
#             'Brand': brand_names,
#             'Cluster': clusters,
#             'X': coords_2d[:, 0],
#             'Y': coords_2d[:, 1]
#         })
        
#         fig = px.scatter(
#             cluster_df, 
#             x='X', 
#             y='Y', 
#             color='Cluster',
#             text='Brand',
#             title="<b>Brand Clustering Analysis (PCA Projection)</b>",
#             color_continuous_scale='viridis',
#             size_max=20
#         )
        
#         fig.update_traces(
#             textposition="top center",
#             marker=dict(size=15, line=dict(width=2, color='white'))
#         )
        
#         fig.update_layout(
#             template="plotly_white",
#             width=800,
#             height=600,
#             xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
#             yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
#         )
        
#         return fig

#     def create_competitive_landscape(self, tone_df, logo_df):
#         """Create competitive positioning bubble chart"""
#         # Merge data for bubble chart
#         bubble_data = tone_df.merge(logo_df[['Brand', '%With_Logo']], on='Brand')
        
#         fig = go.Figure()
        
#         colors = px.colors.qualitative.Set3
        
#         for i, row in bubble_data.iterrows():
#             fig.add_trace(go.Scatter(
#                 x=[row['premium']],
#                 y=[row['innovation']],
#                 mode='markers+text',
#                 marker=dict(
#                     size=row['%With_Logo'] * 2,  # Logo presence as bubble size
#                     color=colors[i % len(colors)],
#                     opacity=0.7,
#                     line=dict(width=2, color='white')
#                 ),
#                 text=row['Brand'],
#                 textposition="middle center",
#                 textfont=dict(size=10, color='white'),
#                 name=row['Brand'],
#                 hovertemplate=(
#                     '<b>%{text}</b><br>'
#                     'Premium Score: %{x:.1f}%<br>'
#                     'Innovation Score: %{y:.1f}%<br>'
#                     'Logo Presence: %{marker.size:.1f}%<br>'
#                     '<extra></extra>'
#                 )
#             ))
        
#         fig.update_layout(
#             title="<b>Competitive Landscape: Premium vs Innovation</b>",
#             xaxis_title="Premium Positioning (%)",
#             yaxis_title="Innovation Score (%)",
#             template="plotly_white",
#             showlegend=False,
#             width=800,
#             height=600,
#             annotations=[
#                 dict(
#                     text="Bubble size = Logo Presence %",
#                     showarrow=False,
#                     xref="paper", yref="paper",
#                     x=0.02, y=0.98,
#                     font=dict(size=12, color='gray')
#                 )
#             ]
#         )
        
#         # Add quadrant lines
#         fig.add_hline(y=bubble_data['innovation'].median(), line_dash="dash", line_color="gray", opacity=0.5)
#         fig.add_vline(x=bubble_data['premium'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
#         return fig

#     def create_tone_comparison_chart(self, tone_df, selected_brands=None):
#         """Create comparative tone analysis"""
#         if selected_brands:
#             plot_data = tone_df[tone_df['Brand'].isin(selected_brands)].copy()
#         else:
#             plot_data = tone_df.copy()
        
#         categories = [col for col in tone_df.columns if col not in ['Sl No', 'Brand']]
        
#         # Create subplot for each category
#         fig = make_subplots(
#             rows=2, cols=5,
#             subplot_titles=categories,
#             specs=[[{"type": "bar"}]*5, [{"type": "bar"}]*5]
#         )
        
#         colors = px.colors.qualitative.Pastel
        
#         for i, category in enumerate(categories):
#             row = (i // 5) + 1
#             col = (i % 5) + 1
            
#             for j, (_, brand_row) in enumerate(plot_data.iterrows()):
#                 fig.add_trace(
#                     go.Bar(
#                         x=[brand_row['Brand']],
#                         y=[brand_row[category]],
#                         name=brand_row['Brand'] if i == 0 else "",
#                         marker_color=colors[j % len(colors)],
#                         showlegend=(i == 0),
#                         hovertemplate=f'<b>%{{x}}</b><br>{category.title()}: %{{y:.1f}}%<extra></extra>'
#                     ),
#                     row=row, col=col
#                 )
        
#         fig.update_layout(
#             title="<b>Detailed Tone Analysis by Category</b>",
#             template="plotly_white",
#             height=600,
#             width=1200
#         )
        
#         return fig

#     def create_visual_style_analysis(self, visual_df):
#         """Enhanced visual style analysis"""
#         # Create stacked bar chart
#         fig = go.Figure()
        
#         categories = ['%White_BG', '%Face', '%Text_Heavy']
#         colors_map = {'%White_BG': '#ff9999', '%Face': '#66b3ff', '%Text_Heavy': '#99ff99'}
        
#         for category in categories:
#             fig.add_trace(go.Bar(
#                 name=category.replace('%', '').replace('_', ' '),
#                 x=visual_df['Brand'],
#                 y=visual_df[category],
#                 marker_color=colors_map[category],
#                 hovertemplate=f'<b>%{{x}}</b><br>{category}: %{{y:.1f}}%<extra></extra>'
#             ))
        
#         fig.update_layout(
#             title="<b>Visual Style Elements by Brand</b>",
#             xaxis_title="Brands",
#             yaxis_title="Percentage (%)",
#             barmode='group',
#             template="plotly_white",
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
        
#         return fig

#     def create_logo_analysis_dashboard(self, logo_df, logo_ratio_df):
#         """Enhanced logo analysis with multiple visualizations"""
#         # Combine data
#         logo_combined = logo_df.merge(logo_ratio_df[['Brand', 'Avg_Logo_Size_%']], on='Brand')
        
#         # Create subplot
#         fig = make_subplots(
#             rows=1, cols=2,
#             subplot_titles=["Logo Presence (%)", "Average Logo Size (%)"],
#             specs=[[{"type": "bar"}, {"type": "scatter"}]]
#         )
        
#         # Logo presence bar chart
#         fig.add_trace(
#             go.Bar(
#                 x=logo_combined['Brand'],
#                 y=logo_combined['%With_Logo'],
#                 name="Logo Presence",
#                 marker_color=self.colors['info'],
#                 text=logo_combined['%With_Logo'],
#                 texttemplate='%{text:.1f}%',
#                 textposition='auto'
#             ),
#             row=1, col=1
#         )
        
#         # Logo size scatter plot
#         fig.add_trace(
#             go.Scatter(
#                 x=logo_combined['Brand'],
#                 y=logo_combined['Avg_Logo_Size_%'],
#                 mode='markers+lines',
#                 name="Avg Logo Size",
#                 marker=dict(size=12, color=self.colors['warning']),
#                 line=dict(color=self.colors['warning'], width=3)
#             ),
#             row=1, col=2
#         )
        
#         fig.update_layout(
#             title="<b>Logo Strategy Analysis</b>",
#             template="plotly_white",
#             height=500,
#             showlegend=False
#         )
        
#         return fig

#     def create_similarity_distribution(self, similarity_matrix, brand_names):
#         """Create distribution analysis of similarity scores"""
#         # Extract upper triangle of similarity matrix (excluding diagonal)
#         upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
#         fig = make_subplots(
#             rows=1, cols=2,
#             subplot_titles=["Similarity Score Distribution", "Similarity Heatmap"],
#             specs=[[{"type": "histogram"}, {"type": "heatmap"}]]
#         )
        
#         # Histogram
#         fig.add_trace(
#             go.Histogram(
#                 x=upper_triangle,
#                 nbinsx=20,
#                 marker_color=self.colors['primary'],
#                 opacity=0.7,
#                 name="Similarity Scores"
#             ),
#             row=1, col=1
#         )
        
#         # Enhanced heatmap
#         fig.add_trace(
#             go.Heatmap(
#                 z=similarity_matrix,
#                 x=brand_names,
#                 y=brand_names,
#                 colorscale='RdYlBu_r',
#                 text=np.round(similarity_matrix, 3),
#                 texttemplate="%{text}",
#                 textfont={"size": 8}
#             ),
#             row=1, col=2
#         )
        
#         fig.update_layout(
#             title="<b>Similarity Analysis Deep Dive</b>",
#             template="plotly_white",
#             height=500
#         )
        
#         return fig

# def run_enhanced_brand_similarity(st, search_url, num_products=15, out_dir="outputs"):
#     """Enhanced version of the brand similarity analysis with better visuals"""
    
#     analyzer = EnhancedBrandAnalyzer()
    
#     # Create progress bar
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     status_text.text('🔍 Extracting product data...')
#     progress_bar.progress(20)
    
#     # Simulate the data extraction (using the existing logic)
#     # For demo purposes, I'll create sample data that matches the expected structure
    
#     # Sample data structure that would come from your existing extraction logic
#     sample_brands = ['Philips', 'SYSKA', 'Havells', 'Panasonic', 'Nova']
    
#     # Simulate similarity matrices
#     np.random.seed(42)
#     text_sim = np.random.rand(len(sample_brands), len(sample_brands))
#     text_sim = (text_sim + text_sim.T) / 2  # Make symmetric
#     np.fill_diagonal(text_sim, 1.0)
    
#     image_sim = np.random.rand(len(sample_brands), len(sample_brands))
#     image_sim = (image_sim + image_sim.T) / 2
#     np.fill_diagonal(image_sim, 1.0)
    
#     progress_bar.progress(40)
#     status_text.text('🧠 Processing embeddings...')
    
#     # Simulate text embeddings for positioning map
#     text_embeddings = np.random.rand(len(sample_brands), 50)
    
#     # Simulate tone data
#     tone_data = pd.DataFrame({
#         'Brand': sample_brands,
#         'emotional': np.random.rand(len(sample_brands)) * 30 + 10,
#         'functional': np.random.rand(len(sample_brands)) * 25 + 15,
#         'performance': np.random.rand(len(sample_brands)) * 35 + 5,
#         'premium': np.random.rand(len(sample_brands)) * 20 + 5,
#         'financial': np.random.rand(len(sample_brands)) * 15 + 10,
#         'innovation': np.random.rand(len(sample_brands)) * 40 + 10,
#         'aesthetic': np.random.rand(len(sample_brands)) * 25 + 5,
#         'gaming': np.random.rand(len(sample_brands)) * 10 + 2,
#         'masculine': np.random.rand(len(sample_brands)) * 20 + 5,
#         'feminine': np.random.rand(len(sample_brands)) * 15 + 3
#     })
    
#     progress_bar.progress(60)
#     status_text.text('🎨 Analyzing visual elements...')
    
#     # Simulate visual data
#     visual_data = pd.DataFrame({
#         'Brand': sample_brands,
#         '%White_BG': np.random.rand(len(sample_brands)) * 60 + 20,
#         '%Face': np.random.rand(len(sample_brands)) * 30 + 5,
#         '%Text_Heavy': np.random.rand(len(sample_brands)) * 40 + 10
#     })
    
#     # Simulate logo data
#     logo_data = pd.DataFrame({
#         'Brand': sample_brands,
#         'Total_Images': np.random.randint(5, 15, len(sample_brands)),
#         '%With_Logo': np.random.rand(len(sample_brands)) * 80 + 10
#     })
    
#     logo_ratio_data = pd.DataFrame({
#         'Brand': sample_brands,
#         'Avg_Logo_Size_%': np.random.rand(len(sample_brands)) * 15 + 2
#     })
    
#     # Sample color data
#     color_names = ['Blue', 'Red', 'Green', 'Black', 'White', 'Silver', 'Gold']
#     color_data = pd.DataFrame({
#         'Brand': sample_brands,
#         'Top_10_Colors': [', '.join(np.random.choice(color_names, 5, replace=False)) for _ in sample_brands],
#         'Dominant_Color': np.random.choice(color_names, len(sample_brands))
#     })
    
#     progress_bar.progress(80)
#     status_text.text('📊 Creating visualizations...')
    
#     # Generate clusters
#     clusters = np.random.randint(0, 3, len(sample_brands))
    
#     # Create the enhanced dashboard
#     st.markdown("---")
    
#     # Metrics Overview
#     st.markdown("## 📈 Key Metrics Overview")
#     analyzer.create_comprehensive_metrics_dashboard(tone_data, visual_data, logo_data)
    
#     st.markdown("---")
    
#     # Similarity Analysis Section
#     st.markdown("## 🔗 Brand Similarity Analysis")
    
#     tab1, tab2, tab3 = st.tabs(["📊 Similarity Matrices", "🌐 Network View", "📍 Positioning Map"])
    
#     with tab1:
#         col1, col2 = st.columns(2)
#         with col1:
#             text_heatmap = analyzer.create_enhanced_similarity_matrix(
#                 text_sim, sample_brands, "Textual Similarity", "RdYlBu_r"
#             )
#             st.plotly_chart(text_heatmap, use_container_width=True)
        
#         with col2:
#             image_heatmap = analyzer.create_enhanced_similarity_matrix(
#                 image_sim, sample_brands, "Visual Similarity", "Viridis"
#             )
#             st.plotly_chart(image_heatmap, use_container_width=True)
    
#     with tab2:
#         network_fig = analyzer.create_similarity_network(text_sim, sample_brands, threshold=0.6)
#         st.plotly_chart(network_fig, use_container_width=True)
        
#         # Interactive threshold control
#         threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.05)
#         if threshold != 0.6:
#             network_fig_updated = analyzer.create_similarity_network(text_sim, sample_brands, threshold)
#             st.plotly_chart(network_fig_updated, use_container_width=True)
    
#     with tab3:
#         positioning_map = analyzer.create_brand_positioning_map(text_embeddings, sample_brands)
#         st.plotly_chart(positioning_map, use_container_width=True)
    
#     st.markdown("---")
    
#     # Brand Characteristics Analysis
#     st.markdown("## 🎯 Brand Characteristics Analysis")
    
#     tab4, tab5, tab6 = st.tabs(["📊 Tone Analysis", "🎨 Visual Style", "🔍 Detailed Comparison"])
    
#     with tab4:
#         # Radar chart for tone analysis
#         radar_chart = analyzer.create_tone_radar_chart(tone_data)
#         st.plotly_chart(radar_chart, use_container_width=True)
        
#         # Competitive landscape
#         competitive_chart = analyzer.create_competitive_landscape(tone_data, logo_data)
#         st.plotly_chart(competitive_chart, use_container_width=True)
    
#     with tab5:
#         col1, col2 = st.columns(2)
        
#         with col1:
#             visual_chart = analyzer.create_visual_style_analysis(visual_data)
#             st.plotly_chart(visual_chart, use_container_width=True)
        
#         with col2:
#             logo_chart = analyzer.create_logo_analysis_dashboard(logo_data, logo_ratio_data)
#             st.plotly_chart(logo_chart, use_container_width=True)
    
#     with tab6:
#         # Brand comparison selector
#         st.markdown("### 🔍 Custom Brand Comparison")
#         selected_brands = st.multiselect(
#             "Select brands to compare",
#             sample_brands,
#             default=sample_brands[:3]
#         )
        
#         if selected_brands:
#             comparison_chart = analyzer.create_tone_comparison_chart(tone_data, selected_brands)
#             st.plotly_chart(comparison_chart, use_container_width=True)
            
#             # Side-by-side comparison table
#             comparison_df = tone_data[tone_data['Brand'].isin(selected_brands)].copy()
#             st.markdown("#### 📋 Detailed Comparison Table")
#             st.dataframe(comparison_df.style.format(precision=1).highlight_max(axis=0))
    
#     st.markdown("---")
    
#     # Color Analysis Section
#     st.markdown("## 🌈 Color Palette Analysis")
    
#     color_freq_chart, color_diversity_chart = analyzer.create_color_analysis_chart(color_data)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(color_freq_chart, use_container_width=True)
#     with col2:
#         st.plotly_chart(color_diversity_chart, use_container_width=True)
    
#     st.markdown("---")
    
#     # Clustering Analysis
#     st.markdown("## 🎯 Brand Clustering & Segmentation")
    
#     cluster_chart = analyzer.create_cluster_analysis_chart(clusters, sample_brands, text_embeddings)
#     st.plotly_chart(cluster_chart, use_container_width=True)
    
#     # Cluster summary table
#     cluster_summary = pd.DataFrame({
#         'Cluster': range(len(set(clusters))),
#         'Brands': [', '.join([sample_brands[i] for i in range(len(clusters)) if clusters[i] == c]) 
#                   for c in range(len(set(clusters)))],
#         'Size': [sum(clusters == c) for c in range(len(set(clusters)))]
#     })
    
#     st.markdown("#### 📊 Cluster Summary")
#     st.dataframe(cluster_summary, use_container_width=True)
    
#     progress_bar.progress(100)
#     status_text.text('✅ Analysis complete!')
    
#     # Advanced Analytics Section
#     st.markdown("---")
#     st.markdown("## 🔬 Advanced Analytics")
    
#     # Similarity distribution analysis
#     similarity_dist_chart = analyzer.create_similarity_distribution(text_sim, sample_brands)
#     st.plotly_chart(similarity_dist_chart, use_container_width=True)
    
#     # Brand insights box
#     st.markdown("""
#     <div class="insight-box">
#         <h3>🧠 Key Insights</h3>
#         <ul>
#             <li><b>Market Leaders:</b> Brands with highest premium and innovation scores</li>
#             <li><b>Visual Strategy:</b> Logo presence correlates with brand recognition</li>
#             <li><b>Positioning:</b> Clear clusters emerge based on tone and visual elements</li>
#             <li><b>Opportunities:</b> Gaps in competitive landscape reveal positioning opportunities</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Export section
#     st.markdown("---")
#     st.markdown("## 💾 Export Results")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("📊 Download Charts as HTML"):
#             st.success("Charts exported successfully!")
    
#     with col2:
#         if st.button("📈 Download Data as Excel"):
#             st.success("Data exported successfully!")
    
#     with col3:
#         if st.button("📋 Generate Report"):
#             st.success("Report generated successfully!")


# # Enhanced main application
# def main():
#     st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
#     # URL input with validation
#     search_url = st.sidebar.text_input(
#         "🔗 Amazon Search URL", 
#         value="https://www.amazon.in/s?k=trimmer",
#         help="Enter the Amazon search URL for products to analyze"
#     )
    
#     # Advanced options
#     with st.sidebar.expander("⚙️ Advanced Options"):
#         num_products = st.slider("Number of Products", 5, 30, 15)
#         similarity_threshold = st.slider("Network Similarity Threshold", 0.0, 1.0, 0.6, 0.05)
#         enable_ocr = st.checkbox("Enable OCR Analysis", value=True)
#         color_analysis = st.checkbox("Enable Color Analysis", value=True)
    
#     # Analysis type selection
#     analysis_type = st.sidebar.radio(
#         "📊 Analysis Type",
#         ["🏢 Brand Similarity", "⭐ Review Analysis", "🔄 Combined Analysis"]
#     )
    
#     # Run analysis button
#     run_analysis = st.sidebar.button(
#         "🚀 Run Analysis", 
#         type="primary",
#         use_container_width=True
#     )
    
#     if run_analysis:
#         if analysis_type == "🏢 Brand Similarity":
#             results = run_brand_similarity(search_url, num_products=num_products, out_dir="outputs")

#             # Unpack results
#             brand_names = results["brand_names"]
#             text_sim = results["text_sim"]
#             image_sim = results["image_sim"]
#             clusters = results["clusters"]
#             tone_df = results["tone_df"]
#             color_df = results["color_df"]
#             logo_df = results["logo_df"]
#             logo_ratio_df = results["logo_ratio_df"]
#             visual_df = results["visual_df"]
#             text_embeddings = results["text_embeddings"]
#             insight_text = results["insight_text"]

#             analyzer = EnhancedBrandAnalyzer()

#             st.markdown("## 📈 Key Metrics Overview")
#             analyzer.create_comprehensive_metrics_dashboard(tone_df, visual_df, logo_df)

#             st.markdown("## 🔗 Brand Similarity Analysis")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.plotly_chart(analyzer.create_enhanced_similarity_matrix(text_sim, brand_names, "Textual Similarity", "RdYlBu_r"), use_container_width=True)
#             with col2:
#                 st.plotly_chart(analyzer.create_enhanced_similarity_matrix(image_sim, brand_names, "Visual Similarity", "Viridis"), use_container_width=True)

#             st.plotly_chart(analyzer.create_similarity_network(text_sim, brand_names, threshold=0.6), use_container_width=True)
#             st.plotly_chart(analyzer.create_brand_positioning_map(text_embeddings, brand_names), use_container_width=True)

#             st.markdown("## 🎯 Brand Characteristics Analysis")
#             st.plotly_chart(analyzer.create_tone_radar_chart(tone_df), use_container_width=True)
#             st.plotly_chart(analyzer.create_competitive_landscape(tone_df, logo_df), use_container_width=True)
#             st.plotly_chart(analyzer.create_visual_style_analysis(visual_df), use_container_width=True)
#             st.plotly_chart(analyzer.create_logo_analysis_dashboard(logo_df, logo_ratio_df), use_container_width=True)

#             st.markdown("## 🌈 Color Palette Analysis")
#             color_freq_chart, color_diversity_chart = analyzer.create_color_analysis_chart(color_df)
#             st.plotly_chart(color_freq_chart, use_container_width=True)
#             st.plotly_chart(color_diversity_chart, use_container_width=True)

#             st.markdown("## 🎯 Brand Clustering & Segmentation")
#             st.plotly_chart(analyzer.create_cluster_analysis_chart(clusters, brand_names, text_embeddings), use_container_width=True)

#             st.markdown("## 🔬 Advanced Analytics")
#             st.plotly_chart(analyzer.create_similarity_distribution(text_sim, brand_names), use_container_width=True)

#             st.subheader("🤖 AI Insights on Brand Similarity")
#             st.markdown(insight_text)

#     # Sidebar info
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("""
#     ### 📚 How to Use
#     1. **Enter URL**: Paste Amazon search URL
#     2. **Configure**: Set analysis parameters
#     3. **Run**: Click the analysis button
#     4. **Explore**: Use interactive charts
#     5. **Export**: Download results
    
#     ### 🎯 Features
#     - Interactive similarity matrices
#     - Brand positioning maps
#     - Tone analysis radar charts
#     - Color palette analysis
#     - Logo presence tracking
#     - Competitive landscape view
#     """)

# if __name__ == "__main__":

#     main()
