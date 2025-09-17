import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from googleapiclient.discovery import build
# from youtube_transcript_api import YouTubeTranscriptApi
import isodate
import time
import random
from functools import lru_cache
import os
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import urllib.request, json
import requests
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import networkx as nx
import plotly.graph_objects as go


# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="YouTube Embedding App", layout="wide")
st.title("🎬 YouTube Embedding & Similarity Analysis")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("YouTube Search")
query = st.sidebar.text_input("Enter search query:", value="AI technology")
max_results = st.sidebar.slider("Number of results", 5, 20, 10)
include_transcripts = st.sidebar.checkbox("Include transcripts (slower)", value=True)
run_button = st.sidebar.button("Run Search")

# -----------------------------
# API Configuration
# -----------------------------
# IMPORTANT: Replace with your actual API key
API_KEY = "AIzaSyBdrsz6LnGaDwgdGCrf5-NqcTnPlfy6bDY"  # Replace with your actual API key
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Rate limiting configuration
TRANSCRIPT_DELAY = 2  # seconds between transcript requests
MAX_RETRIES = 3

# -----------------------------
# Description extraction Function
# -----------------------------
def get_full_description(video_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(video_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        desc_elements = soup.find_all("yt-formatted-string", class_="yt-core-attributed-string--link-inherit-color")

        # Join all text pieces
        full_desc = " ".join([el.get_text(separator=" ", strip=True) for el in desc_elements])

        return full_desc if full_desc.strip() else ""
    except Exception as e:
        return f"Description not available ({str(e)})"

# -----------------------------
# Enhanced Helper Functions
# -----------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def youtube_search(query, max_results=10):
    """Search YouTube videos with caching"""
    if API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        st.error("⚠️ Please set your YouTube API key in the code!")
        return pd.DataFrame()
    
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=max_results
        ).execute()

        videos = []
        video_ids = [item["id"]["videoId"] for item in search_response["items"]]
        
        # Batch request for video statistics
        video_response = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(video_ids)
        ).execute()
        
        # Batch request for channel statistics
        channel_ids = [item["snippet"]["channelId"] for item in search_response["items"]]
        channel_response = youtube.channels().list(
            part="statistics",
            id=",".join(set(channel_ids))  # Remove duplicates
        ).execute()
        
        # Create lookup for channel subscribers
        channel_subs = {ch["id"]: ch["statistics"].get("subscriberCount", "0") 
                       for ch in channel_response["items"]}

        for i, item in enumerate(search_response["items"]):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            channel_id = item["snippet"]["channelId"]
            channel = item["snippet"]["channelTitle"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
            published_at = item["snippet"]["publishedAt"]
            description = get_full_description(url)
            if not description:
                description = item["snippet"].get("description", "")

            # Get statistics from batch response
            stats = video_response["items"][i]["statistics"]
            content = video_response["items"][i]["contentDetails"]

            views = int(stats.get("viewCount", "0"))
            likes = int(stats.get("likeCount", "0"))
            comments = int(stats.get("commentCount", "0"))
            duration = isodate.parse_duration(content["duration"]).total_seconds()
            subscribers = int(channel_subs.get(channel_id, "0"))

            videos.append({
                "video_id": video_id,
                "title": title,
                "description": description,  # ✅ Add this line
                "url": url,
                "channel": channel,
                "subscribers": subscribers,
                "views": views,
                "likes": likes,
                "comments": comments,
                "duration_seconds": duration,
                "published": published_at,
                "thumbnail": thumbnail,
                "transcript": ""  # Will be filled later if requested
            })

        return pd.DataFrame(videos)
    
    except Exception as e:
        st.error(f"Error searching YouTube: {str(e)}")
        return pd.DataFrame()


# 🔧 Set FFmpeg path manually (adjust if your ffmpeg is in a different folder)
FFMPEG_PATH = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# -----------------------------
# Audio Download + Transcription
# -----------------------------
def download_audio(video_url, output_path="sample_audio.mp3"):
    """Download YouTube video audio using yt-dlp"""
    def fetch_video_metadata(video_url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,   # don’t download video/audio
        "noplaylist": True,
        "extract_flat": False,   # get full metadata, not just flat list
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info

    # Ensure correct extension
    if not output_path.endswith(".mp3"):
        output_path += ".mp3"
    if not os.path.exists(output_path):
        if os.path.exists(output_path.replace(".mp3", "") + ".mp3"):
            os.rename(output_path.replace(".mp3", "") + ".mp3", output_path)
    return output_path

def transcribe_audio(audio_file, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # 👇 auto-detect language, but force English output
    segments, _ = model.transcribe(audio_file, task="translate")
    transcript = " ".join([seg.text.strip() for seg in segments])
    return transcript

def get_transcript_from_youtube(video_url):
    """Download audio and transcribe instead of Selenium transcript"""
    try:
        audio_file = download_audio(video_url, "temp_audio.mp3")
        transcript = transcribe_audio(audio_file)
        return transcript if transcript.strip() else "Transcript not available"
    except Exception as e:
        return f"Transcript not available ({str(e)})"

    
@st.cache_data
def compute_embeddings(texts):
    """Compute embeddings with caching"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings)

def analyze_similarity(df, text_column="title"):
    """Analyze similarity between texts"""
    embeddings = compute_embeddings(df[text_column].tolist())
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix

def plot_heatmap(sim_matrix, labels, title="Video Similarity Heatmap"):
    fig, ax = plt.subplots(figsize=(12, 8))
    # short_labels = [label[:30] + "..." if len(label) > 30 else label for label in labels]
    short_labels = [f"#{i+1} {label[:30]}..." if len(label) > 30 else f"#{i+1} {label}" 
                for i, label in enumerate(labels)]

    sns.heatmap(sim_matrix,
                xticklabels=short_labels,
                yticklabels=short_labels,
                cmap="viridis",
                annot=True,
                fmt='.2f',
                square=True)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def format_number(num):
    """Format large numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def format_duration(seconds):
    """Format duration in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

# -----------------------------
# Main Application Logic
# -----------------------------
if API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
    st.warning("⚠️ Please set your YouTube API key in the code to use this app!")
    st.info("You need to:")
    st.write("1. Get a YouTube Data API key from Google Cloud Console")
    st.write("2. Replace 'YOUR_YOUTUBE_API_KEY_HERE' with your actual API key")
    st.write("3. Enable YouTube Data API v3 in your Google Cloud project")

if run_button and API_KEY != "YOUR_YOUTUBE_API_KEY_HERE":
    df = youtube_search(query, max_results)
    if not df.empty:
        st.session_state['video_df'] = df
        st.session_state['query'] = query
        st.session_state['transcripts_done'] = False
        # Ensure rank column exists
        if "rank" not in df.columns:
            df.insert(0, "rank", range(1, len(df) + 1))

        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_views = df['views'].sum()
            st.metric("Total Views", format_number(total_views))
        with col2:
            avg_duration = df['duration_seconds'].mean()
            st.metric("Avg Duration", format_duration(avg_duration))
        with col3:
            total_likes = df['likes'].sum()
            st.metric("Total Likes", format_number(total_likes))
        with col4:
            unique_channels = df['channel'].nunique()
            st.metric("Unique Channels", unique_channels)

        # Video display with enhanced formatting
        st.subheader("🎥 Video Results")
        for idx, row in df.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(row['thumbnail'], width=150)
                with col2:
                    st.markdown(f"**[{row['title']}]({row['url']})**")
                    st.write(f"📺 **{row['channel']}** • {format_number(row['subscribers'])} subscribers")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    with metrics_col1:
                        st.write(f"👁️ {format_number(row['views'])}")
                    with metrics_col2:
                        st.write(f"👍 {format_number(row['likes'])}")
                    with metrics_col3:
                        st.write(f"💬 {format_number(row['comments'])}")
                    with metrics_col4:
                        st.write(f"⏱️ {format_duration(row['duration_seconds'])}")
                
                st.divider()

        # Transcript fetching (optional)
        if 'video_df' in st.session_state:
            df = st.session_state['video_df']
            SAVE_DIR = "runs"
            os.makedirs(SAVE_DIR, exist_ok=True)

            transcript_status = st.empty()
            
            if include_transcripts and not st.session_state.get('transcripts_done', False):
                st.subheader("📝 Fetching Transcripts...")
                for idx, row in df.iterrows():
                    transcript = get_transcript_from_youtube(row['url'])
                    df.at[idx, 'transcript'] = transcript
                
                # Save the run
                run_id = f"{query.replace(' ', '_')}_{int(time.time())}"
                file_path = os.path.join(SAVE_DIR, f"{run_id}.parquet")
                df.to_parquet(file_path, index=False)

                # Store path in session state
                st.session_state['current_run_file'] = file_path
                st.session_state['transcripts_done'] = True
            
            transcript_status.text("✅ All transcripts processed!")
            
            # Show transcript preview
            st.subheader("📜 Transcript Previews")
            for idx, row in df.iterrows():
                with st.expander(f"Transcript: {row['title'][:50]}..."):
                    if "not available" in row['transcript'].lower() or "error" in row['transcript'].lower():
                        st.warning(row['transcript'])
                    else:
                        # Show first 500 characters
                        preview = row['transcript'][:500] + "..." if len(row['transcript']) > 500 else row['transcript']
                        st.write(preview)

        # --- Always check if data exists in session_state ---
        if 'video_df' in st.session_state:
            df = st.session_state['video_df']

            # Fetch transcripts only once
            if include_transcripts and not st.session_state.get('transcripts_done', False):
                for idx, row in df.iterrows():
                    transcript = get_transcript_from_youtube(row['url'])
                    df.at[idx, 'transcript'] = transcript
                st.session_state['video_df'] = df
                st.session_state['transcripts_done'] = True

            # -----------------------------
            # Precompute Embeddings
            # -----------------------------
            st.subheader("⚡ Computing Embeddings...")

            # Title + Description
            combined_td = (df['title'].fillna('') + " " + df['description'].fillna(''))
            embeddings_td = compute_embeddings(combined_td.tolist())
            sim_matrix_td = cosine_similarity(embeddings_td)

            # Title + Description + Transcript
            combined_tdt = (df['title'].fillna('') + " " + df['description'].fillna('') + " " + df['transcript'].fillna(''))
            embeddings_tdt = compute_embeddings(combined_tdt.tolist())
            sim_matrix_tdt = cosine_similarity(embeddings_tdt)


            # -----------------------------
            # Interactive Clustering Plot
            # -----------------------------
            # st.subheader("🎯 Video Clustering View")

            cluster_option = st.radio(
                "Select Embedding Basis:",
                ["Title + Description", "Title + Description + Transcript"],
                horizontal=True
            )

            # if cluster_option == "Title + Description":
            #     embeddings = embeddings_td
            #     labels = "Title + Description"
            # else:
            #     embeddings = embeddings_tdt
            #     labels = "Title + Description + Transcript"
            
            # # embeddings = embeddings_td or embeddings_tdt depending on your selection
            # n_samples = len(embeddings)

            # # Auto-adjust perplexity
            # perplexity = min(30, max(2, n_samples - 1))

            # # Use t-SNE for better separation
            # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            # coords = tsne.fit_transform(embeddings)

            # cluster_df = pd.DataFrame({
            #     "x": coords[:, 0],
            #     "y": coords[:, 1],
            #     "title": df["title"],
            #     "rank": df["rank"],
            #     "views": df["views"],
            #     "likes": df["likes"]
            # })

            # fig_cluster = px.scatter(
            #     cluster_df,
            #     x="x",
            #     y="y",
            #     size="views",                # bubble size by views
            #     color="rank",                # color by rank
            #     hover_name="title",
            #     hover_data=["rank", "views", "likes"],
            #     title=f"Video Clusters based on {labels}"
            # )
            # fig_cluster.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
            # fig_cluster.update_layout(
            #     font=dict(size=14),
            #     title=dict(font=dict(size=20)),
            #     xaxis=dict(showgrid=False, zeroline=False),
            #     yaxis=dict(showgrid=False, zeroline=False)
            # )
            # st.plotly_chart(fig_cluster, use_container_width=True)

            st.subheader("🌐 Video Similarity Network")

            # Pick embeddings
            if cluster_option == "Title + Description":
                embeddings = embeddings_td
                labels = "Title + Description"
            else:
                embeddings = embeddings_tdt
                labels = "Title + Description + Transcript"

            # Cosine similarity
            sim_matrix = cosine_similarity(embeddings)

            # Build graph
            threshold = 0.7
            G = nx.Graph()

            for i, row in df.iterrows():
                G.add_node(
                    i,
                    title=row["title"],
                    rank=row["rank"],
                    views=row["views"],
                    likes=row["likes"]
                )

            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if sim_matrix[i, j] >= threshold:
                        G.add_edge(i, j, weight=sim_matrix[i, j])

            # Layout
            pos = nx.spring_layout(G, seed=42, k=0.5)

            # Edges
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="gray"),
                mode="lines",
                hoverinfo="none"
            )

            # Nodes
            node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
            for node in G.nodes(data=True):
                idx, meta = node
                x, y = pos[idx]
                node_x.append(x)
                node_y.append(y)
                # Scaled size: log transform for clarity
                node_size.append(np.log1p(meta["views"]) * 5 + 10)
                node_color.append(meta["rank"])
                node_text.append(
                    f"#{meta['rank']} {meta['title']}<br>Views: {meta['views']} | Likes: {meta['likes']}"
                )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[f"#{meta['rank']}" for _, meta in G.nodes(data=True)],  # rank label
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale="Viridis",
                    showscale=False,  # turn off continuous scale
                    line_width=1.5
                )
            )

            # Plot
            fig_network = go.Figure(data=[edge_trace, node_trace])
            fig_network.update_layout(
                title_text=f"Video Similarity Network ({labels})",
                title_font=dict(size=20),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                font=dict(size=12),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )

            st.plotly_chart(fig_network, use_container_width=True)


            # -----------------------------
            # Similarity Analysis
            # -----------------------------
            st.subheader("🔍 Similarity Analysis")

            # 1. Title + Description
            st.markdown("### 📌 Similarity based on Title + Description")
            combined_td = (df['title'].fillna('') + " " + df['description'].fillna(''))
            embeddings_td = compute_embeddings(combined_td.tolist())
            sim_matrix_td = cosine_similarity(embeddings_td)
            st.pyplot(plot_heatmap(sim_matrix_td, df['title'].tolist(), "Title + Description Similarity"))

            # 2. Title + Description + Transcript
            st.markdown("### 📌 Similarity based on Title + Description + Transcript")
            combined_tdt = (df['title'].fillna('') + " " +
                            df['description'].fillna('') + " " +
                            df['transcript'].fillna(''))
            embeddings_tdt = compute_embeddings(combined_tdt.tolist())
            sim_matrix_tdt = cosine_similarity(embeddings_tdt)
            st.pyplot(plot_heatmap(sim_matrix_tdt, df['title'].tolist(), "Title + Description + Transcript Similarity"))

            # Save matrices for Excel export
            sim_matrices = {
                "Similarity_Title_Description": sim_matrix_td,
                "Similarity_Title_Description_Transcript": sim_matrix_tdt
            }


        # -----------------------------
        # Performance Distribution (Plotly)
        # -----------------------------
        # st.subheader("📈 Video Performance Distribution")
        # df['performance_score'] = (df['likes'] + df['comments']) / (df['views'] + 1)
        # fig_perf = px.scatter(
        #     df,
        #     x="views",
        #     y="likes",
        #     size="performance_score",
        #     color="performance_score",
        #     hover_name="title",
        #     hover_data={"channel": True, "comments": True, "duration_seconds": True},
        #     color_continuous_scale="viridis",
        #     title="Video Performance: Views vs Likes"
        # )
        # st.plotly_chart(fig_perf, use_container_width=True)

        # -----------------------------
        # Performance Distribution (Ranked)
        # -----------------------------
        st.subheader("📈 Video Performance Distribution")

        df["performance_score"] = (df["likes"] + df["comments"]) / (df["views"] + 1)
        df["performance_rank"] = df["performance_score"].rank(ascending=False).astype(int)
        df["ranked_title"] = "#" + df["performance_rank"].astype(str) + " " + df["title"]

        fig_perf = px.scatter(
            df,
            x="views",
            y="likes",
            size="performance_score",
            color="performance_rank",
            hover_name="ranked_title",
            hover_data={"performance_score": True, "comments": True, "duration_seconds": True},
            color_continuous_scale="viridis",
            title="Video Performance: Views vs Likes"
        )

        fig_perf.update_layout(
            font=dict(size=36),  # general text
            title=dict(text="Video Performance: Views vs Likes", font=dict(size=24)),  # much bigger title
            xaxis=dict(
                title=dict(text="Views", font=dict(size=20)),
                tickfont=dict(size=18)   # tick labels bigger
            ),
            yaxis=dict(
                title=dict(text="Likes", font=dict(size=20)),
                tickfont=dict(size=18)   # tick labels bigger
            ),
            legend=dict(font=dict(size=16))  # legend text bigger
        )

        st.plotly_chart(fig_perf, use_container_width=True)

        # -----------------------------
        # Duration Distribution (Interactive)
        # -----------------------------
        # st.subheader("⏱ Video Duration Distribution")

        # df['duration_minutes'] = (df['duration_seconds'] / 60).round().astype(int)

        # fig_dur = px.histogram(
        #     df,
        #     x="duration_minutes",
        #     nbins=20,
        #     title="Video Duration Distribution (Minutes)",
        #     labels={"duration_minutes": "Duration (minutes)", "count": "Number of Videos"},
        #     hover_data=["title", "channel"]
        # )

        # fig_dur.update_layout(
        #     bargap=0.1,
        #     xaxis=dict(dtick=1),  # force whole numbers on x-axis
        #     yaxis_title="Number of Videos"
        # )

        # st.plotly_chart(fig_dur, use_container_width=True)

        # -----------------------------
        # Video Duration vs Rank
        # -----------------------------
        st.subheader("⏱ Video Duration by Rank")

        df["duration_minutes"] = (df["duration_seconds"] / 60).round().astype(int)

        fig_dur = px.bar(
            df.sort_values("rank"),
            x="rank",
            y="duration_minutes",
            text="duration_minutes",
            hover_name="title",
            title="Video Duration by Rank (minutes)",
            labels={"duration_minutes": "Duration (minutes)", "rank": "Video Rank"}
        )

        fig_dur.update_traces(textposition="outside")
        fig_dur.update_layout(
            font=dict(size=16),
            title=dict(text="Video Duration by Rank (minutes)", font=dict(size=24)),
            xaxis=dict(
                title=dict(text="Rank", font=dict(size=20)),
                tickfont=dict(size=18)
            ),
            yaxis=dict(
                title=dict(text="Duration (minutes)", font=dict(size=20)),
                tickfont=dict(size=18)
            ),
            legend=dict(font=dict(size=16))
        )


        st.plotly_chart(fig_dur, use_container_width=True)



        # # Additional Analytics
        # st.subheader("📈 Additional Analytics")
        
        # # Performance metrics
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.write("**📊 Performance Distribution**")
        #     fig, ax = plt.subplots(figsize=(8, 6))
            
        #     # Create performance score (normalized combination of views, likes, comments)
        #     df['performance_score'] = (
        #         (df['views'] / df['views'].max()) * 0.5 +
        #         (df['likes'] / df['likes'].max()) * 0.3 +
        #         (df['comments'] / df['comments'].max()) * 0.2
        #     )
            
        #     scatter = ax.scatter(
        #         df['views'], df['likes'],
        #         s=df['performance_score']*200,
        #         alpha=0.6,
        #         c=df['performance_score'],
        #         cmap='viridis',
        #         label=df['title']  # attach titles for legend
        #     )

        #     # Add a colorbar for performance score
        #     cbar = plt.colorbar(scatter, ax=ax)
        #     cbar.set_label("Performance Score")

        #     # Add labels for top N videos (to avoid clutter, e.g. top 5 by views)
        #     top_n = df.nlargest(5, 'views')
        #     for _, row in top_n.iterrows():
        #         ax.text(row['views'], row['likes'], row['title'][:20] + "...",
        #                 fontsize=8, ha='right', va='bottom')

        #     ax.set_xlabel('Views')
        #     ax.set_ylabel('Likes')
        #     ax.set_title('Video Performance: Views vs Likes')

        #     plt.tight_layout()
        #     st.pyplot(fig)
        
        # with col2:
        #     st.write("**⏱️ Duration Analysis**")
        #     fig, ax = plt.subplots(figsize=(8, 6))
            
        #     # Create duration bins
        #     df['duration_minutes'] = df['duration_seconds'] / 60
        #     bins = [0, 5, 10, 20, 30, 60, float('inf')]
        #     labels = ['0-5min', '5-10min', '10-20min', '20-30min', '30-60min', '60min+']
        #     df['duration_category'] = pd.cut(df['duration_minutes'], bins=bins, labels=labels)
            
        #     duration_counts = df['duration_category'].value_counts()
        #     ax.bar(duration_counts.index, duration_counts.values)
        #     ax.set_title('Video Duration Distribution')
        #     ax.set_xlabel('Duration Category')
        #     ax.set_ylabel('Number of Videos')
        #     plt.xticks(rotation=45)
        #     plt.tight_layout()
        #     st.pyplot(fig)

        # Top performers
        st.subheader("🏆 Top Performing Videos")
        top_videos = df.nlargest(3, 'performance_score')[['title', 'channel', 'views', 'likes', 'performance_score']]
        for idx, row in top_videos.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"   Channel: {row['channel']} | Views: {format_number(row['views'])} | Likes: {format_number(row['likes'])} | Score: {row['performance_score']:.3f}")

        class InsightGenerator:
            def __init__(self, api_key="Your_Key", deployment="grok-3"):
                self.url = "https://ai-api-dev.dentsu.com/foundry/chat/completions?api-version=2024-05-01-preview"
                self.deployment = deployment
                self.headers = {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                    "x-service-line": "media",
                    "x-brand": "dentsu",
                    "x-project": "YouTube Similarity Dashboard",
                    "api-version": "v1",
                    "Ocp-Apim-Subscription-Key": api_key
                }

            def generate_insight(self, sim_df_td, sim_df_tdt, df):
                # Build structured prompt
                summary_prompt = f"""
                You are analyzing YouTube videos for brand/content similarity.

                Title + Description Similarity Matrix:
                {sim_df_td.to_string()}

                Title + Description + Transcript Similarity Matrix:
                {sim_df_tdt.to_string()}
                
                Metadata of videos:
                {df[['title','channel','views','likes','comments']].to_string(index=False)}

                Based on this, explain concisely:
                1. Which videos are most similar to each other and why.
                2. What factors (title, description, transcript) contribute to similarity.
                3. Where the strongest differences appear.
                4. Provide actionable insights in 3 to 4 bullet points to describe what are the traits that can 
                contribute to better advertising and seo visibility to come on top of searches.
                """

                system_prompt = """You are an expert video marketing analyst.
                Your task is to interpret similarity results and provide clear, concise professional insights, make sure the complete insight is within 500 words/tokens"""

                data = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": summary_prompt}
                    ],
                    "model": self.deployment,
                    "temperature": 0.2,
                    "max_tokens": 500
                }

                try:
                    req = urllib.request.Request(
                        self.url,
                        headers=self.headers,
                        data=json.dumps(data).encode("utf-8")
                    )
                    req.get_method = lambda: "POST"

                    with urllib.request.urlopen(req) as response:
                        resp_data = response.read().decode("utf-8")
                        resp_json = json.loads(resp_data)
                        return resp_json["choices"][0]["message"]["content"]

                except Exception as e:
                    return f"❌ Insight generation failed: {e}"
                
            def explain_pairwise_similarity(self, v1, v2):
                """
                v1 and v2 should be dicts like:
                {"title": ..., "description": ..., "transcript": ...}
                """
                summary_prompt = f"""
                Compare the following two YouTube videos and explain why they might be considered similar:

                Video 1:
                Title: {v1['title']}
                Description: {v1['description']}
                Transcript (first 500 chars): {v1['transcript'][:500]}

                Video 2:
                Title: {v2['title']}
                Description: {v2['description']}
                Transcript (first 500 chars): {v2['transcript'][:500]}

                Write your explanation as if you are a human analyst, not a machine.
                Be specific and natural:
                - Point out what themes, keywords, or angles make them similar.
                - Highlight at least one small difference in focus or tone between them.
                - Use slightly different wording/style each time (avoid repeating the same phrases).
                Keep the explanation short (3–4 sentences).
                """

                system_prompt = """You are an expert marketing/video analyst.
                Your job is to explain in plain, human-like language why two videos feel similar,
                while also mentioning subtle differences to make the explanation credible.
                Avoid repeating the same template across explanations and writing 'Hey there'
                                                
                Write your explanation as if you are a human analyst, not a machine.
                Be specific and natural:
                - Point out what themes, keywords, or angles make them similar.
                - Highlight at least one small difference in focus or tone between them.
                - Use slightly different wording/style each time (avoid repeating the same phrases).
                Keep the explanation short (3–4 sentences).
                """

                # system_prompt = """You are an expert marketing/video analyst.
                # Your job is to explain in plain, human-like language why two videos feel similar,
                # while also mentioning subtle differences to make the explanation credible.
                # Avoid repeating the same template across explanations and writing 'Hey there'.

                data = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": summary_prompt}
                    ],
                    "model": self.deployment,
                    "temperature": 0.3,
                    "max_tokens": 250
                }

                try:
                    req = urllib.request.Request(
                        self.url,
                        headers=self.headers,
                        data=json.dumps(data).encode("utf-8")
                    )
                    req.get_method = lambda: "POST"

                    with urllib.request.urlopen(req) as response:
                        resp_data = response.read().decode("utf-8")
                        resp_json = json.loads(resp_data)
                        return resp_json["choices"][0]["message"]["content"]

                except Exception as e:
                    return f"❌ Explanation failed: {e}"        

        # -----------------------------
        # Most Similar Videos
        # -----------------------------
        # st.subheader("🔗 Most Similar Video Pairs")

        # # Ensure rank column exists
        # if 'rank' not in df.columns:
        #     df['rank'] = range(1, len(df) + 1)

        # # Use Title+Description+Transcript similarity
        # pairs = []
        # titles = df['title'].tolist()

        # for i in range(len(titles)):
        #     for j in range(i + 1, len(titles)):
        #         pairs.append({
        #             "Video 1": f"#{df.iloc[i]['rank']} {titles[i]}",
        #             "Video 2": f"#{df.iloc[j]['rank']} {titles[j]}",
        #             "Similarity": sim_matrix_tdt[i, j]
        #         })

        # pairs_df = pd.DataFrame(pairs).sort_values(by="Similarity", ascending=False).head(10)

        # st.dataframe(pairs_df.reset_index(drop=True))

        # -----------------------------
        # Most Similar Video Pairs (Scatter)
        # -----------------------------
        st.subheader("🔗 Most Similar Video Pairs")

        pairs = []
        titles = df["title"].tolist()
        ranks = df["rank"].tolist()

        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                pairs.append({
                    "Video 1": f"#{ranks[i]} {titles[i][:40]}",
                    "Video 2": f"#{ranks[j]} {titles[j][:40]}",
                    "Similarity": sim_matrix_tdt[i, j]
                })

        pairs_df = pd.DataFrame(pairs).sort_values(by="Similarity", ascending=False).head(20)

        fig_pairs = px.scatter(
            pairs_df,
            x="Video 1",
            y="Video 2",
            size="Similarity",
            color="Similarity",
            hover_name="Similarity",
            color_continuous_scale="blues",
            title="Most Similar Video Pairs"
        )

        fig_pairs.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        fig_pairs.update_layout(
            title_text="Most Similar Video Pairs",
            title_font=dict(size=20),
            xaxis_title="Video 1",
            yaxis_title="Video 2",
            font=dict(size=14)
        )
        st.plotly_chart(fig_pairs, use_container_width=True)


        # -----------------------------
        # Why Are They Similar? (LLM Explanations)
        # -----------------------------
        st.subheader("🧐 Why Are They Similar?")

        explanations = []
        insight_gen = InsightGenerator(api_key="79fb00c0680e4137acd298b1466e8fdd")

        for _, row in pairs_df.iterrows():
            v1 = df.iloc[int(row.name.split('-')[0])] if isinstance(row.name, str) else None
            # Instead of relying on index parsing, use rank lookup
            v1_title = row["Video 1"].split(" ", 1)[1]
            v2_title = row["Video 2"].split(" ", 1)[1]

            v1_row = df[df['title'].str.contains(v1_title[:20], case=False)].iloc[0].to_dict()
            v2_row = df[df['title'].str.contains(v2_title[:20], case=False)].iloc[0].to_dict()

            explanation = insight_gen.explain_pairwise_similarity(v1_row, v2_row)
            explanations.append({
                "Video Pair": f"{row['Video 1']} ↔ {row['Video 2']}",
                "Reason": explanation
            })

        explanations_df = pd.DataFrame(explanations)
        st.dataframe(explanations_df)



        # Download Options
        st.subheader("💾 Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Video Data', index=False)

                # Title + Description
                sim_df_td = pd.DataFrame(sim_matrix_td, index=df['title'], columns=df['title'])
                sim_df_td.to_excel(writer, sheet_name='Similarity_Title_Description')

                # Title + Description + Transcript
                sim_df_tdt = pd.DataFrame(sim_matrix_tdt, index=df['title'], columns=df['title'])
                sim_df_tdt.to_excel(writer, sheet_name='Similarity_Title_Description_Transcript')


                if df['description'].str.strip().any():
                    valid_desc = df[df['description'].str.strip() != ""]
                    if len(valid_desc) > 1:
                        sim_desc = analyze_similarity(valid_desc, "description")
                        pd.DataFrame(
                            sim_desc,
                            index=valid_desc['title'],
                            columns=valid_desc['title']
                        ).to_excel(writer, sheet_name='Similarity_Descriptions')

                valid_transcripts = df[~df['transcript'].str.contains("not available|error|Failed", case=False, na=False)]
                if len(valid_transcripts) > 1:
                    sim_trans = analyze_similarity(valid_transcripts, "transcript")
                    pd.DataFrame(
                        sim_trans,
                        index=valid_transcripts['title'],
                        columns=valid_transcripts['title']
                    ).to_excel(writer, sheet_name='Similarity_Transcripts')

            
            st.download_button(
                label="📊 Download Excel Report",
                data=output.getvalue(),
                file_name=f"youtube_analysis_{query.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV download
            csv_output = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Data",
                data=csv_output,
                file_name=f"youtube_data_{query.replace(' ', '_')}.csv",
                mime="text/csv"
            )

            
        # ---- Run Insight Generation ----
        try:
            insight_gen = InsightGenerator(api_key="79fb00c0680e4137acd298b1466e8fdd")  # replace with your real key
            insight_text = insight_gen.generate_insight(sim_df_td, sim_df_tdt, df)

            # Show in Streamlit
            st.subheader("🤖 AI Insights on Video Similarity")
            st.markdown(insight_text)

            # Save to file
            with open("youtube_similarity_insights.txt", "w", encoding="utf-8") as f:
                f.write(insight_text)

        except Exception as e:
            st.error(f"Insight generation failed: {e}")


# -----------------------------
# Usage Instructions
# -----------------------------
if not run_button:
    st.markdown("""
    ## 🚀 How to Use This App
    
    1. **Setup**: Replace `YOUR_YOUTUBE_API_KEY_HERE` with your actual YouTube Data API v3 key
    2. **Search**: Enter a search query in the sidebar
    3. **Configure**: Choose number of results and whether to include transcripts
    4. **Analyze**: Click "Run Search" to get results and similarity analysis
    
    ### 🔧 Features
    - **Video Search**: Find YouTube videos based on your query
    - **Metadata**: Views, likes, comments, duration, subscriber count
    - **Transcripts**: Optional transcript fetching with rate limiting
    - **Similarity Analysis**: Compare videos using AI embeddings
    - **Analytics**: Performance metrics and duration analysis
    - **Export**: Download results as Excel or CSV
    
    ### ⚠️ Rate Limiting Notes
    - Transcript fetching includes delays to avoid API limits
    - Results are cached for 1 hour to reduce API calls
    - If you hit rate limits, try reducing the number of results or disable transcripts
    
    ### 🔑 API Key Setup
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project or select existing one
    3. Enable YouTube Data API v3
    4. Create credentials (API key)
    5. Replace the placeholder in the code with your key
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data from YouTube API v3")

