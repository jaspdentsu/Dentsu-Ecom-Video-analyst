import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import time
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
import streamlit as st
import cv2
import pytesseract
import tensorflow_hub as hub
import torch
from transformers import CLIPProcessor, CLIPModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pandas.io.formats.style import Styler
import urllib.request, json

# ==== OPTIONAL local OCR for logo/text on images ====
# Make sure Tesseract is installed & in PATH.
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_use_embedding(text):
    model = load_embedding_model()
    return model.encode(text, convert_to_tensor=False, show_progress_bar=False)

# later
#use_model = load_embedding_model()

# ==== Load models ====
model = SentenceTransformer('all-MiniLM-L6-v2')
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\jsingh11\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
#use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ==== Helpers ====
def get_soup_from_url(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup

def extract_top_15_product_links(soup):
    product_links = []
    seen = set()
    cards = soup.find_all('div', {'data-cy': 'title-recipe'})
    for card in cards:
        if card.find('div', {'class': 'a-row a-spacing-micro'}):
            continue
        link = card.find('a', class_=lambda x: x and 'a-link-normal' in x and 'a-text-normal' in x)
        if link and link['href'] and link['href'] not in seen:
            full_url = 'https://www.amazon.in' + link['href']
            product_links.append(full_url)
            seen.add(link['href'])
        if len(product_links) == 15:
            break
    return product_links

def extract_product_data(url):
    soup = get_soup_from_url(url)
    try:
        title = soup.find('span', {'class': 'a-size-large product-title-word-break'}).text.strip()
    except:
        title = ''
    try:
        desc_block = soup.find('ul', {'class': 'a-unordered-list a-vertical a-spacing-mini'})
        descriptions = [li.text.strip() for li in desc_block.find_all('span', {'class': 'a-list-item'})]
    except:
        descriptions = []

    img_urls = []
    try:
        scripts = soup.find_all('script')
        image_data_script = next((s for s in scripts if 'colorImages' in s.text and 'hiRes' in s.text), None)
        if image_data_script:
            matches = re.findall(r'\"hiRes\":\"(https:[^\"]+?\.jpg)\"', image_data_script.text)
            img_urls = list(set(matches))
    except:
        pass

    # --- NEW: extract 5, 4, 3 star review links ---
    review_links = {"5": "", "4": "", "3": ""}
    try:
        hist_ul = soup.find("ul", id="histogramTable")
        if hist_ul:
            a_tags = hist_ul.find_all("a", href=True)
            for a in a_tags:
                href = a["href"]
                if "filterByStar=five_star" in href and review_links["5"] == "":
                    review_links["5"] = "https://www.amazon.in" + href
                elif "filterByStar=four_star" in href and review_links["4"] == "":
                    review_links["4"] = "https://www.amazon.in" + href
                elif "filterByStar=three_star" in href and review_links["3"] == "":
                    review_links["3"] = "https://www.amazon.in" + href
    except Exception as e:
        print(f"Review link extraction error: {e}")

    return title, descriptions, img_urls, review_links

def get_image_tags(img: Image.Image, ocr_text: str):
    img_cv = np.array(img)
    if img_cv.shape[-1] == 4:
        img_cv = img_cv[..., :3]
    tags = []
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    white_ratio = np.sum(gray > 230) / gray.size
    if white_ratio > 0.6:
        tags.append('white_background')
    if len(ocr_text.strip().split()) > 10:
        tags.append('text_heavy')
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            tags.append('face_present')
    except:
        pass
    return tags

def download_image_text_embed(img_urls):
    image_texts, embeddings, visual_tags = [], [], []
    for url in img_urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            text = pytesseract.image_to_string(img, config='--psm 6').strip()
            tags = get_image_tags(img, text)
            image_texts.append(text)
            visual_tags.append(tags)
            clip_input = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model.get_image_features(**clip_input).numpy().flatten()
            embeddings.append(emb)
        except:
            image_texts.append("")
            visual_tags.append([])
    return image_texts, np.array(embeddings), visual_tags

#def compute_use_embedding(text):
#    return use_model([text])[0].numpy()

def detect_brand(text):
    known = ["Philips", "SYSKA", "VEGA", "Mi Xiaomi", "Havells", "Panasonic", "Braun", "Bombay Shaving Company", 
    "Nova", "Morphy Richards", "Beardo", "VGR", "Concepta", "Agaro", "Noymi", "Zlade", "Lifelong", 
    "Pigeon", "Sellbotic", "Painless", "Ardaki", "DMFS", "Uzox", "Winston", "HP", "Dell", "Lenovo", "Asus", 
    "Acer", "MSI", "Apple", "Samsung", "Microsoft", "LG", "iBall", "Avita", "OnePlus", "Realme", "Vivo", 
    "Oppo", "Motorola", "Nokia", "Honor","Nothing", "Google Pixel", "Infinix", "Tecno", "Lava", 
    "Micromax", "iQOO", "Redmi", "POCO"]
    for k in known:
        if k.lower() in text.lower():
            return k
    match = re.findall(r'\b([A-Z][a-zA-Z]+)\b', text)
    return match[0] if match else 'Unknown'

def detect_tone_scores(text):
    """Return raw scores for each tone category."""
    text = text.lower()
    categories = {
        'emotional': [
            'love','care','gentle','comfort','soft','happy','peace','confidence','soothing','touch','smooth',
            'relief','fresh','hygienic','calm','trust','family','safe','secure','joy','satisfaction',
            'cozy','reliable','wellbeing','warm','ease','relax','healthy','pure','refreshing'
        ],

        'functional': [
            'cordless','battery','runtime','attachments','durable','settings','head','waterproof','rechargeable',
            'stainless','blade','cutting','precision','trim','groom','multigroom','utility','sensor',
            'wash cycle','spin speed','drum','detergent','filter','storage','capacity','door','hinge',
            'compartment','adjustable','temperature','auto clean'
        ],

        'performance': [
            'fast','performance','speed','efficient','powerful','octa-core','processor','snapdragon',
            'mediatek','ram','ssd','hdd','boot','refresh rate','turbo','cooling',
            'spin','wash power','drying speed','frost free','airflow','compressor','wattage','rpm',
            'frame rate','lag free','optimized','high capacity','endurance','throughput','benchmark'
        ],

        'premium': [
            'luxury','premium','exclusive','elegant','crafted','refined','high-end','signature','designer',
            'sleek','metallic','aluminium','bezel-less','platinum','elite',
            'glass finish','chrome','aura','heritage','limited edition','artisanal','posh','sophisticated',
            'royal','plush','ornate','top notch','flagship','genuine leather','prestige'
        ],

        'financial': [
            'budget','value','worth','affordable','cheap','deal','discount','save','low price','emi','offer',
            'cashback','cost effective','economical','promotion','exchange offer','sale','rebate',
            'bargain','festive price','bundle','combo','voucher','coupon','clearance','pay later',
            'zero cost emi','low emi','finance','installment'
        ],

        'innovation': [
            'new','latest','next-gen','ai','ai-powered','smart','machine learning','innovation','cutting-edge',
            'tech','intelligent','revolutionary','5g','wi-fi 6','fingerprint','face unlock',
            'iot','voice control','wireless','automation','self clean','adaptive','hybrid','dual inverter',
            'eco bubble','direct drive','quantum','oled','nanocell','holographic','gesture control'
        ],

        'aesthetic': [
            'design','modern','compact','slim','stylish','minimal','bezel-less','matte','finish','appearance',
            'look','feel','color options','vibrant','polished','curved','sleek','trendy','portable','foldable',
            'premium finish','symmetric','aura design','glossy','contoured','streamlined','patterned',
            'decor','elegant','fashionable','smooth edges'
        ],

        'gaming': [
            'gaming','fps','graphics','gpu','nvidia','geforce','rtx','gaming mode','rgb','cooling','fan',
            'overclocked','gaming performance','frame rate','g-sync','ray tracing','vram','refresh','hz',
            'lag free','low latency','multiplayer','headset','controller','esports','tournament','arcade',
            'gamepad','rpg','battle royale','open world'
        ],

        'masculine': [
            'men','rugged','masculine','bold','beard','power look','stubble',
            'strength','sturdy','heavy duty','tough','solid','ironclad','durable','resilient','robust',
            'beard styling','trimmer','grit','raw','strong','assertive','dominant','steel','rough','dark',
            'brawny','outdoors','warrior','hardcore','enduring'
        ],

        'feminine': [
            'women','feminine','elegant','beauty','soft skin','gentle care','face hair','eyebrow','sensitive',
            'smooth finish','refined','delicate','stylish','slim','graceful','lightweight','pastel','pretty',
            'makeup','grooming','self care','tender','glamour','shine','fragrance','curves','touch up',
            'silky','aesthetic','appeal','ornamental','gentleness'
        ],
    }

    scores = {cat: 0 for cat in categories}
    for cat, words in categories.items():
        for w in words:
            if w in text:
                scores[cat] += 1
    return scores

def get_top_keywords(corpus, top_n=15):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(corpus)
    top_words = []
    for row in tfidf_matrix:
        scores = row.toarray().flatten()
        indices = scores.argsort()[-top_n:][::-1]
        words = [tfidf.get_feature_names_out()[i] for i in indices if scores[i] > 0]
        top_words.append(words)
    return top_words

def rgb_to_hex_tuple(rgb):
    """rgb as (R,G,B) 0..255 -> '#RRGGBB'."""
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def closest_color_name(hex_code: str) -> str:
    """
    Map hex to nearest CSS4 color name using Euclidean distance in RGB space.
    No webcolors needed.
    """
    try:
        base_rgb = np.array(mcolors.to_rgb(hex_code)) * 255  # floats 0..1 -> 0..255
    except ValueError:
        return "Unknown"

    closest, best = None, float("inf")
    for name, hex_val in mcolors.CSS4_COLORS.items():  # name -> '#RRGGBB'
        ref_rgb = np.array(mcolors.to_rgb(hex_val)) * 255
        dist = np.sum((base_rgb - ref_rgb) ** 2)
        if dist < best:
            best, closest = dist, name
    return closest.capitalize() if closest else "Unknown"

def extract_top_colors_from_url(img_url, k=5, max_pixels=200_000):
    """
    Extract k dominant colors from an image URL.
    - Downsamples large images for speed.
    - Returns list of hex strings.
    """
    try:
        resp = requests.get(img_url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        arr = np.array(img)
        # Optional downsample to keep KMeans fast:
        h, w = arr.shape[:2]
        if h * w > max_pixels:
            scale = (max_pixels / float(h * w)) ** 0.5
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            arr = np.array(img)

        flat = arr.reshape(-1, 3)
        # KMeans
        clt = KMeans(n_clusters=k, n_init=5, random_state=42)
        clt.fit(flat)
        centers = clt.cluster_centers_
        hex_colors = [rgb_to_hex_tuple(c) for c in centers]
        return hex_colors
    except Exception:
        return []

def detect_logo(image_paths: list, brand_lexicon: list[str]):
    if not TESS_AVAILABLE or not image_paths:
        return []

    found = set()
    for p in image_paths[:5]:  # limit OCR calls per brand for speed
        if not os.path.exists(p):
            continue
        try:
            img = Image.open(p).convert("RGB")
            # Slight denoise / sharpen helps OCR a bit
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv_img = cv2.bilateralFilter(cv_img, 7, 50, 50)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            # adaptive threshold
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 31, 10)
            ocr_text = pytesseract.image_to_string(th, lang="eng")
            text_norm = re.sub(r"[^A-Za-z0-9 ]+", " ", ocr_text).lower()
            for b in brand_lexicon:
                b_norm = b.lower()
                if len(b_norm) >= 3 and b_norm in text_norm:
                    found.add(b)
        except Exception:
            continue
    return sorted(found)

def center_align_except(df, exclude_cols):
    """Center-align all dataframe columns except exclude_cols."""
    return (
        df.style.set_properties(**{'text-align': 'center'})
          .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
          .set_properties(subset=exclude_cols, **{'text-align': 'left'})
    )

def add_slno_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'Sl No' column exists as the first column (1..n)."""
    if 'Sl No' not in df.columns:
        df = df.reset_index(drop=True).copy()
        df.insert(0, 'Sl No', np.arange(1, len(df) + 1))
    return df

def style_center_except_first_two(df: pd.DataFrame):
    """Left-align first 2 cols (Sl No, Brand), center-align all others, format to 2 decimals."""
    cols = list(df.columns)
    left_cols = cols[:2]
    center_cols = cols[2:]

    styler = (df.style
                .format(precision=2)  # force 2 decimal places
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]))

    if left_cols:
        styler = styler.set_properties(subset=left_cols, **{'text-align': 'left'})
    if center_cols:
        styler = styler.set_properties(subset=center_cols, **{'text-align': 'center'})

    return styler


def detect_logo_and_size(img: Image.Image, brand_lexicon: list[str]):
    """
    Detect brand logos in an image via OCR.
    Returns (has_logo: bool, logo_area_ratio: float).
    """
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    found, max_ratio = False, 0.0

    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        for b in brand_lexicon:
            if len(b) < 3:
                continue
            if b.lower() in word.lower():
                found = True
                (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                ratio = (w * h) / float(cv_img.shape[0] * cv_img.shape[1])
                max_ratio = max(max_ratio, ratio)
    return found, max_ratio

def run_brand_similarity(st, search_url, num_products=15, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    soup = get_soup_from_url(search_url)
    product_links = extract_top_15_product_links(soup)

    records = []
    brand_texts = defaultdict(list)
    brand_image_embeds = defaultdict(list)
    brand_text_embeds = {}
    brand_visual_tags = defaultdict(list)

    for url in product_links[:num_products]:
        title, descriptions, img_urls, review_links = extract_product_data(url)
        brand = detect_brand(title)
        image_texts, image_embeds, image_tags = download_image_text_embed(img_urls)
        text_block = title + " " + " ".join(descriptions + image_texts)
        brand_texts[brand].append(text_block)
        brand_visual_tags[brand].extend(image_tags)
        if len(image_embeds) > 0:
            brand_image_embeds[brand].append(np.mean(image_embeds, axis=0))
        records.append({
            'Brand': brand,
            'Title': title,
            'Descriptions': descriptions,
            'ImageURLs': img_urls,
            'ImageTexts': image_texts,
            'Text': text_block,
            'review_5_link': review_links["5"],
            'review_4_link': review_links["4"],
            'review_3_link': review_links["3"]
        })

    brand_names = list(brand_texts.keys())
    for b in brand_names:
        combined_text = " ".join(brand_texts[b])
        brand_text_embeds[b] = compute_use_embedding(combined_text)

    text_vecs = np.array([brand_text_embeds[b] for b in brand_names])
    image_vecs = np.array([
        np.mean(brand_image_embeds[b], axis=0) if brand_image_embeds[b] else np.zeros(512)
        for b in brand_names
    ])
    text_sim = cosine_similarity(text_vecs)
    image_sim = cosine_similarity(image_vecs)

    clusters = KMeans(n_clusters=min(5, len(brand_names)), random_state=42).fit_predict(text_vecs)
    summary_df = pd.DataFrame({
        'Brand': brand_names,
        'Cluster': clusters,
        'Top_Keywords': get_top_keywords([" ".join(brand_texts[b]) for b in brand_names])
    })

    # Save files
    pd.DataFrame(records).to_excel(os.path.join(out_dir, "product_data.xlsx"), index=False)
    summary_df.to_excel(os.path.join(out_dir, "brand_summary.xlsx"), index=False)

    # Heatmaps
    st.subheader("Brand Similarity Heatmaps")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(text_sim, xticklabels=brand_names, yticklabels=brand_names, annot=True, cmap="YlGnBu", ax=ax1)
        plt.title("Brand Textual Similarity", fontsize=14)
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(image_sim, xticklabels=brand_names, yticklabels=brand_names, annot=True, cmap="YlOrRd", ax=ax2)
        plt.title("Visual Brand Similarity", fontsize=14)
        st.pyplot(fig2)

    # Explore Similarity vs Selected Brand
    st.subheader("Explore Similarity vs Selected Brand")
    selected_brand = st.selectbox("Select a Brand", brand_names)
    sel_idx = brand_names.index(selected_brand)
    sim_scores = pd.DataFrame({
        'Brand': brand_names,
        'Text_Similarity': text_sim[sel_idx],
        'Image_Similarity': image_sim[sel_idx]
    }).sort_values("Text_Similarity", ascending=False)
    sim_scores = add_slno_if_missing(sim_scores)
    st.table(style_center_except_first_two(sim_scores))

    # Visual Style Summary
    visual_summary = []
    for b in brand_names:
        flat_tags = [tag for sublist in brand_visual_tags[b] for tag in sublist]
        count = Counter(flat_tags)
        visual_summary.append({
            'Brand': b,
            'White_BG': count.get('white_background', 0),
            'Face': count.get('face_present', 0),
            'Text_Heavy': count.get('text_heavy', 0),
            'Total_Images': len(brand_visual_tags[b])
        })

    visual_df = pd.DataFrame(visual_summary)
    visual_df['%White_BG'] = (visual_df['White_BG'] / visual_df['Total_Images'] * 100).round(1)
    visual_df['%Face'] = (visual_df['Face'] / visual_df['Total_Images'] * 100).round(1)
    visual_df['%Text_Heavy'] = (visual_df['Text_Heavy'] / visual_df['Total_Images'] * 100).round(1)
    vis_table = visual_df[['Brand', '%White_BG', '%Face', '%Text_Heavy']]
    vis_table = add_slno_if_missing(vis_table)

    st.subheader("Visual Style Tag Summary")
    st.table(style_center_except_first_two(vis_table))

    # Tone Category Distribution (%)
    tone_distributions = []
    for b in brand_names:
        combined_text = " ".join(brand_texts[b])
        scores = detect_tone_scores(combined_text)
        total = sum(scores.values())
        if total == 0:
            perc = {k: 0 for k in scores}
        else:
            perc = {k: round(v/total*100, 1) for k,v in scores.items()}
        perc['Brand'] = b
        tone_distributions.append(perc)

    tone_df = pd.DataFrame(tone_distributions)
    tone_df = add_slno_if_missing(tone_df)

    # Move Brand next to Sl No
    cols = ['Sl No', 'Brand'] + [c for c in tone_df.columns if c not in ['Sl No','Brand']]
    tone_df = tone_df[cols]

    st.subheader("Tone Category Distribution (%) Across Brands")
    st.table(style_center_except_first_two(tone_df))

    # ---- Color Analysis: Top 10 per brand (names in UI; hex saved) ----
    brand_colors = []
    for b in brand_names:
        all_hex = []
        # Collect colors from all product images of this brand
        for rec in records:
            if rec['Brand'] == b:
                for img_url in rec['ImageURLs']:
                    # Grab 5 per image; across many images this yields a good pool
                    all_hex.extend(extract_top_colors_from_url(img_url, k=5))

        if all_hex:
            counts = Counter(all_hex)
            # Top 10 hex codes overall
            top10_hex = [c for c, _ in counts.most_common(10)]
            top10_names = [closest_color_name(c) for c in top10_hex]
            dominant_hex = top10_hex[0]
            dominant_name = top10_names[0]
        else:
            top10_hex, top10_names = [], []
            dominant_hex, dominant_name = "", ""

        brand_colors.append({
            'Brand': b,
            # For saving (complete with hex)
            'Top_10_Colors_Hex': ", ".join(top10_hex),
            'Dominant_Color_Hex': dominant_hex,
            # For display (names only)
            'Top_10_Colors': ", ".join(top10_names),
            'Dominant_Color': dominant_name,
        })

    color_df = pd.DataFrame(brand_colors)
    # Save full hex info to a separate file
    color_save = color_df[['Brand', 'Top_10_Colors_Hex', 'Dominant_Color_Hex']].copy()
    color_save = color_save.reset_index(drop=True)
    color_save.to_excel(os.path.join(out_dir, "brand_colors.xlsx"), index=False)

    # Show names only in Streamlit, with Sl No + Brand left, rest centered
    color_show = color_df[['Brand', 'Top_10_Colors', 'Dominant_Color']].copy()
    color_show = add_slno_if_missing(color_show)
    st.subheader("Top Colors Detected Across Brands (Names)")
    st.table(style_center_except_first_two(color_show))

    # ---- Logo Analysis ----
    brand_logos = []
    brand_logo_ratios = []

    for b in brand_names:
        logo_count, total_imgs, ratios = 0, 0, []
        for rec in records:
            if rec['Brand'] == b:
                for img_url in rec['ImageURLs']:
                    try:
                        resp = requests.get(img_url, timeout=10)
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        has_logo, ratio = detect_logo_and_size(img, [b])
                        total_imgs += 1
                        if has_logo:
                            logo_count += 1
                            ratios.append(ratio * 100)  # percentage
                    except:
                        continue

        perc_logo = round((logo_count / total_imgs) * 100, 2) if total_imgs > 0 else 0.0
        avg_ratio = round(np.mean(ratios), 2) if ratios else 0.0

        brand_logos.append({'Brand': b, 'Total_Images': total_imgs, '%With_Logo': perc_logo})
        brand_logo_ratios.append({'Brand': b, 'Avg_Logo_Size_%': avg_ratio})

    # Convert to DataFrames
    logo_df = pd.DataFrame(brand_logos)
    logo_df = add_slno_if_missing(logo_df)

    logo_ratio_df = pd.DataFrame(brand_logo_ratios)
    logo_ratio_df = add_slno_if_missing(logo_ratio_df)

    # Save to Excel
    logo_save = logo_df.merge(logo_ratio_df, on="Brand")
    logo_save.to_excel(os.path.join(out_dir, "brand_logos.xlsx"), index=False)

    # Show in Streamlit
    st.subheader("Logo Presence in Brand Images (%)")
    st.table(style_center_except_first_two(logo_df))

    st.subheader("Average Logo Size Ratio (%)")
    st.table(style_center_except_first_two(logo_ratio_df))

    # Most similar brand
    text_sim_idx = np.argsort(-text_sim[sel_idx])[1]
    image_sim_idx = np.argsort(-image_sim[sel_idx])[1]
    st.markdown(f"**Textually most similar brand:** {brand_names[text_sim_idx]}")
    st.markdown(f"**Visually most similar brand:** {brand_names[image_sim_idx]}")

    class InsightGenerator:
        def __init__(self, api_key="Your_Key", deployment="grok-3"):
            self.url = "https://ai-api-dev.dentsu.com/foundry/chat/completions?api-version=2024-05-01-preview"
            self.deployment = deployment
            self.headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "x-service-line": "media",
                "x-brand": "dentsu",
                "x-project": "EDA Smart Dashboard 2",
                "api-version": "v1",
                "Ocp-Apim-Subscription-Key": api_key
            }

        def generate_insight(self, tone_df, color_df, logo_df, logo_ratio_df):
            # Build structured prompt
            summary_prompt = f"""
            You are analyzing brand similarity across competitors.
            Below is the extracted data:

            Tone distribution by brand:
            {tone_df.to_string(index=False)}

            Color usage:
            {color_df[['Brand','Top_10_Colors','Dominant_Color']].to_string(index=False)}

            Logo presence:
            {logo_df.to_string(index=False)}

            Logo size ratios:
            {logo_ratio_df.to_string(index=False)}

            Based on this, explain in concise:
            1. Which brands appear most similar to each other.
            2. What factors (tone, color, logo usage, etc.) make them credible to be called similar.
            3. Where they differ the most.
            4. Provide actionable insight in 3 to 4 bullet points.
            """

            system_prompt = """You are an expert marketing analyst.
            Your task is to interpret brand similarity data and provide clear, concise professional insights. make sure the complete insight is within 500 words/tokens"""

            # Request body
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
                return f"🤖 Insight generation failed: {e}"


    # ---- Run Insight Generation ----
    try:
        insight_gen = InsightGenerator(api_key="79fb00c0680e4137acd298b1466e8fdd")  # replace with your real key
        insight_text = insight_gen.generate_insight(tone_df, color_df, logo_df, logo_ratio_df)

        # Show in Streamlit
        st.subheader("🤖 AI Insights on Brand Similarity")
        st.markdown(insight_text)

        # Save to file
        with open(os.path.join(out_dir, "brand_similarity_insights.txt"), "w", encoding="utf-8") as f:
            f.write(insight_text)

    except Exception as e:
        st.error(f"Insight generation failed: {e}")



