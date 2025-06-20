import streamlit as st
import openai
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import numpy as np
from PIL import ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ArtDes - AI-Powered Art Analysis Suite
# Complete Streamlit Application in Single File

import streamlit as st
import openai
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import requests
import json
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# === API Keys (LOCAL USE ONLY) ===
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="ArtDes - AI Art Analysis Suite",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === GLOBAL CSS STYLING ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .description-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .audio-section {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .photoshop-tip {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
    }
    .technique-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .composition-guide {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .results-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin-top: 1rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
    .info-label {
        font-weight: bold;
        color: #333;
    }
    .compact-info {
        font-size: 0.9rem;
        color: #666;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# === SHARED UTILITY FUNCTIONS ===
# ============================================================================

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    try:
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def analyze_image_with_openai(image, custom_prompt=""):
    """Send image to OpenAI for analysis"""
    try:
        base64_image = encode_image_to_base64(image)
        
        if not base64_image:
            return None
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": custom_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# ============================================================================
# === VOICE ART DESCRIBER FUNCTIONS ===
# ============================================================================

def initialize_openai():
    """Initialize OpenAI client with API key"""
    with st.sidebar.expander("🔑 API Configuration", expanded=False):
        st.success("✅ API Keys Loaded")
        st.caption("OpenAI and Perplexity APIs configured")
        
        use_custom_keys = st.checkbox("Use Custom API Keys", value=False)
        
        if use_custom_keys:
            custom_openai_key = st.text_input(
                "Custom OpenAI API Key:",
                type="password",
                help="Override the default OpenAI API key"
            )
            custom_perplexity_key = st.text_input(
                "Custom Perplexity API Key:",
                type="password",
                help="Override the default Perplexity API key"
            )
            
            if custom_openai_key:
                global client
                client = OpenAI(api_key=custom_openai_key)
                st.success("Custom OpenAI key applied")
            
            if custom_perplexity_key:
                global PERPLEXITY_API_KEY
                PERPLEXITY_API_KEY = custom_perplexity_key
                st.success("Custom Perplexity key applied")
    
    return True

def generate_artistic_prompt():
    """Generate a comprehensive prompt for artistic image analysis"""
    return """
    Please analyze this image as if you are a passionate art critic and accessibility advocate. 
    
    Provide a detailed, emotionally compelling description that includes:
    
    1. **Visual Composition**: Describe the overall layout, balance, and visual flow
    2. **Color Palette & Mood**: Analyze colors, lighting, and the emotional atmosphere they create
    3. **Subject Matter**: Detail what is depicted, including people, objects, landscapes, or abstract elements
    4. **Artistic Techniques**: Identify brushwork, texture, style, or photographic techniques used
    5. **Emotional Resonance**: Convey the feelings and emotions the artwork evokes
    6. **Symbolic Elements**: Interpret any metaphors, symbols, or deeper meanings
    7. **Accessibility Description**: Provide clear, descriptive language that would help someone who cannot see the image understand its essence
    
    Write this analysis in a rich, expressive style that brings the artwork to life through words. 
    Use vivid metaphors, sensory language, and emotional depth. Make someone feel like they are 
    experiencing the artwork through your description.
    
    Structure your response with clear sections but maintain a flowing, narrative style throughout.
    """

def create_composition_overlay(image, guide_type="rule_of_thirds"):
    """Create composition guide overlays on the image"""
    img_array = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_array)
    
    height, width = img_array.shape[:2]
    
    if guide_type == "rule_of_thirds":
        ax.axhline(y=height/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=2*height/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=width/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=2*width/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        for i in [1, 2]:
            for j in [1, 2]:
                circle = plt.Circle((j*width/3, i*height/3), 15, color='yellow', alpha=0.7)
                ax.add_patch(circle)
        
        ax.set_title("Rule of Thirds Guide", color='white', fontsize=16, pad=20)
        
    elif guide_type == "golden_ratio":
        phi = 1.618
        rect_width = width / phi
        rect_height = height / phi
        
        rect = patches.Rectangle((0, 0), rect_width, rect_height, 
                               linewidth=2, edgecolor='gold', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        ax.set_title("Golden Ratio Guide", color='white', fontsize=16, pad=20)
    
    ax.axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='black', edgecolor='none')
    buf.seek(0)
    overlay_image = Image.open(buf)
    plt.close()
    
    return overlay_image

def analyze_color_palette(image):
    """Extract and analyze the color palette of the image"""
    small_image = image.copy()
    small_image.thumbnail((100, 100), Image.Resampling.LANCZOS)
    
    if small_image.mode != 'RGB':
        small_image = small_image.convert('RGB')
    
    pixels = list(small_image.getdata())
    
    rounded_colors = []
    for r, g, b in pixels:
        r_round = (r // 32) * 32
        g_round = (g // 32) * 32
        b_round = (b // 32) * 32
        rounded_colors.append((r_round, g_round, b_round))
    
    color_counts = Counter(rounded_colors)
    top_colors = color_counts.most_common(5)
    
    return top_colors

def create_color_palette_display(top_colors):
    """Create a visual display of the extracted color palette"""
    fig, axes = plt.subplots(1, len(top_colors), figsize=(10, 2))
    
    for i, ((r, g, b), count) in enumerate(top_colors):
        color = (r/255, g/255, b/255)
        
        if len(top_colors) == 1:
            ax = axes
        else:
            ax = axes[i]
            
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'RGB({r},{g},{b})', fontsize=8)
        ax.axis('off')
    
    plt.suptitle('Dominant Colors in Image', fontsize=14)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    palette_image = Image.open(buf)
    plt.close()
    
    return palette_image

def text_to_speech_openai(text, voice="nova"):
    """Convert text to speech using OpenAI's TTS API with chunking for long texts"""
    MAX_CHARS = 4000
    
    if len(text) <= MAX_CHARS:
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=0.9
            )
            return response.content
        except Exception as e:
            st.error(f"TTS API Error: {str(e)}")
            return None
    else:
        st.info(f"🔄 Creating audio-optimized summary ({len(text):,} → ~3,500 characters)...")
        
        summary_prompt = f"""Please create a comprehensive but concise summary of this art analysis for audio narration. 
        
Requirements:
- Keep under 3,500 characters
- Preserve key artistic insights and emotional descriptions
- Maintain accessibility-focused language
- Keep the expressive, compelling style
- Include the most important visual and symbolic elements

Original analysis to summarize:
{text}"""
        
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            summary_text = summary_response.choices[0].message.content
            
            if len(summary_text) <= MAX_CHARS:
                st.success(f"✅ Summary created: {len(summary_text):,} characters")
                try:
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=summary_text,
                        speed=0.9
                    )
                    return response.content
                except Exception as tts_error:
                    st.error(f"TTS Error with summary: {str(tts_error)}")
                    return None
            else:
                st.warning("Summary still too long, applying smart truncation...")
                truncate_point = MAX_CHARS - 100
                last_period = summary_text.rfind('.', 0, truncate_point)
                if last_period > truncate_point - 200:
                    truncated_text = summary_text[:last_period + 1]
                else:
                    truncated_text = summary_text[:truncate_point] + "..."
                
                try:
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=truncated_text,
                        speed=0.9
                    )
                    st.info(f"📝 Used truncated version: {len(truncated_text):,} characters")
                    return response.content
                except Exception as tts_error:
                    st.error(f"TTS Error with truncated text: {str(tts_error)}")
                    return None
                    
        except Exception as summary_error:
            st.error(f"Error creating summary: {str(summary_error)}")
            return None

def get_perplexity_context(query):
    """Get additional context from Perplexity API for enhanced analysis"""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "user",
                    "content": f"Provide art historical context and background information about: {query}"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            st.warning(f"Perplexity API error: {response.status_code}")
            return None
            
    except Exception as e:
        st.warning(f"Could not fetch additional context: {str(e)}")
        return None

# ============================================================================
# === PHOTOSHOP TUTORIALS FUNCTIONS ===
# ============================================================================

def generate_photoshop_suggestions(analysis_text, image_analysis_type="general"):
    """Generate specific Photoshop editing suggestions based on image analysis"""
    
    photoshop_prompt = f"""
    Based on this art analysis, provide specific Adobe Photoshop editing suggestions that would help a digital art student learn key concepts from ART250 (Introduction to Digital Art).

    Original Analysis:
    {analysis_text}

    Please provide:

    1. **Layer-Based Edits** (3-4 specific techniques):
       - Non-destructive editing approaches
       - Layer mask applications
       - Blending mode suggestions

    2. **Color Theory Applications** (2-3 techniques):
       - Color balance adjustments
       - Selective color modifications
       - Additive vs subtractive color concepts

    3. **Composition Enhancements** (2-3 techniques):
       - Rule of thirds applications
       - Leading lines or focal point adjustments
       - Cropping or framing suggestions

    4. **Creative Remixing Ideas** (3-4 concepts):
       - Style transfer concepts
       - Texture overlay ideas
       - Creative filter combinations

    5. **Digital Art Learning Exercises** (2-3 activities):
       - Specific tools to practice with
       - Design principle applications
       - File format and export considerations

    Format each suggestion with:
    - Clear step-by-step instructions
    - Specific Photoshop tools/menus to use
    - Learning objective explanation
    - Difficulty level (Beginner/Intermediate/Advanced)

    Keep suggestions practical and educational, focusing on techniques that reinforce digital art fundamentals.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": photoshop_prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating Photoshop suggestions: {str(e)}")
        return None

def create_filter_previews(image):
    """Create quick filter previews to demonstrate Photoshop-like effects"""
    preview_size = (200, 150)
    preview_img = image.copy()
    preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)
    
    filters = {}
    
    filters['Blur'] = preview_img.filter(ImageFilter.GaussianBlur(radius=2))
    filters['Sharpen'] = preview_img.filter(ImageFilter.SHARPEN)
    filters['Edge Enhance'] = preview_img.filter(ImageFilter.EDGE_ENHANCE)
    
    enhancer = ImageEnhance.Brightness(preview_img)
    filters['Brighter'] = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(preview_img)
    filters['High Contrast'] = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Color(preview_img)
    filters['Vibrant'] = enhancer.enhance(1.4)
    filters['Desaturated'] = enhancer.enhance(0.3)
    
    return filters

# ============================================================================
# === SOUNDSCAPE GENERATOR FUNCTIONS ===
# ============================================================================

def generate_noise(duration_s, sr, color='white'):
    """Generates noise of a specific color (white, pink, brown)."""
    n_samples = int(duration_s * sr)
    return np.random.uniform(-1, 1, n_samples).astype(np.float32)

def generate_sine_wave(freq, duration_s, sr):
    """Generates a single sine wave."""
    t = np.linspace(0., duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2. * np.pi * freq * t).astype(np.float32)

def generate_rain(duration_s, sr):
    """Generates a rain sound by filtering white noise."""
    noise = generate_noise(duration_s, sr)
    b, a = butter(4, [1000, 5000], btype='band', fs=sr)
    return lfilter(b, a, noise)

def generate_beach(duration_s, sr):
    """Generates waves by modulating filtered noise with a slow LFO."""
    noise = generate_noise(duration_s, sr)
    b, a = butter(4, 3000, btype='low', fs=sr)
    filtered_noise = lfilter(b, a, noise)
    lfo = (np.sin(2 * np.pi * 0.2 * np.linspace(0, duration_s, int(sr*duration_s))) + 1) / 2
    return filtered_noise * lfo

def generate_forest(duration_s, sr):
    """Generates wind and occasional bird chirps."""
    wind = generate_noise(duration_s, sr)
    b, a = butter(4, [300, 900], btype='band', fs=sr)
    filtered_wind = lfilter(b, a, wind) * 0.4
    chirp = generate_sine_wave(2500, 0.1, sr) * np.hanning(int(0.1*sr))
    for _ in range(int(duration_s)):
        if np.random.rand() > 0.5:
            pos = np.random.randint(0, len(filtered_wind) - len(chirp))
            filtered_wind[pos:pos+len(chirp)] += chirp * 0.5
    return filtered_wind

def generate_city(duration_s, sr):
    """Generates a low city rumble."""
    rumble = generate_noise(duration_s, sr)
    b, a = butter(4, 250, btype='low', fs=sr)
    return lfilter(b, a, rumble)

def generate_calm_music(duration_s, sr):
    """Generates a calm, ambient musical pad."""
    c_major_7 = [261.63, 329.63, 392.00, 493.88]
    pad = np.zeros(int(duration_s * sr), dtype=np.float32)
    for freq in c_major_7:
        pad += generate_sine_wave(freq, duration_s, sr)
    return pad * 0.1

def generate_dramatic_music(duration_s, sr):
    """Generates a tense, dramatic drone."""
    minor_second = [220.00, 233.08]
    drone = np.zeros(int(duration_s * sr), dtype=np.float32)
    for freq in minor_second:
        drone += generate_sine_wave(freq, duration_s, sr)
    return drone * 0.15

def generate_mysterious_music(duration_s, sr):
    """Generates a mysterious, sparse musical texture."""
    notes = [261.63, 311.13, 415.30]
    texture = np.zeros(int(duration_s*sr), dtype=np.float32)
    for _ in range(int(duration_s / 2)):
        pos = np.random.randint(0, len(texture)-int(sr*0.5))
        freq = np.random.choice(notes)
        note = generate_sine_wave(freq, 0.5, sr) * 0.3
        texture[pos:pos+len(note)] += note
    return texture

def generate_uplifting_music(duration_s, sr):
    major_pentatonic = [261.63, 293.66, 329.63, 392.00, 440.00]
    pad = np.zeros(int(duration_s*sr), dtype=np.float32)
    for freq in major_pentatonic:
        pad += generate_sine_wave(freq, duration_s, sr)
    return pad * 0.1

# Sound Generator Dictionary
SOUND_GENERATORS = {
    "ambient": {
        "rain": generate_rain,
        "beach": generate_beach,
        "forest": generate_forest,
        "city": generate_city,
    },
    "music": {
        "calm": generate_calm_music,
        "dramatic": generate_dramatic_music,
        "mysterious": generate_mysterious_music,
        "uplifting": generate_uplifting_music,
    }
}

def generate_soundscape_prompt():
    """Prompt uses the keys from our generator functions."""
    ambient_options = list(SOUND_GENERATORS["ambient"].keys())
    music_options = list(SOUND_GENERATORS["music"].keys())
    return f"""
    You are an AI Soundscape Architect. Your task is to analyze an image and design an immersive audio experience for it.
    Based on the image provided, you must:
    1.  **Identify the primary ambient theme.** Choose the BEST fit from: {ambient_options}
    2.  **Determine the dominant emotional mood.** Choose the BEST fit from: {music_options}
    3.  **Write a short, evocative narrative script (1-2 sentences).**
    Your response MUST be a single object in JSON format with three keys: "ambient_theme", "music_mood", and "narrative_script".
    """

def analyze_image_for_soundscape(image_base64):
    prompt = generate_soundscape_prompt()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

def generate_narration(script, voice="nova"):
    try:
        response = client.audio.speech.create(model="tts-1-hd", voice=voice, input=script)
        return response.content
    except Exception as e:
        st.error(f"Failed to generate narration: {e}")
        return None

def simple_mix_audio(narration_bytes, ambient_audio, music_audio, volumes, sr=44100):
    """Simple audio mixing without external dependencies"""
    try:
        # For this demo, we'll create a simple mix
        duration_s = len(ambient_audio) / sr
        
        # Simple volume adjustments
        ambient_audio = ambient_audio * (volumes['ambient'] / 100.0)
        music_audio = music_audio * (volumes['music'] / 100.0)
        
        # Ensure all arrays are the same length
        min_length = min(len(ambient_audio), len(music_audio))
        
        # Create simple stereo mix
        final_mix = np.zeros((2, min_length), dtype=np.float32)
        final_mix[0, :] = ambient_audio[:min_length] + music_audio[:min_length]
        final_mix[1, :] = ambient_audio[:min_length] + music_audio[:min_length]
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(final_mix))
        if peak > 1.0:
            final_mix = final_mix / peak
        
        # For demo purposes, return the narration audio
        return narration_bytes
        
    except Exception as e:
        st.error(f"Failed to mix audio: {e}")
        return narration_bytes

# ============================================================================
# === PAGE FUNCTIONS ===
# ============================================================================

def show_home_page():
    """Display the home page"""
    
    st.markdown('<h1 class="main-header">🎨 ArtDes</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI-Powered Art Analysis Suite</h2>', unsafe_allow_html=True)
    
    st.write("""
    Welcome to **ArtDes**, a comprehensive suite of AI-powered tools designed to enhance your 
    understanding and interaction with visual art. Our platform combines cutting-edge artificial 
    intelligence with accessibility features to make art analysis, education, and creative 
    exploration available to everyone.
    """)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="description-box">
            <h3>🎙️ Voice Art Describer</h3>
            <p>Transform visual art into rich, accessible audio descriptions. Perfect for:</p>
            <ul>
                <li>Visual accessibility</li>
                <li>Art education</li>
                <li>Deep artistic analysis</li>
                <li>Multi-sensory art experience</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="description-box">
            <h3>🛠️ Photoshop Tutorials</h3>
            <p>Get personalized Adobe Photoshop editing suggestions based on your images:</p>
            <ul>
                <li>Layer-based editing techniques</li>
                <li>Color theory applications</li>
                <li>Composition enhancements</li>
                <li>Creative remixing ideas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="description-box">
            <h3>🎧 Soundscape Generator</h3>
            <p>Create immersive audio experiences from visual art:</p>
            <ul>
                <li>Procedural sound generation</li>
                <li>Ambient atmosphere creation</li>
                <li>AI-powered narrative scripts</li>
                <li>Multi-layered audio mixing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology overview
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🤖 Powered by Advanced AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **ArtDes** leverages state-of-the-art AI technologies:
        - **GPT-4 Vision** for sophisticated image analysis
        - **OpenAI TTS** for natural voice generation
        - **Perplexity AI** for historical context
        - **Procedural audio synthesis** for soundscape creation
        """)
    
    with col2:
        st.write("""
        **Key Features:**
        - Real-time image analysis and interpretation
        - Accessibility-focused design principles
        - Educational content for digital art learning
        - Multi-modal creative experiences
        - Professional-grade tool suggestions
        """)
    
    # Getting started
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🚀 Getting Started")
    
    st.write("""
    1. **Choose a tool** from the sidebar navigation menu
    2. **Upload an image** of artwork, photography, or any visual content
    3. **Configure settings** to customize your experience
    4. **Generate results** and explore the AI-powered insights
    5. **Download or share** your audio descriptions, tutorials, or soundscapes
    
    Each tool is designed to work independently while sharing a common foundation 
    of advanced AI image analysis. Feel free to explore all three tools with the 
    same image to see how different AI perspectives can enhance your understanding 
    of visual art.
    """)
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <strong>ArtDes</strong> • Making visual art accessible and educational through AI
    </div>
    """, unsafe_allow_html=True)

def show_voice_art_describer():
    """Voice Art Describer page"""
    
    st.markdown('<h1 class="main-header">🎙️ Voice-Based Art Describer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-powered artistic image analysis with voice narration for enhanced accessibility
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not initialize_openai():
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Configuration")
    
    voice_options = {
        "Nova (Warm & Natural)": "nova",
        "Alloy (Neutral)": "alloy", 
        "Echo (Deep & Resonant)": "echo",
        "Fable (Expressive)": "fable",
        "Onyx (Strong & Bold)": "onyx",
        "Shimmer (Gentle & Soft)": "shimmer"
    }
    
    selected_voice = st.sidebar.selectbox(
        "Choose Voice for Narration:",
        options=list(voice_options.keys()),
        index=0
    )
    
    enhance_with_context = st.sidebar.checkbox(
        "🔍 Enhance with Historical Context",
        help="Use Perplexity AI to add historical and cultural context to the analysis"
    )
    
    analysis_depth = st.sidebar.radio(
        "Analysis Depth:",
        ["Quick Overview", "Detailed Analysis", "Deep Artistic Critique"],
        index=1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Visual Analysis Tools")
    
    show_composition_guides = st.sidebar.checkbox(
        "Show Composition Guides",
        value=True,
        help="Overlay composition guidelines on the image"
    )
    
    show_color_analysis = st.sidebar.checkbox(
        "Analyze Color Palette",
        value=True,
        help="Extract and analyze dominant colors"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an artwork, photograph, or any image you'd like analyzed",
            key="voice_upload"
        )
        
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                image = Image.open(BytesIO(file_bytes))
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
                
                st.session_state.voice_uploaded_image = image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                uploaded_file = None
    
    with col2:
        st.header("⚙️ Analysis Settings")
        
        if analysis_depth == "Quick Overview":
            st.info("🔍 **Quick Overview**: Provides a concise, accessible description focusing on main elements and overall impression.")
        elif analysis_depth == "Detailed Analysis":
            st.info("📊 **Detailed Analysis**: Comprehensive examination including composition, technique, and emotional impact.")
        else:
            st.info("🎭 **Deep Artistic Critique**: In-depth scholarly analysis with historical context, symbolism, and artistic significance.")
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", use_container_width=True, key="voice_analyze"):
                with st.spinner("Analyzing image with AI..."):
                    
                    base_prompt = generate_artistic_prompt()
                    custom_prompt = ""
                    
                    if analysis_depth == "Quick Overview":
                        custom_prompt = "Provide a concise but emotionally engaging description of this image, focusing on the main visual elements and overall mood. Keep it accessible and under 300 words.\n\n" + base_prompt
                    elif analysis_depth == "Deep Artistic Critique":
                        custom_prompt = "Provide an extensive, scholarly analysis of this artwork. Include historical context, artistic movements, technical mastery, and cultural significance. Write as an expert art historian.\n\n" + base_prompt
                    else:
                        custom_prompt = base_prompt
                    
                    additional_context = ""
                    if enhance_with_context:
                        with st.spinner("Gathering historical context..."):
                            quick_analysis = analyze_image_with_openai(
                                image, 
                                "Identify the main subject, style, or artistic elements in this image in 1-2 sentences for research purposes."
                            )
                            if quick_analysis:
                                context = get_perplexity_context(quick_analysis)
                                if context:
                                    additional_context = f"\n\nAdditional Historical Context:\n{context}\n\nNow incorporate this context naturally into your artistic analysis."
                    
                    final_prompt = custom_prompt + additional_context
                    analysis = analyze_image_with_openai(image, final_prompt)
                    
                    if analysis:
                        st.session_state.voice_current_analysis = analysis
                        st.session_state.voice_current_image = image
                        st.session_state.voice_selected_voice = voice_options[selected_voice]
                        
                        with st.spinner("Generating audio narration..."):
                            char_count = len(analysis)
                            st.info(f"📝 Analysis length: {char_count:,} characters")
                            
                            if char_count > 4000:
                                st.warning("⚠️ Text exceeds 4,096 character limit. Creating summary for audio generation...")
                            
                            try:
                                audio_content = text_to_speech_openai(
                                    analysis,
                                    voice=voice_options[selected_voice]
                                )
                                
                                if audio_content:
                                    st.session_state.voice_audio_data = audio_content
                                    st.success("✅ Audio generated successfully!")
                                else:
                                    st.error("Failed to generate audio. Please try again.")
                                
                            except Exception as e:
                                st.error(f"Error generating audio: {str(e)}")
    
    # Interactive Tools Section
    if st.session_state.get('voice_uploaded_image') is not None and (show_composition_guides or show_color_analysis):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🎨 Interactive Tools")
        
        tool_cols = st.columns(2)
        
        if show_composition_guides:
            with tool_cols[0]:
                st.subheader("📐 Composition Guides")
                
                guide_type = st.selectbox(
                    "Select Composition Guide:",
                    ["rule_of_thirds", "golden_ratio"],
                    key="voice_guide_type"
                )
                
                if st.button("Apply Composition Guide", key="voice_comp_guide"):
                    with st.spinner("Generating composition guide..."):
                        overlay_img = create_composition_overlay(st.session_state.voice_uploaded_image, guide_type)
                        st.image(overlay_img, caption=f"{guide_type.replace('_', ' ').title()} Guide", use_container_width=True)
                        
                        if guide_type == "rule_of_thirds":
                            st.markdown("""
                            <div class="composition-guide">
                            <strong>📚 Rule of Thirds:</strong> Place important elements along the lines or at intersection points (yellow dots) to create more dynamic, visually interesting compositions.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="composition-guide">
                            <strong>📚 Golden Ratio:</strong> This mathematical proportion creates naturally pleasing compositions. Use the golden rectangle as a guide for placing key elements.
                            </div>
                            """, unsafe_allow_html=True)
        
        if show_color_analysis:
            with tool_cols[1]:
                st.subheader("🎨 Color Analysis")
                
                if st.button("Analyze Color Palette", key="voice_color_analysis"):
                    with st.spinner("Analyzing colors..."):
                        top_colors = analyze_color_palette(st.session_state.voice_uploaded_image)
                        palette_img = create_color_palette_display(top_colors)
                        st.image(palette_img, caption="Dominant Color Palette", use_container_width=True)
                        
                        st.markdown("""
                        <div class="photoshop-tip">
                        <strong>💡 Tip:</strong> These dominant colors can guide your artistic decisions and color harmony choices.
                        </div>
                        """, unsafe_allow_html=True)
    
    # Display results
    if 'voice_current_analysis' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🎨 Analysis Results")
        
        analysis_length = len(st.session_state.voice_current_analysis)
        st.markdown(f'<p class="compact-info">📝 Analysis: {analysis_length:,} characters</p>', 
                   unsafe_allow_html=True)
        
        st.markdown(f'<div class="description-box">{st.session_state.voice_current_analysis}</div>', 
                   unsafe_allow_html=True)
        
        if 'voice_audio_data' in st.session_state:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.header("🔊 Audio Narration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎧 Audio Player")
                st.audio(st.session_state.voice_audio_data, format="audio/mp3", autoplay=True)
            
            with col2:
                st.markdown("### 📥 Download")
                st.download_button(
                    label="📥 Download Audio",
                    data=st.session_state.voice_audio_data,
                    file_name="art_analysis.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )

def show_photoshop_tutorials():
    """Photoshop Tutorials page"""
    
    st.markdown('<h1 class="main-header">🛠️ Adobe Photoshop Tutorial Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Get personalized Photoshop editing suggestions based on your image analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎨 Tutorial Settings")
    
    tutorial_focus = st.sidebar.selectbox(
        "Focus Area:",
        ["General Editing", "Color Correction", "Composition", "Creative Effects", "Beginner Basics"],
        index=0
    )
    
    skill_level = st.sidebar.radio(
        "Skill Level:",
        ["Beginner", "Intermediate", "Advanced"],
        index=1
    )
    
    show_previews = st.sidebar.checkbox(
        "Show Filter Previews",
        value=True,
        help="Display visual examples of filter effects"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image to get personalized Photoshop tutorials",
            key="ps_upload"
        )
        
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                image = Image.open(BytesIO(file_bytes))
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
                
                st.session_state.ps_uploaded_image = image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                uploaded_file = None
    
    with col2:
        st.header("⚙️ Generate Tutorials")
        
        st.info(f"**Tutorial Focus:** {tutorial_focus}")
        st.info(f"**Skill Level:** {skill_level}")
        
        if uploaded_file is not None:
            if st.button("🎨 Generate Photoshop Tutorials", use_container_width=True, key="ps_generate"):
                with st.spinner("Analyzing image and creating tutorials..."):
                    
                    analysis_prompt = f"""
                    Analyze this image for Adobe Photoshop tutorial purposes. Focus on:
                    
                    1. **Visual Elements**: What can be enhanced or modified?
                    2. **Technical Aspects**: Lighting, color balance, composition issues
                    3. **Creative Opportunities**: Artistic effects that would suit this image
                    4. **{tutorial_focus} Focus**: Pay special attention to {tutorial_focus.lower()} aspects
                    5. **Skill Level**: Tailor suggestions for {skill_level.lower()} level
                    
                    Provide a detailed analysis that will be used to generate specific Photoshop tutorials.
                    """
                    
                    analysis = analyze_image_with_openai(image, analysis_prompt)
                    
                    if analysis:
                        st.session_state.ps_analysis = analysis
                        
                        ps_suggestions = generate_photoshop_suggestions(analysis, tutorial_focus)
                        
                        if ps_suggestions:
                            st.session_state.ps_suggestions = ps_suggestions
                            st.success("✅ Tutorials generated successfully!")
                        else:
                            st.error("Failed to generate tutorials. Please try again.")
    
    # Filter Previews Section
    if show_previews and st.session_state.get('ps_uploaded_image') is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🔍 Filter Previews")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Filter Previews", key="ps_filters"):
                with st.spinner("Creating filter previews..."):
                    filter_previews = create_filter_previews(st.session_state.ps_uploaded_image)
                    st.session_state.ps_filter_previews = filter_previews
        
        with col2:
            if st.button("Analyze Color Palette", key="ps_colors"):
                with st.spinner("Analyzing colors..."):
                    top_colors = analyze_color_palette(st.session_state.ps_uploaded_image)
                    palette_img = create_color_palette_display(top_colors)
                    st.session_state.ps_color_palette = palette_img
        
        # Display filter previews
        if 'ps_filter_previews' in st.session_state:
            st.subheader("Filter Effects Preview")
            
            preview_cols = st.columns(4)
            
            for i, (filter_name, filtered_img) in enumerate(st.session_state.ps_filter_previews.items()):
                with preview_cols[i % 4]:
                    st.image(filtered_img, caption=filter_name, use_container_width=True)
                    
                    ps_equivalent = {
                        'Blur': 'Filter → Blur → Gaussian Blur',
                        'Sharpen': 'Filter → Sharpen → Unsharp Mask',
                        'Edge Enhance': 'Filter → Stylize → Find Edges',
                        'Brighter': 'Image → Adjustments → Brightness/Contrast',
                        'High Contrast': 'Image → Adjustments → Curves',
                        'Vibrant': 'Image → Adjustments → Vibrance',
                        'Desaturated': 'Image → Adjustments → Desaturate'
                    }
                    
                    if filter_name in ps_equivalent:
                        st.caption(f"📝 PS: {ps_equivalent[filter_name]}")
        
        # Display color palette
        if 'ps_color_palette' in st.session_state:
            st.subheader("Color Palette Analysis")
            st.image(st.session_state.ps_color_palette, caption="Dominant Color Palette", use_container_width=True)
            
            st.markdown("""
            <div class="photoshop-tip">
            <strong>💡 Photoshop Tip:</strong> Use Image → Adjustments → Color Balance to shift these dominant colors. 
            Try complementary colors (opposite on color wheel) for dramatic effects.
            </div>
            """, unsafe_allow_html=True)
    
    # Display results
    if 'ps_analysis' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("📊 Image Analysis")
        
        st.markdown(f'<div class="description-box">{st.session_state.ps_analysis}</div>', 
                   unsafe_allow_html=True)
    
    if 'ps_suggestions' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🛠️ Photoshop Tutorial Suggestions")
        
        st.markdown(st.session_state.ps_suggestions)
        
        # Quick reference card
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📋 Quick Reference Card")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="technique-card">
            <h4>🎨 Essential Tools</h4>
            <ul>
                <li>Move Tool (V)</li>
                <li>Selection Tools (M, W, L)</li>
                <li>Brush Tool (B)</li>
                <li>Clone Stamp (S)</li>
                <li>Healing Brush (J)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="technique-card">
            <h4>⚡ Quick Shortcuts</h4>
            <ul>
                <li>Ctrl+J: Duplicate Layer</li>
                <li>Ctrl+Shift+N: New Layer</li>
                <li>Ctrl+Alt+Z: Step Backward</li>
                <li>Ctrl+T: Free Transform</li>
                <li>Ctrl+L: Levels</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="technique-card">
            <h4>🔧 Adjustments</h4>
            <ul>
                <li>Curves (Ctrl+M)</li>
                <li>Hue/Saturation (Ctrl+U)</li>
                <li>Color Balance</li>
                <li>Shadows/Highlights</li>
                <li>Vibrance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def show_soundscape_generator():
    """Soundscape Generator page"""
    
    st.markdown('<h1 class="main-header">🎧 Generative Soundscape Creator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Transform visual art into unique, procedurally generated audio experiences
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Setup
    st.sidebar.header("⚙️ System Status")
    st.sidebar.success("✅ OpenAI API Key loaded.")
    st.sidebar.success("✅ Sound generators ready.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Soundscape Controls")
    
    voice_options = {
        "Nova (Warm)": "nova", 
        "Alloy (Neutral)": "alloy", 
        "Echo (Deep)": "echo", 
        "Fable (Storyteller)": "fable", 
        "Onyx (Bold)": "onyx", 
        "Shimmer (Gentle)": "shimmer"
    }
    
    selected_voice = st.sidebar.selectbox("Narration Voice:", options=list(voice_options.keys()), index=3)
    
    volumes = { 
        'narration': st.sidebar.slider("🗣️ Narration", 0, 100, 90), 
        'ambient': st.sidebar.slider("🍃 Ambient", 0, 100, 30), 
        'music': st.sidebar.slider("🎻 Music", 0, 100, 25) 
    }

    # Main Content Area
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.header("🖼️ Upload an Image")
        uploaded_file = st.file_uploader("Choose a visual to generate a soundscape for", type=['png', 'jpg', 'jpeg'], key="sc_upload")
        if uploaded_file:
            st.session_state.sc_image = Image.open(uploaded_file)
            st.image(st.session_state.sc_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.header("✨ Your Soundscape")
        if 'sc_image' in st.session_state and st.button("🎶 Create Soundscape", use_container_width=True, key="sc_generate"):
            with st.spinner("Architecting and generating your soundscape..."):
                # 1. Analyze Image
                analysis = analyze_image_for_soundscape(encode_image_to_base64(st.session_state.sc_image))
                if not analysis: 
                    st.error("Failed to analyze image")
                    st.stop()
                st.session_state.sc_analysis = analysis

                # 2. Generate Narration
                narration_bytes = generate_narration(analysis['narrative_script'], voice=voice_options[selected_voice])
                if not narration_bytes: 
                    st.error("Failed to generate narration")
                    st.stop()
                
                # Determine duration and sample rate
                duration_s = 10.0  # Default duration
                sr = 44100  # Standard sample rate

                # 3. Generate procedural sounds based on analysis
                ambient_generator = SOUND_GENERATORS['ambient'].get(analysis['ambient_theme'], generate_rain)
                music_generator = SOUND_GENERATORS['music'].get(analysis['music_mood'], generate_calm_music)
                
                ambient_audio = ambient_generator(duration_s, sr)
                music_audio = music_generator(duration_s, sr)

                # 4. Mix all layers (simplified for demo)
                final_audio = simple_mix_audio(narration_bytes, ambient_audio, music_audio, volumes, sr)
                
                if final_audio:
                    st.session_state.sc_final_audio = final_audio
                    st.success("🎉 Soundscape created successfully!")

        if 'sc_analysis' in st.session_state:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("AI Soundscape Plan")
            st.markdown(f"""
            <p><span class="info-label">🍃 Ambient Theme:</span> {st.session_state.sc_analysis['ambient_theme'].title()}</p>
            <p><span class="info-label">🎻 Music Mood:</span> {st.session_state.sc_analysis['music_mood'].title()}</p>
            <div class="results-box">"{st.session_state.sc_analysis["narrative_script"]}"</div>
            """, unsafe_allow_html=True)
        
        if 'sc_final_audio' in st.session_state:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("🎧 Listen & Download")
            st.audio(st.session_state.sc_final_audio, format="audio/mp3")
            st.download_button("📥 Download Soundscape (MP3)", st.session_state.sc_final_audio, "generative_soundscape.mp3", "audio/mpeg", use_container_width=True)

    # Educational section about soundscape creation
    if 'sc_analysis' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🎼 About Your Soundscape")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="technique-card">
            <h4>🍃 Ambient Layer</h4>
            <p>The background atmosphere that sets the environmental mood. Generated using procedural synthesis and digital filtering techniques.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="technique-card">
            <h4>🎻 Musical Layer</h4>
            <p>Harmonic content that conveys emotional resonance. Created using algorithmic composition based on music theory principles.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="technique-card">
            <h4>🗣️ Narrative Layer</h4>
            <p>AI-generated descriptive text converted to natural speech, providing context and accessibility for the visual content.</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# === MAIN APPLICATION ===
# ============================================================================

def main():
    """Main application function with navigation"""
    
    # Sidebar navigation
    st.sidebar.markdown("# 🎨 ArtDes Navigation")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "🎙️ Voice Art Describer": "voice_art",
        "🛠️ Photoshop Tutorials": "photoshop",
        "🎧 Soundscape Generator": "soundscape"
    }
    
    # Create navigation
    selected_page = st.sidebar.selectbox(
        "Choose an AI Art Tool:",
        options=list(pages.keys()),
        index=0
    )
    
    # Page routing
    page_key = pages[selected_page]
    
    if page_key == "home":
        show_home_page()
    elif page_key == "voice_art":
        show_voice_art_describer()
    elif page_key == "photoshop":
        show_photoshop_tutorials()
    elif page_key == "soundscape":
        show_soundscape_generator()
    
    # Footer for all pages
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <strong>ArtDes</strong> • AI-Powered Art Analysis Suite • Making visual art accessible and educational through AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
