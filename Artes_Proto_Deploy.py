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

# Configure the page
st.set_page_config(
    page_title="Voice-Based Art Describer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
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
    .learning-objective {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 0.5rem 0;
    }
    .technique-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .filter-preview {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem;
        text-align: center;
    }
    .composition-guide {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .course-info {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
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
    .compact-info {
        font-size: 0.9rem;
        color: #666;
        margin: 0.5rem 0;
    }
    .settings-expander {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_openai():
    """Initialize OpenAI client with API key"""
    # For local development, use hardcoded keys
    # For production, always use environment variables or secure key management
    
    with st.sidebar.expander("🔑 API Configuration", expanded=False):
        st.success("✅ API Keys Loaded")
        st.caption("OpenAI and Perplexity APIs configured")
        
        # Option to override with custom keys if needed
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

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    try:
        # Ensure image is in RGB mode for consistency
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

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

def generate_photoshop_suggestions(analysis_text, learning_objective="both"):
    """Generate specific Photoshop editing suggestions targeting ART250 learning objectives"""
    
    # Define the two hardcoded learning objectives
    raster_objective = """Applies the elements and principles of design in a finished digital image to produce a raster-based digital artwork that utilizes non-destructive editing, layers and layer masks to combine elements from three separate visual images."""
    
    vector_objective = """Applies the elements and principles of design in a finished digital image, to produce a vector-based digital project designing a decorative letterforms alphabet, crafted with precise vector shapes, utilizing the pen tool "suite" of tools."""
    
    if learning_objective == "raster":
        target_objective = raster_objective
        focus_area = "raster-based digital artwork with non-destructive editing"
    elif learning_objective == "vector":
        target_objective = vector_objective
        focus_area = "vector-based digital artwork with precise shapes"
    else:
        target_objective = f"{raster_objective}\n\nOR\n\n{vector_objective}"
        focus_area = "both raster and vector-based digital artwork"
    
    photoshop_prompt = f"""
    Based on this art analysis, provide specific Adobe Photoshop editing tutorials that directly support this ART250 learning objective:

    **TARGET LEARNING OBJECTIVE:**
    {target_objective}

    **Original Image Analysis:**
    {analysis_text}

    Create tutorials that help students apply elements and principles of design while learning {focus_area}. Structure your response as follows:

    1. **Learning Objective Connection** (1-2 sentences):
       - Explicitly state how these tutorials support the ART250 learning objective

    2. **Non-Destructive Raster Techniques** (4-5 specific tutorials):
       - Layer-based editing approaches that preserve original image data
       - Layer mask applications for combining multiple visual elements
       - Adjustment layer techniques for color and tone
       - Smart object workflows
       - Each tutorial should include: Tool/Menu path, Step-by-step instructions, Design principle application

    3. **Vector-Based Techniques** (3-4 specific tutorials):
       - Pen tool applications for precise shape creation
       - Vector shape design principles
       - Typography and letterform creation
       - Path manipulation techniques
       - Each tutorial should include: Tool/Menu path, Step-by-step instructions, Design principle application

    4. **Design Elements & Principles Integration** (3-4 applications):
       - How to apply specific design principles (balance, contrast, emphasis, etc.)
       - Color theory applications
       - Composition enhancement techniques
       - Visual hierarchy creation

    5. **Hands-On Project Suggestions** (2-3 projects):
       - Specific projects that combine multiple images using the analyzed artwork as inspiration
       - Vector letterform design projects inspired by the artwork's style
       - Assessment criteria aligned with learning objectives

    Format each tutorial with:
    - **Tutorial Title**
    - **Learning Objective Alignment**: (Raster/Vector/Both)
    - **Difficulty**: Beginner/Intermediate/Advanced
    - **Tools Required**: Specific Photoshop tools/panels
    - **Step-by-Step Process**: Numbered instructions
    - **Design Principle Focus**: Which element/principle of design this reinforces
    - **Assessment Checkpoint**: How to evaluate success

    Make tutorials practical, educational, and directly tied to demonstrating competency in the ART250 learning objectives.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": photoshop_prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating Photoshop suggestions: {str(e)}")
        return None

def create_composition_overlay(image, guide_type="rule_of_thirds"):
    """Create composition guide overlays on the image"""
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_array)
    
    height, width = img_array.shape[:2]
    
    if guide_type == "rule_of_thirds":
        # Add rule of thirds lines
        ax.axhline(y=height/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=2*height/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=width/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(x=2*width/3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add intersection points
        for i in [1, 2]:
            for j in [1, 2]:
                circle = plt.Circle((j*width/3, i*height/3), 15, color='yellow', alpha=0.7)
                ax.add_patch(circle)
        
        ax.set_title("Rule of Thirds Guide", color='white', fontsize=16, pad=20)
        
    elif guide_type == "golden_ratio":
        # Golden ratio spiral approximation
        phi = 1.618
        rect_width = width / phi
        rect_height = height / phi
        
        rect = patches.Rectangle((0, 0), rect_width, rect_height, 
                               linewidth=2, edgecolor='gold', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        ax.set_title("Golden Ratio Guide", color='white', fontsize=16, pad=20)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='black', edgecolor='none')
    buf.seek(0)
    overlay_image = Image.open(buf)
    plt.close()
    
    return overlay_image

def create_filter_previews(image):
    """Create quick filter previews to demonstrate Photoshop-like effects"""
    
    # Resize image for faster processing
    preview_size = (200, 150)
    preview_img = image.copy()
    preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)
    
    filters = {}
    
    # Blur effect (similar to Gaussian Blur in Photoshop)
    filters['Blur'] = preview_img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Sharpen effect
    filters['Sharpen'] = preview_img.filter(ImageFilter.SHARPEN)
    
    # Edge enhance
    filters['Edge Enhance'] = preview_img.filter(ImageFilter.EDGE_ENHANCE)
    
    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(preview_img)
    filters['Brighter'] = enhancer.enhance(1.3)
    
    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(preview_img)
    filters['High Contrast'] = enhancer.enhance(1.5)
    
    # Saturation adjustment
    enhancer = ImageEnhance.Color(preview_img)
    filters['Vibrant'] = enhancer.enhance(1.4)
    filters['Desaturated'] = enhancer.enhance(0.3)
    
    return filters

def analyze_color_palette(image):
    """Extract and analyze the color palette of the image"""
    
    # Resize for faster processing
    small_image = image.copy()
    small_image.thumbnail((100, 100), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if small_image.mode != 'RGB':
        small_image = small_image.convert('RGB')
    
    # Get pixel data
    pixels = list(small_image.getdata())
    
    # Simple color analysis (you could use more sophisticated clustering)
    # Round colors to reduce variation
    rounded_colors = []
    for r, g, b in pixels:
        # Round to nearest 32 to group similar colors
        r_round = (r // 32) * 32
        g_round = (g // 32) * 32
        b_round = (b // 32) * 32
        rounded_colors.append((r_round, g_round, b_round))
    
    # Get most common colors
    color_counts = Counter(rounded_colors)
    top_colors = color_counts.most_common(5)
    
    return top_colors

def create_color_palette_display(top_colors):
    """Create a visual display of the extracted color palette"""
    
    fig, axes = plt.subplots(1, len(top_colors), figsize=(10, 2))
    
    for i, ((r, g, b), count) in enumerate(top_colors):
        # Normalize RGB values
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
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    palette_image = Image.open(buf)
    plt.close()
    
    return palette_image

def analyze_image_with_openai(image, custom_prompt=""):
    """Send image to OpenAI for analysis"""
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        
        if not base64_image:
            return None
        
        # Use custom prompt or default artistic prompt
        prompt = custom_prompt if custom_prompt.strip() else generate_artistic_prompt()
        
        # Create the API request using the global client
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 with vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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

def text_to_speech_openai(text, voice="nova"):
    """Convert text to speech using OpenAI's TTS API with chunking for long texts"""
    # OpenAI TTS has a 4096 character limit
    MAX_CHARS = 4000  # Leave some buffer
    
    if len(text) <= MAX_CHARS:
        # Text is short enough, process normally
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
        # Text is too long, need to create summary
        st.info(f"🔄 Creating audio-optimized summary ({len(text):,} → ~3,500 characters)...")
        
        # Create a summary for audio generation
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
            
            # Double-check the summary length
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
                # Summary is still too long, truncate intelligently
                st.warning("Summary still too long, applying smart truncation...")
                # Find a good breaking point (end of sentence)
                truncate_point = MAX_CHARS - 100
                last_period = summary_text.rfind('.', 0, truncate_point)
                if last_period > truncate_point - 200:  # If period is reasonably close
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
            # Final fallback: simple truncation
            st.info("🔄 Falling back to simple truncation...")
            truncate_point = MAX_CHARS - 100
            last_period = text.rfind('.', 0, truncate_point)
            if last_period > truncate_point - 200:
                truncated_text = text[:last_period + 1]
            else:
                truncated_text = text[:truncate_point] + "..."
            
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=truncated_text,
                    speed=0.9
                )
                st.info(f"📝 Used fallback truncation: {len(truncated_text):,} characters")
                return response.content
            except Exception as final_error:
                st.error(f"Final TTS attempt failed: {str(final_error)}")
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Voice-Based Art Describer</h1>', unsafe_allow_html=True)
    
    # Course Information Section - Simplified
    st.write("## 📚 ART250: Introduction to Digital Art")
    
    st.write("""This application supports digital art education by providing AI-powered image analysis and Adobe Photoshop tutorials 
    specifically designed to help students meet ART250 learning objectives. The tool focuses on both raster-based digital artwork 
    using non-destructive editing techniques and vector-based design projects.""")
    
    st.write("**🔗 Course Resources:** [View Official ART250 Course Outline of Record](https://laccd.elumenapp.com/public/courses/course-cor/27398)")
    
    st.write("**Key Learning Focus:** Elements and principles of design, layer techniques, non-destructive editing, and vector-based design using precise pen tool workflows.")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-powered artistic image analysis with voice narration for enhanced accessibility
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize OpenAI
    if not initialize_openai():
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Configuration")
    
    # Voice selection
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
    
    # Enhanced analysis options
    enhance_with_context = st.sidebar.checkbox(
        "🔍 Enhance with Historical Context",
        help="Use Perplexity AI to add historical and cultural context to the analysis"
    )
    
    # Analysis depth
    analysis_depth = st.sidebar.radio(
        "Analysis Depth:",
        ["Quick Overview", "Detailed Analysis", "Deep Artistic Critique"],
        index=1
    )
    
    # New: Learning Objective Focus
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 ART250 Learning Objectives")
    
    learning_objective_focus = st.sidebar.selectbox(
        "Tutorial Focus:",
        ["Both Objectives", "Raster-Based Digital Art", "Vector-Based Design"],
        help="Choose which ART250 learning objective to emphasize in Photoshop tutorials"
    )
    
    # Digital Art Tools Toggle
    st.sidebar.header("🛠️ Digital Art Tools")
    
    show_photoshop_tools = st.sidebar.checkbox(
        "Enable Photoshop Tutorials",
        value=True,
        help="Show educational tutorials aligned with ART250 learning objectives"
    )
    
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
    
    show_filter_previews = st.sidebar.checkbox(
        "Show Filter Previews",
        value=True,
        help="Preview different filter effects"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an artwork, photograph, or any image you'd like analyzed"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file content properly
                file_bytes = uploaded_file.read()
                
                # Create PIL image from bytes
                image = Image.open(BytesIO(file_bytes))
                
                # Display the image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image info
                st.info(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
                
                # Store image in session state for analysis
                st.session_state.uploaded_image = image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.info("Please make sure you've uploaded a valid image file (PNG, JPG, JPEG, GIF, BMP)")
                uploaded_file = None
    
    with col2:
        st.header("⚙️ Analysis Settings")
        
        # Preset prompt information based on analysis depth
        if analysis_depth == "Quick Overview":
            st.info("🔍 **Quick Overview**: Provides a concise, accessible description focusing on main elements and overall impression.")
        elif analysis_depth == "Detailed Analysis":
            st.info("📊 **Detailed Analysis**: Comprehensive examination including composition, technique, and emotional impact.")
        else:
            st.info("🎭 **Deep Artistic Critique**: In-depth scholarly analysis with historical context, symbolism, and artistic significance.")
        
        # Learning objective information
        if learning_objective_focus == "Raster-Based Digital Art":
            st.markdown("""
            <div class="learning-objective">
            <strong>🎯 Raster Focus:</strong> Tutorials will emphasize non-destructive editing, layers, and layer masks 
            for combining elements from multiple images.
            </div>
            """, unsafe_allow_html=True)
        elif learning_objective_focus == "Vector-Based Design":
            st.markdown("""
            <div class="learning-objective">
            <strong>🎯 Vector Focus:</strong> Tutorials will emphasize pen tool techniques, precise vector shapes, 
            and decorative letterform design.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="learning-objective">
            <strong>🎯 Comprehensive Focus:</strong> Tutorials will cover both raster and vector techniques 
            to support all ART250 learning objectives.
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image with AI..."):
                    
                    # Modify prompt based on depth selection
                    base_prompt = generate_artistic_prompt()
                    custom_prompt = ""
                    
                    if analysis_depth == "Quick Overview":
                        custom_prompt = "Provide a concise but emotionally engaging description of this image, focusing on the main visual elements and overall mood. Keep it accessible and under 300 words.\n\n" + base_prompt
                    elif analysis_depth == "Deep Artistic Critique":
                        custom_prompt = "Provide an extensive, scholarly analysis of this artwork. Include historical context, artistic movements, technical mastery, and cultural significance. Write as an expert art historian.\n\n" + base_prompt
                    else:
                        custom_prompt = base_prompt
                    
                    # Get additional context from Perplexity if requested
                    additional_context = ""
                    if enhance_with_context:
                        with st.spinner("Gathering historical context..."):
                            # Try to identify key elements for context search
                            quick_analysis = analyze_image_with_openai(
                                image, 
                                "Identify the main subject, style, or artistic elements in this image in 1-2 sentences for research purposes."
                            )
                            if quick_analysis:
                                context = get_perplexity_context(quick_analysis)
                                if context:
                                    additional_context = f"\n\nAdditional Historical Context:\n{context}\n\nNow incorporate this context naturally into your artistic analysis."
                    
                    # Analyze the image
                    final_prompt = custom_prompt + additional_context
                    analysis = analyze_image_with_openai(image, final_prompt)
                    
                    if analysis:
                        st.session_state.current_analysis = analysis
                        st.session_state.current_image = image
                        st.session_state.selected_voice = voice_options[selected_voice]
                        st.session_state.learning_objective = learning_objective_focus
                        
                        # Automatically generate audio after analysis
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
                                    # Store audio data
                                    st.session_state.audio_data = audio_content
                                    st.success("✅ Audio generated successfully!")
                                else:
                                    st.error("Failed to generate audio. Please try again.")
                                
                            except Exception as e:
                                st.error(f"Error generating audio: {str(e)}")
                                # Additional fallback - offer to create shorter version
                                if "string_too_long" in str(e).lower():
                                    st.info("💡 Tip: Try using 'Quick Overview' analysis depth for shorter audio-friendly results.")
    
    # New: Interactive Tools Section
    if st.session_state.get('uploaded_image') is not None and (show_composition_guides or show_color_analysis or show_filter_previews):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🎨 Interactive Tools")
        
        # Create columns for tools
        tool_cols = st.columns(3)
        
        # Composition Guides
        if show_composition_guides:
            with tool_cols[0]:
                st.subheader("📐 Composition Guides")
                
                guide_type = st.selectbox(
                    "Select Composition Guide:",
                    ["rule_of_thirds", "golden_ratio"]
                )
                
                if st.button("Apply Composition Guide"):
                    with st.spinner("Generating composition guide..."):
                        overlay_img = create_composition_overlay(st.session_state.uploaded_image, guide_type)
                        st.image(overlay_img, caption=f"{guide_type.replace('_', ' ').title()} Guide", use_container_width=True)
                        
                        # Educational info
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
        
        # Color Analysis
        if show_color_analysis:
            with tool_cols[1]:
                st.subheader("🎨 Color Analysis")
                
                if st.button("Analyze Color Palette"):
                    with st.spinner("Analyzing colors..."):
                        top_colors = analyze_color_palette(st.session_state.uploaded_image)
                        palette_img = create_color_palette_display(top_colors)
                        st.image(palette_img, caption="Dominant Color Palette", use_container_width=True)
                        
                        # Color theory tips
                        st.markdown("""
                        <div class="photoshop-tip">
                        <strong>💡 Photoshop Tip:</strong> Use Image → Adjustments → Color Balance to shift these dominant colors. Try complementary colors (opposite on color wheel) for dramatic effects.
                        </div>
                        """, unsafe_allow_html=True)
        
        # Filter Previews
        if show_filter_previews:
            with tool_cols[2]:
                st.subheader("🔍 Filter Previews")
                
                if st.button("Generate Filter Previews"):
                    with st.spinner("Creating filter previews..."):
                        filter_previews = create_filter_previews(st.session_state.uploaded_image)
                        
                        # Display in a grid
                        for i, (filter_name, filtered_img) in enumerate(filter_previews.items()):
                            st.image(filtered_img, caption=filter_name, use_container_width=True)
                            
                            # Add Photoshop equivalent info
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
    
    # Display analysis results (without the full text section for voice-over)
    if 'current_analysis' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("🎨 Analysis Summary")
        
        # Just show a brief summary instead of the full analysis text
        st.write("✅ **Image analysis completed successfully!** The detailed analysis has been processed and is available in the audio narration below.")
        
        # Show analysis length info
        analysis_length = len(st.session_state.current_analysis)
        st.markdown(f'<p class="compact-info">📝 Analysis: {analysis_length:,} characters generated</p>', 
                   unsafe_allow_html=True)
        
        # Audio section
        if 'audio_data' in st.session_state:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.header("🔊 Audio Narration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎧 Audio Player")
                st.audio(st.session_state.audio_data, format="audio/mp3", autoplay=True)
            
            with col2:
                st.markdown("### 📥 Download")
                st.download_button(
                    label="📥 Download Audio",
                    data=st.session_state.audio_data,
                    file_name="art_analysis.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )
        
        # ART250 Photoshop Tutorials Section
        if show_photoshop_tools:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.header("🛠️ ART250 Adobe Photoshop Tutorials")
            
            # Map learning objective selection to function parameter
            objective_mapping = {
                "Raster-Based Digital Art": "raster",
                "Vector-Based Design": "vector", 
                "Both Objectives": "both"
            }
            
            selected_objective = objective_mapping.get(st.session_state.get('learning_objective', 'Both Objectives'), 'both')
            
            if st.button("Generate ART250-Aligned Photoshop Tutorials"):
                with st.spinner("Creating personalized Photoshop tutorials aligned with ART250 learning objectives..."):
                    ps_suggestions = generate_photoshop_suggestions(
                        st.session_state.current_analysis,
                        learning_objective=selected_objective
                    )
                    
                    if ps_suggestions:
                        st.markdown(ps_suggestions)
                    else:
                        # Enhanced fallback suggestions aligned with ART250
                        st.markdown(f"""
                        ### 🎯 ART250-Aligned Photoshop Techniques
                        
                        **Learning Objective Connection:** These tutorials support the ART250 learning objectives by teaching students to apply design elements and principles through both raster-based non-destructive editing and vector-based precise shape creation.
                        
                        #### **Non-Destructive Raster Techniques** 
                        
                        **1. Layer Mask Combination Project**
                        - **Learning Objective Alignment**: Raster
                        - **Difficulty**: Intermediate
                        - **Tools Required**: Layer Panel, Layer Mask, Brush Tool
                        - **Step-by-Step Process**:
                          1. Open three separate visual images
                          2. Create new document (File → New)
                          3. Drag each image into document as separate layers
                          4. Add layer masks (Layer → Layer Mask → Add Layer Mask)
                          5. Use soft brush (B) to blend elements seamlessly
                          6. Apply design principle of balance by distributing visual weight
                        - **Design Principle Focus**: Balance and Unity
                        - **Assessment Checkpoint**: Successfully combined three images with smooth transitions
                        
                        **2. Smart Object Workflow**
                        - **Learning Objective Alignment**: Raster
                        - **Difficulty**: Advanced
                        - **Tools Required**: Smart Objects, Adjustment Layers
                        - **Step-by-Step Process**:
                          1. Convert layers to Smart Objects (Right-click → Convert to Smart Object)
                          2. Apply filters non-destructively
                          3. Add Adjustment Layers above Smart Objects
                          4. Use layer masks for selective editing
                        - **Design Principle Focus**: Contrast and Emphasis
                        - **Assessment Checkpoint**: Edits remain reversible and non-destructive
                        
                        #### **Vector-Based Techniques**
                        
                        **1. Decorative Letterform Creation**
                        - **Learning Objective Alignment**: Vector
                        - **Difficulty**: Intermediate
                        - **Tools Required**: Pen Tool (P), Shape Tools, Type Tool
                        - **Step-by-Step Process**:
                          1. Create new document
                          2. Use Pen Tool to create precise letter outlines
                          3. Apply Path Operations (Window → Pathfinder) 
                          4. Add decorative elements with shape tools
                          5. Ensure vector scalability
                        - **Design Principle Focus**: Proportion and Rhythm
                        - **Assessment Checkpoint**: Letters are scalable vector shapes with precise curves
                        
                        **2. Pen Tool Suite Mastery**
                        - **Learning Objective Alignment**: Vector  
                        - **Difficulty**: Beginner to Advanced
                        - **Tools Required**: Pen Tool, Direct Selection Tool, Convert Point Tool
                        - **Step-by-Step Process**:
                          1. Practice anchor point placement
                          2. Master curve creation with direction handles
                          3. Use Convert Point Tool to adjust curve types
                          4. Create complex shapes through path combination
                        - **Design Principle Focus**: Movement and Direction
                        - **Assessment Checkpoint**: Smooth, precise vector paths with proper curve flow
                        
                        #### **Design Elements & Principles Integration**
                        
                        **1. Visual Hierarchy Project**
                        - Apply contrast through scale, color, and position
                        - Use typography and imagery to guide viewer's eye
                        - Implement focal points using design principles
                        
                        **2. Color Theory Application**
                        - Use Color Picker and Adjustment Layers
                        - Apply complementary color schemes
                        - Create mood through color temperature
                        
                        **Reference:** [ART250 Course Outline](https://laccd.elumenapp.com/public/courses/course-cor/27398)
                        """)

    # Minimal footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <strong>Voice-Based Art Describer</strong> • Supporting ART250 Digital Art Education • Making visual art accessible through AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
