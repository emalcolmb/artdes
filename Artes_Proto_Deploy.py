import streamlit as st
import openai
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import requests
import json

# === API Keys (LOCAL USE ONLY) ===
PERPLEXITY_API_KEY = "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"
OPENAI_API_KEY = "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"


# Initialize OpenAI client
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
    .conversational-info {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_openai():
    """Initialize OpenAI client with API key"""
    
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

def generate_conversational_prompt(user_name, image_source):
    """Generate a conversational prompt for three friends discussing art"""
    return f"""
    Create a natural, engaging conversation between three friends discussing this artwork. The conversation should feel authentic and immersive, as if the listener is part of the group.

    **Characters:**
    - **Alex**: An enthusiastic art critic with expertise in contemporary art and visual analysis
    - **Morgan**: A knowledgeable art historian who focuses on technique, historical context, and artistic movements  
    - **{user_name}**: A friend who discovered this image from {image_source.lower()} (silent participant but actively referenced)

    **Conversation Guidelines:**
    - Start with Alex or Morgan greeting {user_name} and asking about where they found this image
    - Include natural dialogue, interruptions, and back-and-forth discussion
    - Direct questions to {user_name} like "{user_name}, what do you think about..." or "Hey {user_name}, where did you find this?"
    - Include pauses and natural speech patterns: "Oh wow, look at that..." "Wait, do you see how..."
    - Make it feel like {user_name} is sitting with them, even though they're listening
    - Cover the same analytical depth as a formal critique but in casual, friendly language
    - Include moments where they ask {user_name}'s opinion or reference their discovery from {image_source.lower()}

    **Content to Cover (naturally woven into conversation):**
    - Visual composition and what immediately catches the eye
    - Color choices and emotional impact
    - Artistic technique and style
    - Subject matter and any symbolic meaning
    - Personal reactions and interpretations
    - Historical or cultural context if relevant
    - Accessibility-focused descriptions for those who can't see the image

    **Tone:** Friendly, enthusiastic, knowledgeable but not pretentious. Like friends who love art sharing their excitement about a new discovery.

    Format the response as a natural conversation with speaker labels:
    Alex: [dialogue]
    Morgan: [dialogue]
    etc.

    The conversation should be substantial (8-12 exchanges) and feel like a real discussion between art-loving friends.
    """

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
    
    # Analysis depth with conversational option
    analysis_depth = st.sidebar.radio(
        "Analysis Depth:",
        ["Quick Overview", "Detailed Analysis", "Deep Artistic Critique", "Conversational"],
        index=1
    )
    
    # Conversational mode additional components
    user_name = ""
    image_source = ""
    
    if analysis_depth == "Conversational":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💬 Conversational Settings")
        
        user_name = st.sidebar.text_input(
            "Enter Your Name:",
            placeholder="e.g., Sarah",
            help="Your name will be used in the conversation between the art critics"
        )
        
        image_source_options = [
            "Social Media",
            "Museum", 
            "Art Show",
            "Gallery",
            "Film"
        ]
        
        image_source = st.sidebar.selectbox(
            "Image Source:",
            options=image_source_options,
            help="Where did you find this image? This will be referenced in the conversation."
        )
        
        if user_name:
            st.sidebar.markdown(f'<div class="conversational-info">👥 <strong>Conversation Mode Active</strong><br/>You\'ll hear Alex and Morgan (art critics) discussing the image with references to you ({user_name}) finding it from {image_source.lower()}.</div>', unsafe_allow_html=True)
    
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
        elif analysis_depth == "Deep Artistic Critique":
            st.info("🎭 **Deep Artistic Critique**: In-depth scholarly analysis with historical context, symbolism, and artistic significance.")
        elif analysis_depth == "Conversational":
            if user_name:
                st.info(f"💬 **Conversational Mode**: Listen to Alex and Morgan (art critics) discuss the image in a natural conversation, with references to you ({user_name}) and your discovery from {image_source.lower()}.")
            else:
                st.warning("⚠️ **Conversational Mode**: Please enter your name in the sidebar to activate conversation mode.")
        
        # Analysis button
        if uploaded_file is not None:
            # Check if conversational mode requires name
            can_analyze = True
            if analysis_depth == "Conversational" and not user_name.strip():
                can_analyze = False
                st.warning("Please enter your name in the sidebar for Conversational mode.")
            
            if can_analyze and st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image with AI..."):
                    
                    # Modify prompt based on depth selection
                    base_prompt = generate_artistic_prompt()
                    custom_prompt = ""
                    
                    if analysis_depth == "Quick Overview":
                        custom_prompt = "Provide a concise but emotionally engaging description of this image, focusing on the main visual elements and overall mood. Keep it accessible and under 300 words.\n\n" + base_prompt
                    elif analysis_depth == "Deep Artistic Critique":
                        custom_prompt = "Provide an extensive, scholarly analysis of this artwork. Include historical context, artistic movements, technical mastery, and cultural significance. Write as an expert art historian.\n\n" + base_prompt
                    elif analysis_depth == "Conversational":
                        custom_prompt = generate_conversational_prompt(user_name.strip(), image_source)
                    else:
                        custom_prompt = base_prompt
                    
                    # Get additional context from Perplexity if requested
                    additional_context = ""
                    if enhance_with_context and analysis_depth != "Conversational":
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
                        st.session_state.analysis_mode = analysis_depth
                        
                        # Store conversational details if applicable
                        if analysis_depth == "Conversational":
                            st.session_state.user_name = user_name.strip()
                            st.session_state.image_source = image_source
                        
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
    
    # Display results
    if 'current_analysis' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Dynamic header based on analysis mode
        if st.session_state.get('analysis_mode') == "Conversational":
            st.header("💬 Art Conversation")
            if 'user_name' in st.session_state:
                st.markdown(f"**🎧 Listen to Alex and Morgan discuss the artwork with references to {st.session_state.user_name}**")
        else:
            st.header("🎨 Analysis Results")
        
        # Display the analysis with character count
        analysis_length = len(st.session_state.current_analysis)
        st.markdown(f'<p class="compact-info">📝 Content: {analysis_length:,} characters</p>', 
                   unsafe_allow_html=True)
        
        st.markdown(f'<div class="description-box">{st.session_state.current_analysis}</div>', 
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
                # Dynamic filename based on mode
                filename = "art_conversation.mp3" if st.session_state.get('analysis_mode') == "Conversational" else "art_analysis.mp3"
                st.download_button(
                    label="📥 Download Audio",
                    data=st.session_state.audio_data,
                    file_name=filename,
                    mime="audio/mpeg",
                    use_container_width=True
                )

    # Minimal footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <strong>Voice-Based Art Describer</strong> • Making visual art accessible through AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()