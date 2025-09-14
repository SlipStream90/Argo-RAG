import streamlit as st
import time
import re
from typing import List, Dict, Tuple, Optional
import json
from RAG_main import main

# RAG Output Cleaning Functions
def clean_rag_output(raw_output: str) -> str:
    """
    Clean the RAG system output by removing internal thinking processes and formatting the response.
    
    Args:
        raw_output (str): The raw output from the RAG system containing internal reasoning
        
    Returns:
        str: Clean, user-friendly response
    """
    
    # Handle tuple output (output, score) format
    if isinstance(raw_output, tuple):
        output_text = raw_output[0] if raw_output[0] else ""
        confidence_score = raw_output[1] if len(raw_output) > 1 else None
    else:
        output_text = str(raw_output)
        confidence_score = None
    
    # Remove common internal reasoning patterns
    patterns_to_remove = [
        # Remove detailed parsing explanations
        r"First, the question is:.*?I need to look for.*?\n\n",
        r"The question is about.*?That's exact\.\n\n",
        r"Let me confirm.*?\n\n",
        r"So, for.*?exact match\.\n\n",
        
        # Remove field parsing explanations
        r"The data structure is:.*?\n\n",
        r"In the context data, it's listed as:.*?\n\n",
        r"I should parse this carefully\..*?\n\n",
        r"Let me write it out:.*?\n\n",
        r"The description says:.*?\n\n",
        
        # Remove step-by-step reasoning
        r"Let me list.*?\n\n",
        r"I think the fields are:.*?\n\n",
        r"Similarly.*?\n\n",
        r"The string is:.*?\n\n",
        
        # Remove uncertainty and validation thoughts
        r"The \[Other\] field is.*?I'm not sure.*?\n\n",
        r"But for the answer.*?\n\n",
        r"To be precise.*?\n\n",
        r"I think for conciseness.*?\n\n",
        
        # Remove format explanations
        r"The context data has dates in.*?\n\n",
        r"In the data, it's listed as.*?\n\n",
        
        # Remove numbered lists of thinking
        r"\d+\.\s+.*?:\s+\d+.*?\n",
        
        # Remove validation statements
        r"Now, I need to extract.*?\n\n",
        r"Since the date is exact.*?\n\n",
        r"The question asks for.*?\n\n",
    ]
    
    cleaned_text = output_text
    
    # Apply regex patterns to remove internal reasoning
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove excessive newlines and whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text)
    
    # Extract the actual answer part
    # Look for patterns that indicate the start of the actual answer
    answer_indicators = [
        r"For \d+/\d+/\d+.*?:",
        r"For \d{4}-\d{2}-\d{2}.*?:",
        r"Measurements:",
        r"Details for",
        r"The data for",
        r"On \d+/\d+/\d+"
    ]
    
    # Try to find where the actual answer starts
    answer_start = 0
    for indicator in answer_indicators:
        match = re.search(indicator, cleaned_text, re.IGNORECASE)
        if match:
            answer_start = match.start()
            break
    
    if answer_start > 0:
        cleaned_text = cleaned_text[answer_start:]
    
    # Format the final answer nicely
    cleaned_text = format_ocean_data_response(cleaned_text)
    
    return cleaned_text.strip()


def format_ocean_data_response(text: str) -> str:
    """
    Format ocean data responses to be more user-friendly.
    """
    # Ensure proper units are displayed
    unit_replacements = {
        r'(\d+\.?\d*)\s*°C': r'\1°C',
        r'(\d+\.?\d*)\s*PSU': r'\1 PSU',
        r'(\d+\.?\d*)\s*meters?': r'\1 meters',
        r'(\d+\.?\d*)\s*m\b': r'\1 meters',
        r'(\d+\.?\d*)\s*decibars?': r'\1 decibars',
        r'(\d+\.?\d*)\s*db\b': r'\1 decibars',
    }
    
    formatted_text = text
    for pattern, replacement in unit_replacements.items():
        formatted_text = re.sub(pattern, replacement, formatted_text)
    
    # Clean up coordinate formatting
    formatted_text = re.sub(r'Latitude:\s*(-?\d+\.?\d*).*?\(South\)', r'Latitude: \1° S', formatted_text)
    formatted_text = re.sub(r'Latitude:\s*(-?\d+\.?\d*).*?\(North\)', r'Latitude: \1° N', formatted_text)
    formatted_text = re.sub(r'Longitude:\s*(-?\d+\.?\d*).*?\(East\)', r'Longitude: \1° E', formatted_text)
    formatted_text = re.sub(r'Longitude:\s*(-?\d+\.?\d*).*?\(West\)', r'Longitude: \1° W', formatted_text)
    
    # Handle negative coordinates
    formatted_text = re.sub(r'Latitude:\s*(-\d+\.?\d*)°', r'Latitude: \1° S', formatted_text)
    formatted_text = re.sub(r'Longitude:\s*(-\d+\.?\d*)°', r'Longitude: \1° W', formatted_text)
    
    return formatted_text


def clean_and_format_ocean_response(query: str, raw_response: str) -> str:
    """
    Main function to clean and format ocean data responses.
    
    Args:
        query (str): The original user query
        raw_response (str): Raw response from RAG system
        
    Returns:
        str: Clean, formatted response
    """
    # Clean the output
    cleaned = clean_rag_output(raw_response)
    
    # Add context if the response is very short or unclear
    if len(cleaned.strip()) < 50:
        return f"I found limited information for your query: '{query}'. {cleaned}"
    
    return cleaned

# Streamlit App Configuration
st.set_page_config(page_title="FloatChat", layout="wide")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Custom CSS for the perfected layout and styling
st.markdown("""
    <style>
    /* Main background gradient for a smoother, glowing effect */
    .stApp {
        background-color: #0f172a; /* Fallback for older browsers */
        background: radial-gradient(circle at center, #1e3a8a 0%, #0f172a 70%);
    }

    /* General text and font styles */
    h1, h2, h3, h4, h5, h6, p, li, a {
        color: white;
        font-family: 'Inter', sans-serif;
    }

    /* Adjust Streamlit's main content wrapper */
    .main .block-container {
        max-width: 1280px;
        padding-top: 5rem;
        padding-bottom: 5rem;
    }

    /* Styles for the fixed navigation bar */
    .stApp > header {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        position: fixed;
        width: 100%;
        top: 0;
        z-index: 1000;
        padding: 0;
    }

    /* Ensure Streamlit's internal blocks don't add extra space */
    div[data-testid="stVerticalBlock"] { gap: 1.5rem; }
    div[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
    div[data-testid="stColumn"] { gap: 1.5rem; }

    /* Specific CSS for the main hero heading */
    .hero-heading {
        text-align: center;
        margin-bottom: 20px;
    }

    /* Download button style */
    .stDownloadButton > button {
        background-image: linear-gradient(to right, #3b82f6, #22d3ee);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-image: linear-gradient(to right, #2563eb, #0ea5e9);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Glassmorphism card effect */
    .card-container {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    .card-container:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.4);
    }
    .metric-card { text-align: center; }
    .metric-value { font-size: 40px; font-weight: bold; margin-bottom: 5px; color: white; }
    .metric-label { font-size: 16px; opacity: 0.7; color: white; }
    .feature-card { text-align: left; padding: 30px; color: white; }
    .feature-card h5 { font-size: 18px; font-weight: bold; margin-top: 15px; }
    .feature-card p { opacity: 0.8; }
    .feature-icon-circle {
        background: linear-gradient(to right, #3b82f6, #22d3ee);
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
    }

    .quick-query-button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        width: 100%;
        text-align: left;
        transition: all 0.2s ease;
    }
    .quick-query-button:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border-color: #22d3ee;
    }
    /* FIX: Style the text input to match your design */
    div[data-testid="stTextInput"] > div > div > input {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: none !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 25px !important;
        padding: 10px 15px !important;
        flex-grow: 1 !important;
    }

    /* Style the submit button to match background */
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        font-weight: bold !important;
        border-radius: 25px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
        height: fit-content !important;
    }

    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.6) !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* Make the form elements align horizontally like your original HTML */
    div[data-testid="stForm"] > div {
        display: flex !important;
        gap: 10px !important;
        align-items: center !important;
    }

    div[data-testid="stForm"] > div:last-child { 
        background-color: transparent !important; 
    }

    div[data-testid="stForm"] { 
        background-color: transparent !important; 
        margin-top: 10px !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > div > div {
        background-color: transparent !important;
    }

    /* ✅ Custom hero button */
    .hero-gradient-button {
        display: inline-block;
        padding: 1rem 2.5rem;
        font-size: 1.25rem;
        font-weight: 600;
        border-radius: 50px;
        text-decoration: none;
        background-image: linear-gradient(to right, #3b82f6, #22d3ee);
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .hero-gradient-button:hover {
        transform: scale(1.05);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
    }

    /* Chat message styling */
    .chat-message {
        margin-bottom: 15px;
        animation: fadeIn 0.5s ease-in;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
    }
    .bot-message {
        display: flex;
        justify-content: flex-start;
    }
    .message-bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 18px;
        word-wrap: break-word;
        line-height: 1.4;
    }
    .user-bubble {
        background: linear-gradient(to right, #3b82f6, #22d3ee);
        color: white;
    }
    .bot-bubble {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .message-time {
        font-size: 0.7em;
        opacity: 0.8;
        margin-top: 5px;
        display: block;
    }
    .user-time {
        text-align: right;
    }
    .bot-time {
        text-align: left;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        max-width: 70%;
    }
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22d3ee;
        animation: typing 1.5s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.3s; }
    .typing-dot:nth-child(3) { animation-delay: 0.6s; }
    @keyframes typing {
        0%, 60%, 100% { opacity: 0.3; transform: scale(0.8); }
        30% { opacity: 1; transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Function to handle quick query buttons
def handle_quick_query(query: str):
    """Handle quick query button clicks"""
    st.session_state.chat_history.append({
        "type": "user",
        "message": query,
        "timestamp": time.strftime("%I:%M %p")
    })
    st.session_state.is_processing = True

# Function to process user query
def process_query(query: str) -> str:
    """Process user query through RAG system with cleaned output"""
    try:
        # Call your main RAG function
        raw_answer = main(query)
        
        # Clean and format the response
        clean_answer = clean_and_format_ocean_response(query, raw_answer)
        
        return clean_answer
    except Exception as e:
        return f"🚫 Sorry, I encountered an error processing your query: {str(e)}"

# ---- Navigation Bar ---- (FIXED: Removed duplicate)
with st.container():
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 64px;
            max-width: 1280px;
            margin: auto;
            padding: 0 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="display: flex; align-items: center; justify-content: center; width: 40px; height: 40px; background: linear-gradient(to right, #3b82f6, #22d3ee); border-radius: 12px;">
                </div>
                <div>
                    <h1 style="font-size: 1.25rem; font-weight: bold; color: white; margin: 0;">FloatChat</h1>
                    <p style="font-size: 0.75rem; color: #bfdbfe; margin: 0;">AI Ocean Intelligence</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 32px;">
                <button class="stButton" style="color: white; border: none; background: none;">Home</button>
                <button class="stButton" style="color: white; border: none; background: none;">Features</button>
                <button class="stButton" style="color: white; border: none; background: none;">Demo</button>
                <button class="stButton" style="color: white; border: none; background: none;">Impact</button>
                <button class="stButton hero-gradient-button" style="padding: 8px 24px; font-size: 14px; border-radius: 9999px; text-decoration: none;">Get Started</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- Hero Section ----
st.markdown("<div style='text-align: center; margin-top: 100px; padding: 20px;'>", unsafe_allow_html=True)
st.markdown("""
    <div class="hero-heading">
        <h1 style='font-size: 5em; font-weight: 800; line-height: 1.2; margin-bottom: 20px;'>Meet Your AI 
            <span style='background: linear-gradient(to right, #60a5fa, #4fd1c5); -webkit-background-clip: text; color: transparent;'>Ocean</span> Expert
        </h1>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 1.5em; max-width: 950px; margin: auto; opacity: 0.8; line-height: 1.6;'>FloatChat transforms complex oceanographic data into instant insights. Ask questions in plain English, get expert analysis powered by real-time satellite data and AI.</p>",
    unsafe_allow_html=True)

# ✅ Centered custom button (fixed)
st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <a href="#demo" class="hero-gradient-button" style="color:white ; text-decoration:none">Try Live Demo</a>
    </div>
""", unsafe_allow_html=True)

# ---- Stats Section ----
st.markdown("<div style='padding-top: 80px; padding-bottom: 20px;'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        "<div class='card-container metric-card'><div class='metric-value'>50M+</div><div class='metric-label'>Data Points</div></div>",
        unsafe_allow_html=True)
with col2:
    st.markdown(
        "<div class='card-container metric-card'><div class='metric-value'>24/7</div><div class='metric-label'>Real-time Updates</div></div>",
        unsafe_allow_html=True)
with col3:
    st.markdown(
        "<div class='card-container metric-card'><div class='metric-value'>95%</div><div class='metric-label'>Accuracy Rate</div></div>",
        unsafe_allow_html=True)
with col4:
    st.markdown(
        "<div class='card-container metric-card'><div class='metric-value'>12</div><div class='metric-label'>Data Sources</div></div>",
        unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Features Section ----
st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
col_features_1, col_features_2, col_features_3 = st.columns(3)
with col_features_1:
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Natural Language Queries</h5><p>Ask complex questions in plain English.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Research-Grade Accuracy</h5><p>Validated with 95%+ accuracy rates.</p></div>""", unsafe_allow_html=True)
with col_features_2:
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Real-time Satellite Data</h5><p>Access live oceanographic data.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Lightning Fast</h5><p>Get analysis results in seconds.</p></div>""", unsafe_allow_html=True)
with col_features_3:
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Instant Visualizations</h5><p>Generate interactive charts instantly.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="card-container feature-card"><div class="feature-icon-circle"></div><h5>Marine Life Tracking</h5><p>Monitor ecosystems and biodiversity.</p></div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Live Demo Chat Section ----
st.markdown("<div id='demo' style='margin-top: 80px; text-align: center; padding: 20px;'>", unsafe_allow_html=True)
st.markdown("<h2 style='font-weight: 600; margin-bottom: 15px;'>Try FloatChat Live</h2>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='opacity: 0.8; max-width: 600px; margin: auto;'>Experience the power of AI-driven ocean analysis. Ask any question about our oceans.</h5>",
    unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
chat_col1, chat_col2 = st.columns([1, 2])

with chat_col1:
    st.markdown("""
        <div class="card-container" style="padding: 20px;">
            <p style='font-weight: bold; font-size: 1.2em; opacity: 0.8;'>Quick Queries</p>
            <div style="margin-top: 15px;">
    """, unsafe_allow_html=True)
    
    # Quick query buttons
    quick_queries = [
        "Show me global sea temperature anomalies",
        "Analyze coral reef health indicators",
        "Compare Arctic ice coverage trends",
        "Find optimal fishing zones",
        "Track hurricane formation patterns"
    ]
    
    for query in quick_queries:
        if st.button(query, key=f"quick_{query}", use_container_width=True):
            handle_quick_query(query)
            st.rerun()
    
    st.markdown("""
            </div>
            <div class="card-container" style="margin-top: 20px; background-color: rgba(59, 130, 246, 0.2); border-color: rgba(59, 130, 246, 0.5);">
                <p style='font-weight: bold; font-size: 1em; color: white;'>Live Data Sources</p>
                <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                    <p style="margin-bottom: 5px;">✅ NOAA Satellites</p>
                    <p style="margin-bottom: 5px;">✅ ARGO Float Network</p>
                    <p style="margin-bottom: 5px;">✅ ESA Sentinel</p>
                    <p style="margin-bottom: 5px;">✅ NASA Ocean Color</p>
                    <p>✅ World Ocean Database</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with chat_col2:
    st.markdown("""
        <div class="card-container" style="min-height: 400px; position: relative;">
            <div style="padding: 20px; border-radius: 10px; margin-bottom: 20px;" class="gradient-bg">
                <p style="font-weight: bold;">FloatChat AI</p>
                <p style="font-style: italic; opacity: 0.9;">Your Ocean Data Expert</p>
                <p style="margin-top: 10px;">Welcome to FloatChat! I'm your AI oceanographer. Ask me about sea temperatures, marine ecosystems, climate patterns, or any ocean data. What would you like to explore?</p>
            </div>
    """, unsafe_allow_html=True)
    
    # Chat history container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div style="height: 300px; overflow-y: auto; margin-bottom: 20px; padding: 10px;">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-bubble user-bubble">
                            <p style="font-size: 0.9em; margin: 0;">{message["message"]}</p>
                            <span class="message-time user-time">{message["timestamp"]}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-bubble bot-bubble">
                            <p style="font-size: 0.9em; margin: 0;">{message["message"]}</p>
                            <span class="message-time bot-time">{message["timestamp"]}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Show typing indicator if processing
        if st.session_state.is_processing:
            st.markdown("""
                <div class="chat-message bot-message">
                    <div class="typing-indicator">
                        <span style="margin-right: 8px;">FloatChat is thinking</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input("", placeholder="Ask me anything about ocean data... 🌊", label_visibility="collapsed")
        
        with col_button:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        
        # Process form submission
        if submit_button and user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({
                "type": "user",
                "message": user_input.strip(),
                "timestamp": time.strftime("%I:%M %p")
            })
            st.session_state.is_processing = True
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Process query if there's a pending one
if st.session_state.is_processing and len(st.session_state.chat_history) > 0:
    last_message = st.session_state.chat_history[-1]
    if last_message["type"] == "user":
        # Process the query through RAG system
        response = process_query(last_message["message"])
        
        # Add bot response to chat history
        st.session_state.chat_history.append({
            "type": "bot",
            "message": response,
            "timestamp": time.strftime("%I:%M %p")
        })
        
        st.session_state.is_processing = False
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer Section ----
st.markdown("<div style='margin-top: 80px; padding: 20px;'>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin-bottom: 40px;'>", unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns([2, 1, 1, 1])
with footer_col1:
    st.markdown("""
        <div>
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; justify-content: center; width: 40px; height: 40px; background: linear-gradient(to right, #3b82f6, #22d3ee); border-radius: 12px;">
                    <img src="https://i.imgur.com/uW24WvE.png" style="width: 24px; height: 24px;" />
                </div>
                <div>
                    <h3 style="font-weight: 600; margin: 0;">FloatChat</h3>
                    <p style="font-size: 0.8em; color: #bfdbfe; margin: 0;">AI Ocean Intelligence</p>
                </div>
            </div>
            <p style="font-size: 0.9em; opacity: 0.7; max-width: 400px; line-height: 1.5;">Democratizing ocean data analysis through AI. Making complex oceanographic insights accessible to researchers, policymakers, and conservationists worldwide.</p>
            <p style="font-size: 0.9em; opacity: 0.7; margin-top: 10px;">Built with ❤️ for our blue planet</p>
        </div>
    """, unsafe_allow_html=True)
with footer_col2:
    st.markdown("""
        <div>
            <h4 style="font-weight: 600;'>Product</h4>
            <ul style="list-style-type: none; padding-left: 0; opacity: 0.7; line-height: 2;">
                <li>Features</li>
                <li>API Access</li>
                <li>Pricing</li>
                <li>Documentation</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
with footer_col3:
    st.markdown("""
        <div>
            <h4 style="font-weight: 600;'>Data Partners</h4>
            <ul style="list-style-type: none; padding-left: 0; opacity: 0.7; line-height: 2;">
                <li>NOAA</li>
                <li>NASA Ocean Color</li>
                <li>ESA Copernicus</li>
                <li>ARGO Network</li>
                <li>World Ocean Database</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
with footer_col4:
    st.markdown("""
        <div>
            <h4 style="font-weight: 600;'>Legal</h4>
            <ul style="list-style-type: none; padding-left: 0; opacity: 0.7; line-height: 2;">
                <li>Privacy Policy</li>
                <li>Terms of Service</li>
                <li>Contact</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='text-align: center; font-size: 0.8em; opacity: 0.5; margin-top: 40px;'>", unsafe_allow_html=True)
st.markdown("© 2024 FloatChat. All rights reserved.", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
