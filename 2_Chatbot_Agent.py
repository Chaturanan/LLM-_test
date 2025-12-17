"""
Chatbot Agent

An AI agent that can search the web using Tavily to answer your questions
with current, real-time information.
"""


# =========================================================
# IMPORTS (Libraries we need)
# =========================================================

# Streamlit: Framework for building web apps with Python
import streamlit as st

# os: For setting environment variables (API keys)
import os

# ChatOpenAI: Connects to OpenAI's GPT models (like ChatGPT)
from langchain_openai import ChatOpenAI

# TavilySearchResults: Tool for searching the web
from langchain_community.tools.tavily_search import TavilySearchResults

# create_react_agent: Creates an agent that can reason and use tools
from langgraph.prebuilt import create_react_agent


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Chatbot Agent",
    page_icon="🔍",
    layout="wide"  # Use full width of browser
)

st.title("🔍 Chatbot Agent")
st.caption("AI agent with web search capabilities")


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Store OpenAI API key

if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = ""  # Store Tavily API key

if "agent" not in st.session_state:
    st.session_state.agent = None  # Store the agent instance

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # Store chat history


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")
    
    if st.session_state.tavily_key:
        st.success("✅ Tavily Connected")
    else:
        st.warning("⚠️ Tavily Not Connected")
    
    if st.session_state.openai_key or st.session_state.tavily_key:
        if st.button("Change API Keys"):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.tavily_key = ""
            st.rerun()


# =========================================================
# API KEYS INPUT
# =========================================================

# Check which keys we still need
keys_needed = []
if not st.session_state.openai_key:
    keys_needed.append("openai")
if not st.session_state.tavily_key:
    keys_needed.append("tavily")

if keys_needed:
    openai_key = st.session_state.openai_key
    tavily_key = st.session_state.tavily_key
    
    if "openai" in keys_needed:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",  # Hide the key as user types
            placeholder="sk-proj-..."
        )
    
    if "tavily" in keys_needed:
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",  # Hide the key as user types
            placeholder="tvly-..."
        )
    
    if st.button("Connect"):
        valid = True
        
        # Validate OpenAI key format
        if "openai" in keys_needed:
            if not openai_key or not openai_key.startswith("sk-"):
                valid = False
        
        # Validate Tavily key format
        if "tavily" in keys_needed:
            if not tavily_key or not tavily_key.startswith("tvly-"):
                valid = False
        
        if valid:
            if "openai" in keys_needed:
                st.session_state.openai_key = openai_key
            if "tavily" in keys_needed:
                st.session_state.tavily_key = tavily_key
            st.rerun()  # Restart to show connected state
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()  # Don't show chat interface until connected


# =========================================================
# CREATE AGENT
# =========================================================

if not st.session_state.agent:
    # Set API keys as environment variables (required by some tools)
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    os.environ["TAVILY_API_KEY"] = st.session_state.tavily_key
    
    # Create language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use GPT-4o-mini (fast and cheap)
        temperature=0  # 0 = deterministic, 1 = creative
    )
    
    # Create Tavily search tool
    search_tool = TavilySearchResults(max_results=3)
    
    # Create agent with search capability
    st.session_state.agent = create_react_agent(llm, [search_tool])


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# HANDLE USER INPUT
# =========================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat history
    st.session_state.agent_messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            # Invoke agent with all messages
            response = st.session_state.agent.invoke({
                "messages": st.session_state.agent_messages
            })
            
            # Extract response text from agent output
            response_text = response["messages"][-1].content
            st.write(response_text)
            
            # Add assistant response to chat history
            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": response_text
            })
