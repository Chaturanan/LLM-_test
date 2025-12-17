"""
MCP Agent

An AI agent that connects to MCP (Model Context Protocol) servers to access
external tools and capabilities. Connect to any MCP server to extend what
the AI can do.
"""


# =========================================================
# IMPORTS (Libraries we need)
# =========================================================

# Streamlit: Framework for building web apps with Python
import streamlit as st

# asyncio: For running asynchronous code (required by MCP)
import asyncio

# os: For setting environment variables (API keys)
import os

# MultiServerMCPClient: Connects to multiple MCP servers at once
from langchain_mcp_adapters.client import MultiServerMCPClient

# create_react_agent: Creates an agent that can reason and use tools
from langgraph.prebuilt import create_react_agent

# ChatOpenAI: Connects to OpenAI's GPT models (like ChatGPT)
from langchain_openai import ChatOpenAI


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="MCP Agent",
    page_icon="🔧",
    layout="wide"  # Use full width of browser
)

st.title("🔧 MCP Agent")
st.caption("AI agent with MCP server integration")


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Store OpenAI API key

if "mcp_server_url" not in st.session_state:
    st.session_state.mcp_server_url = ""  # Store MCP server URL

if "mcp_agent" not in st.session_state:
    st.session_state.mcp_agent = None  # Store the agent instance

if "mcp_messages" not in st.session_state:
    st.session_state.mcp_messages = []  # Store chat history


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")
    
    if st.session_state.mcp_server_url:
        st.success("✅ MCP Server Connected")
        st.caption(f"URL: {st.session_state.mcp_server_url[:30]}...")
    else:
        st.warning("⚠️ MCP Server Not Connected")
    
    if st.session_state.openai_key or st.session_state.mcp_server_url:
        if st.button("Change API Keys"):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.mcp_server_url = ""
            st.session_state.mcp_agent = None
            st.rerun()


# =========================================================
# API KEYS INPUT
# =========================================================

# Check which keys we still need
keys_needed = []
if not st.session_state.openai_key:
    keys_needed.append("openai")
if not st.session_state.mcp_server_url:
    keys_needed.append("mcp")

if keys_needed:
    openai_key = st.session_state.openai_key
    server_url = st.session_state.mcp_server_url
    
    if "openai" in keys_needed:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",  # Hide the key as user types
            placeholder="sk-proj-..."
        )
    
    if "mcp" in keys_needed:
        server_url = st.text_input(
            "MCP Server URL",
            placeholder="http://localhost:3000"
        )
    
    if st.button("Connect"):
        valid = True
        
        # Validate OpenAI key format
        if "openai" in keys_needed:
            if not openai_key or not openai_key.startswith("sk-"):
                valid = False
        
        # Validate MCP server URL format
        if "mcp" in keys_needed:
            if not server_url or not (server_url.startswith("http://") or server_url.startswith("https://")):
                valid = False
        
        if valid:
            if "openai" in keys_needed:
                st.session_state.openai_key = openai_key
            if "mcp" in keys_needed:
                st.session_state.mcp_server_url = server_url
            st.rerun()  # Restart to show connected state
        else:
            st.error("❌ Invalid API key or URL format")
    
    st.stop()  # Don't show chat interface until connected


# =========================================================
# INITIALIZE MCP AGENT
# =========================================================

if not st.session_state.mcp_agent:
    
    with st.spinner("Initializing MCP agent..."):
        
        # Set API key as environment variable
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Configure MCP server
            mcp_config = {
                "server": {
                    "url": st.session_state.mcp_server_url,
                    "transport": "streamable_http",  # Use HTTP streaming
                }
            }
            
            # Create MCP client
            client = MultiServerMCPClient(mcp_config)
            
            # Get tools from server (async operation)
            tools = loop.run_until_complete(client.get_tools())
            
            # Create language model
            llm = ChatOpenAI(
                model="gpt-4o",  # Use GPT-4o
                temperature=0  # 0 = deterministic, 1 = creative
            )
            
            # Create agent with tools
            st.session_state.mcp_agent = create_react_agent(llm, tools)
            
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
            st.stop()
            
        finally:
            loop.close()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.mcp_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# HANDLE USER INPUT
# =========================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    # Save user message
    st.session_state.mcp_messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing with MCP tools..."):
            
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run agent (async operation)
                response = loop.run_until_complete(
                    st.session_state.mcp_agent.ainvoke({
                        "messages": st.session_state.mcp_messages
                    })
                )
                
                # Extract response text
                response_text = response["messages"][-1].content
                st.write(response_text)
                
                # Save to history
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                # Save error to history
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
            finally:
                loop.close()
