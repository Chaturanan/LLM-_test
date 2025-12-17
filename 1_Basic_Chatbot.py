"""
Basic Chatbot with LangGraph

A simple conversational chatbot that responds to your messages naturally.
This is your starting point - the solution will show you how to add intelligent state management!
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Streamlit: Framework for building web apps
import streamlit as st

# ChatOpenAI: Connects to OpenAI's GPT models
from langchain_openai import ChatOpenAI

# Message types for conversation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangGraph: For building AI workflows
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Basic Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Basic Chatbot")
st.caption("A friendly AI assistant that chats with you")


# =============================================================================
# SESSION STATE
# =============================================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
        
        if st.button("Change API Keys"):
            st.session_state.openai_key = ""
            st.session_state.llm = None
            st.session_state.chatbot = None
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")


# =============================================================================
# API KEY INPUT
# =============================================================================

if not st.session_state.openai_key:
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-..."
    )
    
    if st.button("Connect"):
        if api_key and api_key.startswith("sk-"):
            st.session_state.openai_key = api_key
            st.rerun()
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()


# =============================================================================
# INITIALIZE AI
# =============================================================================

if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=st.session_state.openai_key
    )


# =============================================================================
# CREATE SIMPLE CHATBOT
# =============================================================================

if st.session_state.llm and not st.session_state.chatbot:
    
    # Define conversation state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    # Create chatbot node
    def chatbot_node(state: State):
        """Simple chatbot that responds naturally to messages"""
        
        # Add a friendly system message
        system_msg = SystemMessage(
            content="You are a helpful and friendly AI assistant. Have natural conversations with users."
        )
        
        messages = [system_msg] + state["messages"]
        response = st.session_state.llm.invoke(messages)
        
        return {"messages": [response]}
    
    # Build workflow
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    # Compile and save
    st.session_state.chatbot = workflow.compile()


# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =============================================================================
# HANDLE USER INPUT
# =============================================================================

user_input = st.chat_input("Type your message...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build message history
            messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Run chatbot
            result = st.session_state.chatbot.invoke({"messages": messages})
            response = result["messages"][-1].content
            
            # Display and save response
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
