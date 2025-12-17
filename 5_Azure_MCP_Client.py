"""
MCP Agent - Azure Data Explorer Incident Analyst

An AI agent that connects to MCP (Model Context Protocol) servers to access
external tools and capabilities. This agent is specialized for Azure Data Explorer
KQL queries against incident data.

Supports multiple transport types:
- SSE (Server-Sent Events)
- Streamable HTTP
- stdio (for local processes)
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

# SystemMessage for passing the system prompt
from langchain_core.messages import SystemMessage


# =========================================================
# SYSTEM PROMPT - Azure Data Explorer Incident Analyst
# =========================================================

SYSTEM_PROMPT = """You are the Azure Data Explorer Incident Analyst.
Your role is to translate natural-language questions into valid KQL queries and execute them using the Kusto MCP tool.  
You answer ONLY using KQL executed through the tool unless the user explicitly requests an explanation.

Defaults (unless user overrides):
- Subscription: d27c6b88-6870-4df2-8b38-43c16f0f9d52
- Cluster URI: https://mcp-data-explorer.eastus2.kusto.windows.net
- Database: Azure_Issues
- Default table: Azure_Issues

Your primary responsibilities:
1. Retrieve, filter, and summarize incident data
2. Expand and analyze nested support tickets
3. Generate insights such as: impacted regions, time to mitigation, support ticket volume, service outage history
4. Always base results on KQL execution using the Data Explorer MCP connector

===============================================
1. ALWAYS CHECK SCHEMA BEFORE QUERYING
===============================================
Before using 'project', 'extend', or referencing any field:
Azure_Issues | getschema
or
Azure_Issues | take 1

This confirms the structure, especially the dynamic SupportTickets array.

Expected columns in Azure_Issues:
- IncidentId (string)
- CreateDate (datetime)
- MitigationDate (datetime)
- Regions (string)
- Title (string)
- Description (string)
- Service (string)
- SupportTickets (dynamic)

Do NOT invent columns. Use ONLY what exists.

===============================================
2. Working with nested SupportTickets
===============================================
To access support tickets, you MUST mv-expand the array.

Example expansion template:
Azure_Issues
| mv-expand SupportTickets
| project
    IncidentId,
    CaseNumber = SupportTickets.CaseNumber,
    Title = SupportTickets.Title,
    ProductName = SupportTickets.ProductName,
    SupportTopic = SupportTickets.SupportTopic,
    Description = SupportTickets.Description,
    IsTP = SupportTickets.IsTP

Always expand before filtering on ticket attributes.

===============================================
3. How to call the Kusto MCP Tool
===============================================
Always call KQL using:
{
  "name": "kusto",
  "args": {
    "command": "kusto_query",
    "parameters": {
      "cluster-uri": "https://mcp-data-explorer.eastus2.kusto.windows.net",
      "database": "Azure_Issues",
      "query": "<KQL_GOES_HERE>"
    }
  }
}

Rules:
- command MUST be "kusto_query"
- Always include cluster-uri, database, and query
- NEVER escape the query; write clean multiline KQL

===============================================
4. Query Templates
===============================================

--- A. Basic Incident Listing ---
Azure_Issues
| top 20 by CreateDate desc
| project IncidentId, CreateDate, MitigationDate, Service, Regions, Title

--- B. Search incidents by keyword ---
Azure_Issues
| where Title contains "storage" or Description contains "storage"
| project IncidentId, Title, CreateDate, Service, Regions

--- C. Expand and search support tickets ---
Azure_Issues
| mv-expand SupportTickets
| where SupportTickets.Description contains "timeout"
| project IncidentId,
    CaseNumber = SupportTickets.CaseNumber,
    TicketTitle = SupportTickets.Title,
    ProductName = SupportTickets.ProductName,
    SupportTopic = SupportTickets.SupportTopic

--- D. Time-to-mitigation calculation ---
Azure_Issues
| extend Duration = MitigationDate - CreateDate
| project IncidentId, Service, Regions, Duration
| top 20 by Duration desc

--- E. Count support tickets per incident ---
Azure_Issues
| extend TicketCount = array_length(SupportTickets)
| project IncidentId, Service, Regions, TicketCount
| order by TicketCount desc

===============================================
5. Error Handling
===============================================
If a query fails with:
- "Failed to resolve scalar expression 'X'"
- "Column does not exist"
- "Failed to resolve table"

Then:
1. Run:
   Azure_Issues | getschema
2. Identify correct column names
3. Retry the query using ONLY existing columns.

If SupportTickets fails to expand:
- Ensure mv-expand is used
- Ensure SupportTickets exists and is dynamic

===============================================
6. Operational Rules
===============================================
- Never invent data.
- Never answer from memory; always execute KQL unless user says "explain only".
- If user asks for a summary, run a query first, then summarize results.
- If user asks for a natural-language insight, run the query and then explain in plain English.
- If a user gives ambiguous terms (example: "tickets"), default to SupportTickets.
- If table name is not specified, always assume Azure_Issues.
"""


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Azure Data Explorer Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Azure Data Explorer Agent")
st.caption("AI agent specialized for KQL queries against Azure incident data")


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "mcp_server_url" not in st.session_state:
    st.session_state.mcp_server_url = ""

if "mcp_transport" not in st.session_state:
    st.session_state.mcp_transport = "sse"

if "mcp_agent" not in st.session_state:
    st.session_state.mcp_agent = None

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = []

if "mcp_messages" not in st.session_state:
    st.session_state.mcp_messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Connection Status Section
    st.subheader("📡 Connection Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.openai_key:
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    with col2:
        if st.session_state.mcp_server_url:
            st.success("✅ MCP")
        else:
            st.error("❌ MCP")
    
    # Show current configuration
    if st.session_state.mcp_server_url:
        st.divider()
        st.subheader("🔌 Server Details")
        st.text_input("URL", value=st.session_state.mcp_server_url, disabled=True, key="display_url")
        st.text_input("Transport", value=st.session_state.mcp_transport.upper(), disabled=True, key="display_transport")
    
    # Show available tools
    if st.session_state.mcp_tools:
        st.divider()
        st.subheader("🛠️ Available Tools")
        for tool in st.session_state.mcp_tools:
            with st.expander(f"📦 {tool.name}"):
                st.caption(tool.description if hasattr(tool, 'description') else "No description")
    
    # Reset button
    if st.session_state.openai_key or st.session_state.mcp_server_url:
        st.divider()
        if st.button("🔄 Reset Connection", use_container_width=True):
            st.session_state.openai_key = ""
            st.session_state.mcp_server_url = ""
            st.session_state.mcp_transport = "sse"
            st.session_state.mcp_agent = None
            st.session_state.mcp_client = None
            st.session_state.mcp_tools = []
            st.session_state.mcp_messages = []
            st.rerun()
    
    # Clear chat button
    if st.session_state.mcp_messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.mcp_messages = []
            st.rerun()


# =========================================================
# API KEYS INPUT (Only show if not connected)
# =========================================================

keys_needed = []
if not st.session_state.openai_key:
    keys_needed.append("openai")
if not st.session_state.mcp_server_url:
    keys_needed.append("mcp")

if keys_needed:
    st.markdown("---")
    st.subheader("🔐 Connect to Services")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    openai_key = st.session_state.openai_key
    server_url = st.session_state.mcp_server_url
    transport_type = st.session_state.mcp_transport
    
    with col1:
        st.markdown("#### OpenAI Configuration")
        
        if "openai" in keys_needed:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                help="Your OpenAI API key starting with 'sk-'"
            )
        else:
            st.success("✅ OpenAI API Key configured")
    
    with col2:
        st.markdown("#### MCP Server Configuration")
        
        if "mcp" in keys_needed:
            server_url = st.text_input(
                "MCP Server URL",
                placeholder="https://your-mcp-server.com/sse",
                help="The URL of your MCP server endpoint (e.g., Azure Data Explorer MCP)"
            )
            
            transport_type = st.selectbox(
                "Transport Type",
                options=["sse", "streamable_http"],
                index=0,
                help="Select the transport protocol your MCP server uses"
            )
            
            # Transport type descriptions
            transport_info = {
                "sse": "**SSE (Server-Sent Events)**: Best for HTTP servers that stream responses. Common for cloud-hosted MCP servers.",
                "streamable_http": "**Streamable HTTP**: For servers using the newer HTTP streaming transport protocol."
            }
            st.info(transport_info.get(transport_type, ""))
        else:
            st.success("✅ MCP Server configured")
    
    # Example configurations
    with st.expander("📋 Example Configurations"):
        st.markdown("""
        
        **Azure MCP Server (SSE)**
        ```
        URL: https://mcp-azure-agenticai-bootcamp.azurewebsites.net/sse
        Transport: sse
        ```
        """)
    
    st.markdown("---")
    
    # Connect button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Connect", use_container_width=True, type="primary"):
            errors = []
            
            # Validate OpenAI key
            if "openai" in keys_needed:
                if not openai_key:
                    errors.append("OpenAI API Key is required")
                elif not openai_key.startswith("sk-"):
                    errors.append("OpenAI API Key should start with 'sk-'")
            
            # Validate MCP server URL
            if "mcp" in keys_needed:
                if not server_url:
                    errors.append("MCP Server URL is required")
                elif not (server_url.startswith("http://") or server_url.startswith("https://")):
                    errors.append("MCP Server URL must start with http:// or https://")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                if "openai" in keys_needed:
                    st.session_state.openai_key = openai_key
                if "mcp" in keys_needed:
                    st.session_state.mcp_server_url = server_url
                    st.session_state.mcp_transport = transport_type
                st.rerun()
    
    st.stop()


# =========================================================
# INITIALIZE MCP AGENT
# =========================================================

if not st.session_state.mcp_agent:
    
    st.markdown("---")
    
    with st.status("🔄 Initializing Azure Data Explorer Agent...", expanded=True) as status:
        
        # Set API key
        st.write("Setting up OpenAI API key...")
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Configure MCP server
            st.write(f"Connecting to MCP server ({st.session_state.mcp_transport.upper()})...")
            st.write(f"URL: {st.session_state.mcp_server_url}")
            
            mcp_config = {
                "server": {
                    "url": st.session_state.mcp_server_url,
                    "transport": st.session_state.mcp_transport,
                }
            }
            
            # Create MCP client
            st.write("Creating MCP client...")
            client = MultiServerMCPClient(mcp_config)
            st.session_state.mcp_client = client
            
            # Get tools from server
            st.write("Fetching available tools...")
            tools = loop.run_until_complete(client.get_tools())
            st.session_state.mcp_tools = tools
            st.write(f"✅ Found {len(tools)} tools")
            
            # Create language model
            st.write("Initializing GPT-4o model...")
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0
            )
            
            # Create agent with system prompt
            st.write("Creating ReAct agent with Kusto KQL expertise...")
            st.session_state.mcp_agent = create_react_agent(
                llm, 
                tools,
                prompt=SYSTEM_PROMPT
            )
            
            status.update(label="✅ Azure Data Explorer Agent Ready!", state="complete", expanded=False)
            st.rerun()
            
        except Exception as e:
            status.update(label="❌ Initialization Failed", state="error")
            st.error(f"Error: {str(e)}")
            
            # Show troubleshooting tips
            with st.expander("🔍 Troubleshooting Tips"):
                st.markdown("""
                **Common Issues:**
                
                1. **Connection Refused**: Make sure the MCP server is running and accessible
                2. **Invalid URL**: Check the URL format and ensure it's correct
                3. **Wrong Transport**: Try switching between 'sse' and 'streamable_http'
                4. **Firewall Issues**: Ensure your network allows connections to the server
                5. **API Key Invalid**: Verify your OpenAI API key is correct and active
                
                **For Azure Data Explorer MCP Servers:**
                - Ensure the server endpoint is accessible
                - Check if the URL ends with `/sse` for SSE transport
                - Verify your Azure credentials/authentication
                """)
            
            # Reset button
            if st.button("🔄 Try Again"):
                st.session_state.mcp_agent = None
                st.session_state.mcp_client = None
                st.session_state.mcp_tools = []
                st.rerun()
            
            st.stop()
            
        finally:
            loop.close()


# =========================================================
# CHAT INTERFACE
# =========================================================

st.markdown("---")

# Welcome message if no chat history
if not st.session_state.mcp_messages:
    st.markdown("""
    ### 👋 Welcome to Azure Data Explorer Agent!
    
    I'm your specialized KQL query assistant connected to Azure Data Explorer. I can help you:
    
    - **Query incident data** from the Azure_Issues table
    - **Analyze support tickets** with nested data expansion
    - **Generate insights** on impacted regions, time to mitigation, and more
    - **Search and filter** incidents by keywords, dates, and services
    
    **Example queries you can try:**
    - "Show me the latest 10 incidents"
    - "Find all storage-related incidents"
    - "What's the average time to mitigation?"
    - "List incidents with the most support tickets"
    - "Show incidents affecting the East US region"
    
    **Type a message below to get started!**
    """)

# Display chat history
for message in st.session_state.mcp_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# HANDLE USER INPUT
# =========================================================

user_input = st.chat_input("Ask about Azure incidents... I'll write and execute KQL queries!")

if user_input:
    # Add user message to history
    st.session_state.mcp_messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing and executing KQL query..."):
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Prepare messages for agent (include system prompt)
                agent_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.mcp_messages
                ]
                
                # Run agent
                response = loop.run_until_complete(
                    st.session_state.mcp_agent.ainvoke({
                        "messages": agent_messages
                    })
                )
                
                # Extract response
                response_text = response["messages"][-1].content
                st.markdown(response_text)
                
                # Save to history
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
            except Exception as e:
                error_msg = f"❌ Error processing request: {str(e)}"
                st.error(error_msg)
                
                # Add error to history
                st.session_state.mcp_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
            finally:
                loop.close()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🔌 Transport: {st.session_state.mcp_transport.upper()}")
with col2:
    st.caption("📊 Database: Azure_Issues")
with col3:
    st.caption("MCP Client")