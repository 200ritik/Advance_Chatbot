from agentic_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("LangGraph Agentic Chatbot")

CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

# Create session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_input = st.chat_input("Ask anything")

if user_input:
    # Store user message in Streamlit session
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Send message to LangGraph
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=CONFIG
    )

    ai_message = response["messages"][-1].content

    # Store AI response in Streamlit session
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_message
    })

    with st.chat_message("assistant"):
        st.write(ai_message)