from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# now the reducer concept way to store the messages
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
def chat_node(state : ChatState):
    #  user query
    message = state['messages']
    # send to the llm
    response = llm.invoke(message)
    # again stor it in the state message
    return {'messages' : [response]}

graph = StateGraph(ChatState)

# CHECKPOINT
checkpoint = MemorySaver()

# now the edges and nodes
graph.add_node("chat_node", chat_node)

# edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)
