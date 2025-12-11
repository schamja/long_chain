from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithMessageHistory  # Modern memory
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, \
    ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Create a chat history memory
history = InMemoryChatMessageHistory()

# Create prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"), # Automatically inserts chat history
    ("human", "{input}")  # User input placeholder
])

# Create the chain
chain = prompt | llm  # Pipe operator

conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)
print("Chatbot with memory started!")
print("Type 'exit' to end.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "student_chat"}}
    )

    print(f"Bot: {response.content}\n")




















