from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# Create a PromptTemplate with input_variables
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)

# Create the chain by piping the prompt and LLM
chain = prompt | llm

# ------------ Main Interaction Loop ----------------
print("Welcome to the LangChain Gemini Chatbot!")
print("Type 'exit' to quit the chatbot.")

while True:
    # Get user input
    user_question = input("You: ")

    # Check if the user wants to exit
    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    # Get the AI's response using the modern .invoke() method
    # The input is passed as a dictionary {"question": ...}
    response = chain.invoke({"question": user_question})

    # Print the response
    print(f"AI: {response.content}")
