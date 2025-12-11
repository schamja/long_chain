from langchain_google_genai import ChatGoogleGenerativeAI
#Importiert die Klasse, die dir erlaubt, über LangChain mit Google Gemini zu sprechen.
from langchain_core.prompts import PromptTemplate
#Importiert PromptTemplate, damit du eine Vorlage (Template) für deinen Prompt erstellen kannst.
from dotenv import load_dotenv
#Importiert die Funktion, um Umgebungsvariablen aus einer .env-Datei zu laden.
import os
#Importiert das os-Modul – damit kannst du auf Umgebungsvariablen zugreifen.

load_dotenv()
#Liest die .env-Datei ein und lädt alle enthaltenen Variablen (z. B. GEMINI_API_KEY) in dein System.

print("GOOGLE_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
#Prüft, ob GEMINI_API_KEY existiertWenn ein API-Key vorhanden → True, Wenn nicht → False
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Explain the following topic for a beginner: {topic}. Please ensure the output is well-structured, uses paragraphs, and includes bullet points 
    for better readability, so that the text does not appear as a single long line.
    """
)

chain = prompt | llm

result = chain.invoke({"topic": "LangChain та LangGraph"})

print(result)
