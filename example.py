import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')
print("🤖 Chatbot: Hello! I am powered by Gemini. (Type 'exit' to end)")
print("-" * 60)
while True:
    user_input = input("👤 You: ")
    if user_input.lower() == 'exit': 
        print("🤖 Chatbot: Goodbye!")
        break
    response = model.generate_content(user_input)
    print(f"🤖 Chatbot: {response.text}")