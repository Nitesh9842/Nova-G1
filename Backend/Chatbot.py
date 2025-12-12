from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (works for both local .env and production)
env_vars = dotenv_values(".env")

# Get variables with fallback to os.environ for production (Render)
Username = env_vars.get("Username") or os.getenv("Username", "Developer")
Assistantname = env_vars.get("Assistantname") or os.getenv("Assistantname", "AI Assistant")
GroqAPIKey = env_vars.get("GroqAPIKey") or os.getenv("GROQ_API_KEY")

# Validate API key
if not GroqAPIKey:
    raise ValueError("API Key not found! Set 'GroqAPIKey' in .env or 'GROQ_API_KEY' in environment variables.")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages = []

System = f"""Hello, I am {Username}, I am student of B.Tech CSE AI&ML in Shri Vishwkarma Skill University. You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** If someone asks you 'who is your developer', 'who made you', 'who is your creator', you should say 'I was created by {Username}' and tell about me only.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Always give the answer in easy language and give youtube links for specific topics, never mention your training data.***
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Always provide answer in formatted way, start heading always next line and add colour, emoji to give attractive look.***
"""

systemChatBot = [
    {"role": "system", "content": System}
]

# Ensure Data directory exists
os.makedirs("Data", exist_ok=True)

# Load or create chat log (FIXED: consistent filename)
CHAT_LOG_PATH = "Data/ChatLog.json"

try:
    with open(CHAT_LOG_PATH, "r") as f:
        messages = load(f)
except FileNotFoundError:
    with open(CHAT_LOG_PATH, "w") as f:
        dump([], f)
    messages = []
except Exception as e:
    logger.warning(f"Error loading chat log: {e}. Starting fresh.")
    messages = []


def RealtimeInformation():
    """Get current date and time information"""
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")

    # FIXED: was using 'date' instead of 'data'
    data = f"Please use this real-time information if needed:\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours : {minute} minutes : {second} seconds.\n"
    return data


def AnswerModifier(Answer):
    """Remove empty lines from answer"""
    # FIXED: variable name conflict in list comprehension
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer


def ChatBot(Query):
    """Send user's query to the chatbot and return response"""
    try:
        # Load latest messages
        with open(CHAT_LOG_PATH, "r") as f:
            messages = load(f)

        # Add user query
        messages.append({"role": "user", "content": Query})

        # Get completion from Groq
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=systemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        # Collect response
        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        # Clean up answer
        Answer = Answer.replace("</s>", "")

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": Answer})

        # Save updated messages
        with open(CHAT_LOG_PATH, "w") as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer=Answer)

    except Exception as e:
        logger.error(f"Error in ChatBot: {e}")
        # Don't recursively call ChatBot - return error message instead
        return f"Sorry, I encountered an error: {str(e)}. Please try again."


if __name__ == "__main__":
    print(f"Chatbot '{Assistantname}' is ready! (Type 'exit' to quit)\n")
    while True:
        user_input = input("Enter your Question: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        response = ChatBot(user_input)
        print(f"\n{Assistantname}: {response}\n")
