from googlesearch import search
from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values
import requests
import socket
import urllib3


env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")


client = Groq(api_key=GroqAPIKey)


System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** If some one ask you 'who is your dovleper' ,'who made you', 'who is your creator' , you should say 'I was created by {Username} and tell about me only.'***
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""


try:
    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)
except:
    with open(r"Data/Chatlog.json", "w" ) as f:
        dump([], f)


def GoogleSearch(query):
    try:
        # Add timeout and retries
        results = list(search(
            query, 
            advanced=True, 
            num_results=5,
            timeout=10
        ))
        Answer = f"the search results for '{query}' are :\n [start]\n"

        for i in results:
            Answer += f"Title: {i.title}\nDescription: {i.description}\n\n" # type: ignore

        Answer += "[end]"
        return Answer
    except (requests.exceptions.ConnectionError, 
            requests.exceptions.Timeout,
            socket.gaierror,
            urllib3.exceptions.MaxRetryError) as e:
        # Fallback response when network fails
        return "I apologize, but I'm having trouble connecting to search services right now. Please check your internet connection and try again."




def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [lines for lines in lines if lines.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer  


systemChatBot = [
    {"role": "system", "content": System},
     {"role": "system", "content": "Hi"}, 
     {"role": "system", "content": "Hello, how can I help you"}

]     


def Information():
    data = ""
    current_date_time = datetime.datetime.now()
    day =  current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year =  current_date_time.strftime("%Y") 
    hour =  current_date_time.strftime("%H") 
    minute =  current_date_time.strftime("%M")
    second =  current_date_time.strftime("%S")
    data = f"Use This Real-time Information if needed:\n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second}  seconds.\n"
    return data


def RealtimeSearchEngine(prompt):
    global systemChatBot, messages

    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)

    messages.append({"role": "user", "content": f"{prompt}"})

    systemChatBot.append({"role": "system", "content": GoogleSearch(prompt)})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=systemChatBot + [{"role": "system", "content": Information()}] + messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=1,
        stream=True,
        stop=None
    )

    Answer = ""    



    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content

    Answer = Answer.strip().replace("</s>","")
    messages.append({"role": "assistant", "content": Answer})

    with open(r"Data/ChatLog.json", "w") as f:
        dump(messages, f, indent=4)

    systemChatBot.pop()
    return AnswerModifier(Answer=Answer)

# Add this before running the main loop to test connectivity
def test_connection():
    try:
        requests.get('https://www.google.com', timeout=5)
        print("Internet connection is working")
        return True
    except requests.RequestException:
        print("No internet connection available")
        return False

if __name__ == "__main__":
    if test_connection():
        while True:
            prompt = input("Enter your Realtime Query: ")
            print(RealtimeSearchEngine(prompt))
    else:
        print("Please check your internet connection and try again.")  