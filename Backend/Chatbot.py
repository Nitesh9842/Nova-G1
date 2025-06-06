from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values
import os
import logging

env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")


client = Groq(api_key=GroqAPIKey)

messages = []

System = f"""Hello, I am {Username},I am student of B.Tech CSE AI&ML in Shri Vishwkarma Skill University ,  You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
 *** If some one ask you 'who is your dovleper' ,'who made you', 'who is your creator' , you should say 'I was created by {Username} and tell about me only.'***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** always give the answer in easy lang. and give youtube links for spacfic topic , never mention your training data. ***
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Always provide answer in formatted way , start heading always next line and add colour ,emoji to give attractive look   ***
"""

systemChatBot = [
    {"role": "system", "content": System}
]


try:
    with open(r"Data/ChatLog.json", "r") as f:
        
        messages = load(f)
except FileNotFoundError:

    with open(r"Data/Chatlog.json", "w" ) as f:
        dump([], f)


def RealtimeInformation():
    current_date_time = datetime.datetime.now()
    day =  current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year =  current_date_time.strftime("%Y") 
    hour =  current_date_time.strftime("%H") 
    minute =  current_date_time.strftime("%M")
    second =  current_date_time.strftime("%S")


    data = f"please use this real-time information if needed, \n"
    data += f"Day: {day}\nDate:{date}\nmonth: {month}\nyear: {year}\n"
    date += f"Time: {hour} hours :{minute} minutes :{second} seconds. \n"
    return data

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [lines for lines in lines if lines.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

def ChatBot (Query):
    """this function send the user's query to the chatbot and return to response to Ai"""
    try:
        with open(r"Data/ChatLog.json", "r") as f:
            messages = load(f)

        messages.append({"role": "user", "content": Query})

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=systemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""

        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        Answer = Answer.replace("<\\s>","")

        messages.append({"role": "assistant", "content": Answer})

        with open(r"Data/ChatLog.json", "w") as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer=Answer)  

    except Exception as e:
        print(f"error:{e}")
        with open(r"Data/ChatLog.json", "w") as f:
            dump(messages, f, indent=4)
            return ChatBot(Query)

if __name__ == "__main__":
    while True:
        user_input = input("Enter your Question :  ")
        print(ChatBot(user_input))                     
