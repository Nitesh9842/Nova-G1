import cohere
from rich import print
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
CohereAPIKey = env_vars.get("COHERE_API_KEY")

# Initialize Cohere API
co = cohere.Client(api_key=CohereAPIKey) # type: ignore
funcs = [
    "exit", "general", "realtime", "open", "close", "play",
     "generate image", "system", "content", "google search", 
     "youtube search", "reminder"
]

messages = []

preamble =  """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write a application and open it in notepad'
*** Do not answer any query, just decide what kind of query is given to you. ***
-> Respond with 'general ( query )' if a query can be answered by a llm model (conversational ai chatbot) and doesn't require any up to date information like if the query is 'who was akbar?' respond with 'general who was akbar?', if the query is 'how can i study more effectively?' respond with 'general how can i study more effectively?', if the query is 'can you help me with this math problem?' respond with 'general can you help me with this math problem?', if the query is 'Thanks, i really liked it.' respond with 'general thanks, i really liked it.' , if the query is 'what is python programming language?' respond with 'general what is python programming language?
-> Respond with 'realtime ( query )' if a query requires up to date information like if the query is 'what's the weather like today?' respond with 'realtime what's the weather like today?', if the query is 'what is the capital of india?' respond with 'realtime what is the capital of india?' , if the query is 'what is python programming?' respond with 'realtime what is python programming?'  
-> Respond with 'open ( query )' if a query is asking to open a website or an application like if the query is 'open youtube' respond with 'open youtube', if the query is 'open facebook' respond with 'open facebook', if the query is 'open instagram' respond with 'open instagram'
-> Respond with 'close ( query )' if a query is asking to close a website or an application like if the query is 'close chrome' respond with 'close chrome', if the query is 'close notepad' respond with 'close notepad', if the query is 'close word' respond with 'close word'
-> Respond with 'play ( query )' if a query is asking to play a song or a video like if the query is 'play despacito' respond with 'play despacito', if the query is 'play shape of you' respond with 'play shape of you', if the query is 'play baby' respond with 'play baby'
-> Respond with 'generate image ( query )' if a query is asking to create an image like if the query is 'create an image of a sunset' respond with 'generate image sunset', if the query is 'create an image of a sunrise' respond with 'generate image sunrise', if the query is 'create an image of a beach' respond with 'generate image beach'
-> Respond with 'system ( query )' if a query is asking to perform a system operation like if the query is 'shutdown the system' respond with 'system shutdown the system', if the query is 'restart the system' respond with 'system restart the system', if the query is 'log off the system' respond with 'system log off the system'
-> Respond with 'content ( query )' if a query is asking to create a content like if the query is 'write a story' respond with 'content write a story', if the query is 'write a poem' respond with 'content write a poem', if the query is 'write a application' respond with 'content write a application'
-> Respond with 'google search ( query )' if a query is asking to search something on google like if the query is 'search for cute cats' respond with 'google search cute cats', if the query is 'search for cute dogs' respond with 'google search cute dogs', if the query is 'search for cute birds' respond with 'google search cute birds'
-> Respond with 'youtube search ( query )' if a query is asking to search something on youtube like if the query is 'search youtube for cooking tutorials' respond with 'youtube search cooking tutorials', if the query is 'search youtube for coding tutorials' respond with 'youtube search coding tutorials', if the query is 'search youtube for dancing tutorials' respond with 'youtube search dancing tutorials'
-> Respond with 'reminder ( query )' if a query is asking to set a reminder like if the query is 'remind me to call mom at 5pm' respond with 'reminder call mom at 5pm', if the query is 'remind me to call dad at 6pm' respond with 'reminder call dad at 6pm', if the query is 'remind me to call sister at 7pm' respond with 'reminder call sister at 7pm'
-> Respond with 'exit' if you want to exit the conversation

"""

system_message = {
    "role": "system",
    "message": preamble
}

ChatHistory = [
    {"role" : "user", "message" : "how are you?"},
    {"role" : "Chatbot", "message" : "general how are you?"},
    {"role" : "user", "message" : "do you like pizza?"},
    {"role" : "Chatbot", "message" : "general do you like pizza?"},
    {"role" : "user", "message" : "what is the capital of india?"},
    {"role" : "Chatbot", "message" : "general what is the capital of india?"},
    {"role" : "user", "message" : "can you help me with this math problem?"},
    {"role" : "Chatbot", "message" : "general can you help me with this math problem?"},
    {"role" : "user", "message" : "Thanks, i really liked it."},
    {"role" : "Chatbot", "message" : "general thanks, i really liked it."},
    {"role" : "user", "message" : "what is python programming language?"},
    {"role" : "Chatbot", "message" : "general what is python programming language?"},
    {"role": "user", "message": "how are you?"},
    {"role": "Chatbot", "message": "general how are you?"},
    {"role": "user", "message": "what's the weather like today?"},
    {"role": "Chatbot", "message": "realtime what's the weather like today?"},
    {"role": "user", "message": "open youtube"},
    {"role": "Chatbot", "message": "open youtube"},
    {"role": "user", "message": "search for cute cats"},
    {"role": "Chatbot", "message": "google search cute cats"},
    {"role": "user", "message": "play despacito"},
    {"role": "Chatbot", "message": "play despacito"},
    {"role": "user", "message": "create an image of a sunset"},
    {"role": "Chatbot", "message": "generate image sunset"},
    {"role": "user", "message": "remind me to call mom at 5pm"},
    {"role": "Chatbot", "message": "reminder call mom at 5pm"},
    {"role": "user", "message": "what is the capital of india?"},
    {"role": "Chatbot", "message": "general what is the capital of india?"},
    {"role": "user", "message": "close chrome"},
    {"role": "Chatbot", "message": "close chrome"},
    {"role": "user", "message": "search youtube for cooking tutorials"},
    {"role": "Chatbot", "message": "youtube search cooking tutorials"},
    {"role": "user", "message": "what is python programming?"},
    {"role": "Chatbot", "message": "general what is python programming?"}
]

def FirstLayerDMM(prompt: str, depth=0):
    try:
        messages.append({"role": "user", "content": prompt})

        response = co.chat(
            model='command-r-plus',
            message=prompt,
            temperature=0.7,
            chat_history=[system_message] + ChatHistory,
            prompt_truncation='OFF',
        )

        response_text = response.text.replace("\n", "")
        response_items = [i.strip() for i in response_text.split(", ")]

        temp = []
        for task in response_items:
            for func in funcs:
                if task.startswith(func):
                    temp.append(task)

        if any("(query)" in task for task in temp):
            return FirstLayerDMM(prompt, depth + 1)
        else:
            return temp

    except Exception as e:
        return [f"Error: {str(e)}"]


if __name__ == "__main__":
    while True:
        print(FirstLayerDMM(input(">>> ")))
