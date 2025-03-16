import cohere
from rich import print
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
CohereAPIKey = env_vars.get("COHERE_API_KEY")

# Initialize Cohere API
co = cohere.Client(api_key=CohereAPIKey)
funcs = [
    "exit", "general", "realtime", "open", "close", "play",
     "generate image", "system", "content", "google search", 
     "youtube search", "reminder"
]

messages = []

preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write a application and open it in notepad'
*** Do not answer any query, just decide what kind of query is given to you. ***

-> Respond with 'general ( query )' if a query:
    - Can be answered by a llm model (conversational ai chatbot)
    - Doesn't require real-time information
    - Is about historical facts, concepts, or general knowledge
    - Involves personal advice or opinions
    - Is about programming help or technical concepts
     - Is about a task that can be done by a human
    - Examples: 'who was einstein?', 'how do loops work in python?', 'what's the best way to learn math?'

-> Respond with 'realtime ( query )' if the query:
    - Requires current or real-time information
    - involves link, youtube, website or song
    - Involves news, events, or updates
    - Involves a task that requires real-time information or current data
    - Examples: 'what's the current weather in new york?', 'what's the latest news
    - Needs live data or updates
    - Is about current events, weather, or market conditions
    - Asks about live status or conditions
    - Examples: 'what's the current stock price of apple?', 'what's the current traffic
    - Asks for real-time information or updates
    - Examples: 'what's happening in the world right now?', 'is it raining outside?', 'what's trending on twitter?'

-> Respond with 'open ( application )' if the query:
    - Requests to launch any application, software, or website
    - Uses verbs like 'launch', 'start', 'run', 'open'
    - Examples: 'open spotify', 'launch word processor', 'start calculator'

-> Respond with 'close ( application )' if the query:
    - Requests to terminate any running application
    - Uses verbs like 'close', 'exit', 'quit', 'terminate'
    - Examples: 'close firefox', 'exit spotify', 'terminate word'

-> Respond with 'play ( content )' if the query:
    - Requests to play audio/video content
    - Asks for music, movies, or sound playback
    - Examples: 'play beethoven', 'start some jazz music', 'play meditation sounds'

-> Respond with 'generate image ( description )' if the query:
    - Requests image creation or generation
    - Asks for visual content or artwork
    - Examples: 'create a picture of mountains', 'generate art of space', 'make an image of forest'

-> Respond with 'system ( command )' if the query:
    - Relates to system operations or hardware
    - Requests system information or control
    - Examples: 'check disk space', 'show memory usage', 'system temperature'

-> Respond with 'content ( type )' if the query:
    - Requests creation of text content
    - Asks for writing assistance
    - Examples: 'write an email', 'create a blog post', 'make a resume'

-> Respond with 'google search ( query )' if the query:
    - Explicitly mentions Google search
    - Asks to find information online
    - Examples: 'search google for recipes', 'find hotels on google'

-> Respond with 'youtube search ( query )' if the query:
    - Specifically mentions YouTube search
    - Asks to find videos
    - Examples: 'find tutorials on youtube', 'search youtube for music videos'

-> Respond with 'reminder ( task )' if the query:
    - Requests to set reminders or alarms
    - Involves future tasks or events
    - Examples: 'remind me about meeting', 'set alarm for 7am', 'reminder for doctor appointment'

-> Respond with 'exit' if the query:
    - Indicates ending the conversation
    - Uses words like 'bye', 'quit', 'end', 'stop'
    - Examples: 'goodbye', 'see you later', 'that's all'
"""

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
    {"role" : "Chatbot", "message" : "general what is python programming language?"}
]

def FirstLayerDMM(prompt: str):
    try:
        messages.append({"role": "user", "content": prompt})

        # Using the newer chat method instead of chat_stream
        response = co.chat(
            model='command-r-plus',
            message=prompt,
            temperature=0.7,
            chat_history=ChatHistory,  # type: ignore
            prompt_truncation='OFF',
            preamble=preamble
        )

        # Get the text from the response
        response_text = response.text
        
        # Process response
        response_text = response_text.replace("\n", "")
        response_items = response_text.split(", ")
        response_items = [i.strip() for i in response_items]

        temp = []
        for task in response_items:
            for func in funcs:
                if task.startswith(func):
                    temp.append(task)

        if any("(query)" in task for task in temp):
            return FirstLayerDMM(prompt)
        else:
            return temp

    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == "__main__":
    while True:
        print(FirstLayerDMM(input(">>> ")))
