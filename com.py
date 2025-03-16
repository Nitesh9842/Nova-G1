from Backend.Model import FirstLayerDMM
from Backend.Chatbot import ChatBot
from Backend.RealtimeSearchEngine import RealtimeSearchEngine
from rich import print
import time
import threading
import sys

def loading_animation(stop_event, message):
    while not stop_event.is_set():
        sys.stdout.write(f"\r {message}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()

def process_single_query(user_message):
    try:
        # Get query type from Model.py
        query_type = FirstLayerDMM(user_message)
        
        # Process based on query type
        if "realtime" in str(query_type).lower():
            result = RealtimeSearchEngine(user_message)
        else:
            result = ChatBot(user_message)
        
        return result
    except Exception as e:
        return f"Error: {e}"

def process_query():
    while True:
        try:
            user_input = input("\nEnter your query: ")
            
            # Get query type from Model.py
            query_type = FirstLayerDMM(user_input)
            
            # Create stop event for animation
            stop_event = threading.Event()
            
            # Process based on query type
            if "realtime" in str(query_type).lower():
                # Start loading animation in separate thread
                loading_thread = threading.Thread(
                    target=loading_animation, 
                    args=(stop_event, "Searching...")
                )
                loading_thread.start()
                
                result = RealtimeSearchEngine(user_input)
                
                # Stop loading animation
                stop_event.set()
                loading_thread.join()
                
            else:
                # Start loading animation in separate thread
                loading_thread = threading.Thread(
                    target=loading_animation, 
                    args=(stop_event, "Thinking...")
                )
                loading_thread.start()
                
                result = ChatBot(user_input)
                
                # Stop loading animation
                stop_event.set()
                loading_thread.join()
            
            print(f"\n{result}")
            
        except Exception as e:
            print(f"\n[red] Server Error: {e}.[/red]")

if __name__ == "__main__":
    process_query()
