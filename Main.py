from flask import Flask, render_template, request, jsonify
from com import process_single_query
import logging

app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        app.logger.debug(f"Received message: {user_message}")
        response = process_single_query(user_message)
        app.logger.debug(f"Sending response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: ")
        return jsonify({'response': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
