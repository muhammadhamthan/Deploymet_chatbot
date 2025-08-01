from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_helper import get_qa_chain, convert_links

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

chain, memory = get_qa_chain()  # memory uses return_messages=True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    try:
        print("User message received:", user_message)

        # Build input with user message only — memory is auto-used by chain
        input_data = {
            "question": user_message
        }

        response = chain.invoke(input_data)
        print("Raw response from chain:", response)

        # Format answer with link conversion
        result = convert_links(response['answer'])

        return jsonify({'response': result})

    except Exception as e:
        print("Error during model invocation:", str(e))
        return jsonify({'response': "Sorry, I couldn't process your request."})

