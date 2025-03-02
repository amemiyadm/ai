from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = SentenceTransformer('intfloat/multilingual-e5-large')


@app.route("/process", methods=["POST"])
def hello():
    print(request.form['user-input'])
    user_input = request.form['user-input']
    embeddings = model.encode([user_input])
    return jsonify({'output': embeddings.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
