from flask import Flask, request, jsonify
from flask_cors import CORS

from objects_generation import present

app = Flask(__name__)
CORS(app)


@app.route('/present', methods=['POST'])
def generate_exhibition():
    user_input = request.json.get('input')
    text, info = present(user_input)
    return jsonify({"text": text, "info": info})


if __name__ == '__main__':
    app.run(debug=True)
