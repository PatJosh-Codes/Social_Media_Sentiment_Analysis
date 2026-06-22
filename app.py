"""
app.py
------
Flask entry-point for the Social Media Sentiment Analysis web application.

Usage
-----
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from prediction import predict

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Accept a JSON or form payload with key 'text', run prediction,
    and return a JSON response.

    Request body (JSON)
    -------------------
    { "text": "I love this product!" }

    Response (JSON)
    ---------------
    {
        "sentiment"  : "Positive",
        "confidence" : 94.3,
        "all_probs"  : { "Positive": 0.943, "Negative": 0.032, "Neutral": 0.025 },
        "error"      : null
    }
    """
    try:
        # Accept both JSON and form-encoded requests
        if request.is_json:
            data = request.get_json(force=True)
            text = data.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()

        if not text:
            return jsonify({
                'sentiment': None,
                'confidence': None,
                'all_probs': {},
                'error': 'Please enter some text to analyse.'
            }), 400

        result = predict(text)

        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'all_probs': result['all_probs'],
            'error': None,
        })

    except FileNotFoundError as exc:
        return jsonify({
            'sentiment': None,
            'confidence': None,
            'all_probs': {},
            'error': str(exc),
        }), 503

    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Prediction error")
        return jsonify({
            'sentiment': None,
            'confidence': None,
            'all_probs': {},
            'error': f'An unexpected error occurred: {exc}',
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
