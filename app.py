from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained fine-tuned model
model = T5ForConditionalGeneration.from_pretrained('../model')
tokenizer = T5Tokenizer.from_pretrained('../model')

@app.route('/correct', methods=['POST'])
def correct_text():
    # Get input text from the request
    input_data = request.json.get('incorrect_text', '')

    # Prepare the input for the model
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate corrected text using the model
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512)

    # Decode the output to text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the corrected text as a JSON response
    return jsonify({"corrected_text": corrected_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
