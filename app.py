from flask import Flask, jsonify, request
import torch
from transformers import BertJapaneseTokenizer, BertModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class SentenceBertJapanese:

    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True,
                return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


model = SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')


@app.route("/process", methods=["POST"])
def hello():
    print(request.form['user-input'])
    user_input = request.form['user-input']
    embeddings = model.encode([user_input], batch_size=8)
    return jsonify({'output': embeddings.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
