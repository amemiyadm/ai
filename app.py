#!/usr/local/bin/python3.7

import pickle
import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import util
from transformers import BertJapaneseTokenizer, BertModel
import google.generativeai as genai

app = Flask(__name__)


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                                             truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


model = SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

genai.configure(api_key='AIzaSyAkzQkK_NZsFXK-jZoF9j51zV7F7KWNdm0')
google_model = genai.GenerativeModel("gemini-2.0-flash")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    with open("embeddings.pkl", "rb") as f:
        loaded_embeddings = pickle.load(f)

    user_input = request.form['user-input']

    embeddings2 = model.encode([user_input], batch_size=8)

    most_similar = -1
    most_idx = -1

    # コサイン類似度の計算
    for i, loaded_embedding in enumerate(loaded_embeddings):
        cosine_scores = util.cos_sim(loaded_embedding['vector'], embeddings2)
        best_idx = torch.argmax(cosine_scores)
        best_score = cosine_scores[best_idx]
        if most_similar < best_score:
            most_similar = best_score
            most_idx = i

    # 関連性の高い文章を取得
    most_relevant = loaded_embeddings[most_idx]

    prompt = f'''
    3つの条件を守り、以下の記事から「{user_input}」について記述してある部分を簡潔に説明してください。
    条件1.丁寧過ぎない適度な敬語で回答する
    条件2.回答に説明以外の前置きなどを含まない
    条件3.「こちらの記事では、」から説明を始める
    
    {most_relevant}
    '''

    response = google_model.generate_content(prompt)

    return jsonify({'output': response.text})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
