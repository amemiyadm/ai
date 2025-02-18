import pickle

import numpy as np
import requests
from bs4 import BeautifulSoup
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertJapaneseTokenizer, BertModel


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


# Hatena Blog のテキストを取得する関数
def get_hatenablog_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article = soup.find('div', class_='entry-content')
    if not article:
        return
    print(article.get_text())


# 文章の取得
urls = ['https://paypay16.hatenablog.com/entry/2025/02/01/102044', 'https://note.com/fond_fairy68/n/nf0b513ca1bdb',
        'https://marcadorjellal.hatenablog.com/entry/2025/02/01/111306',
        'https://tomopi.hatenadiary.com/entry/2025/02/01/123825', 'https://note.com/3onto/n/n30e4519c3326',
        'https://note.com/ngthrk27/n/n4533bfcaeb4b', 'https://nagoon.hatenablog.com/entry/2025/02/01/124406',
        'https://kanigame.hatenablog.com/entry/2025/02/01/125356',
        'https://asumi10313.hatenablog.com/entry/2025/02/01/105133',
        'https://tnmakinosuke.hatenablog.jp/entry/2025/02/01/114606',
        'https://pillosuke.hatenablog.com/entry/2025/02/01/144404',
        'https://ukeruichigo.hatenablog.com/entry/2025/02/01/134559',
        'https://tenku64.hatenablog.com/entry/2025/02/01/130940', 'https://note.com/odayakadeareyo2/n/n2c0c6e50ce13',
        'https://m14zz.hatenablog.com/entry/2025/02/01/130545', 'https://note.com/l0nely_rollingc/n/nfb5859a870d6',
        'https://note.com/wachi_yukkuri/n/n2fcba4a1ac97', 'https://ponio.hateblo.jp/entry/S26',
        'https://note.com/https_jyararanga/n/n741c7efb66c2', 'https://yakkun.com/bbs/party/n6325',
        'https://note.com/brisk_ixia3695/n/n9d85e986a751', 'https://maaach-un.hatenablog.com/entry/2025/02/01/160000',
        'https://chiho-poke.hatenablog.com/entry/2025/02/01/160656',
        'https://miwa52476.hateblo.jp/entry/2025/02/01/142223', 'https://note.com/ketuban_kakutei/n/n14268e6b3f67',
        'https://salamence-erist.hatenablog.com/entry/2025/02/01/151735',
        'https://amemiyapoke.hatenablog.com/entry/2025/02/01/150439',
        'https://ycky-poke.hatenablog.com/entry/2025/02/01/171842',
        'https://ataka-dog.hatenablog.com/entry/2025/02/01/162636', 'https://note.com/aimaru4218/n/n80f9db2e141c',
        'https://note.com/firm_cedum8413/n/n603812f82a22', 'https://pokexcel.hatenablog.com/entry/2025/02/01/180000',
        'https://phantasm-poke.hatenablog.com/entry/2025/02/01/183340',
        'https://pokemon-nicoru.hatenablog.jp/entry/2025/02/01/195125',
        'https://siopokemon.hatenablog.com/entry/2025/02/01/173036',
        'https://note.com/ymik0820/n/nfd51aa7b1229?sub_rt=share_b',
        'https://minrai525.hatenablog.com/entry/2025/02/01/180631',
        'https://kapibara0203.hatenablog.com/entry/2025/02/01/200557',
        'https://missdd.hatenablog.com/entry/2025/02/01/193719',
        'https://sso-061.hatenablog.com/entry/2025/02/01/200520',
        'https://booyan-kachiku.hatenadiary.jp/entry/2025/02/01/203913',
        'https://norunpoke.hatenablog.com/entry/2025/02/01/043657',
        'https://kuzuncha.hatenablog.com/entry/2025/02/01/191011',
        'https://kyoto-kyonkyon.hatenablog.com/entry/2025/02/01/205611',
        'https://gangar-poke.hatenablog.com/entry/2025/02/01/192549',
        'https://note.com/jolly_drake1660/n/n23818472d0b3', 'https://hunk42ga3.hatenadiary.jp/entry/2025/02/01/185022',
        'https://note.com/leal_eel7218/n/nd5457265cb2e', 'https://note.com/mari_nyan/n/n6ead10841d9c',
        'https://penpenpendlar.hatenablog.com/entry/2025/02/01/201714',
        'https://dedekasupoly2.hatenablog.com/entry/2025/02/01/130807',
        'https://plusodawara.hatenablog.com/entry/2025/02/01/205418',
        'https://esura0060.hatenablog.com/entry/2025/02/01/183844',
        'https://selfishtv.hatenablog.com/entry/2025/02/01/220434', 'https://note.com/pal640/n/ndc3d24537ba3',
        'https://yakkun.com/bbs/party/n6326', 'https://note.com/hip_auk2706/n/n082aeaf3a765',
        'http://blog.livedoor.jp/sibakentomodachi/archives/1083121789.html',
        'https://imaroot.hatenablog.com/entry/2025/02/02/002800', 'https://note.com/berupiyo1234/n/nae1f9a2e44fb',
        'https://ohmatsu-poke.hatenablog.com/entry/2025/02/02/001335',
        'https://note.com/gulasonchiemi9/n/nbe5ef3d5e74f', 'https://yakkun.com/bbs/party/n6327',
        'https://hutuunohenjin.hatenablog.com/entry/2025/02/01/222728', 'https://yakkun.com/bbs/party/n6328',
        'https://gall.dcinside.com/mgallery/board/view/?id=pokerankbattle&no=20697',
        'https://note.com/rikuyama_jan/n/n1ac9aee95210', 'https://paru-mpg.hatenablog.com/entry/2025/02/02/100237',
        'https://note.com/kawattamonnnsuta/n/n8fd56c517f21', 'https://soyuta369.hatenadiary.jp/entry/2025/02/02/100125',
        'https://kinnikumanpoke.hatenablog.com/entry/2025/02/01/203540', 'https://yakkun.com/bbs/party/n6331',
        'https://yakkun.com/bbs/party/n6329',
        'https://hz7xlcarm9fyuw5.hatenablog.com/entry/2025/02/02/【シーズン26_最終90位_R2037_臥竜ミライロンゲワダチ】',
        'https://note.com/range_poke/n/nd2f1f815388c', 'https://note.com/ponponty/n/n363f260bb10a',
        'https://marunpoke.hatenablog.com/entry/2025/02/01/205817', 'https://note.com/goodbig89/n/n3f1e7ca5fce4',
        'https://note.com/zumashiii/n/n809b95b8fc6d',
        'https://kurifuto-mpk.hatenablog.com/entry/2025/02/02/【ポケモンSV】ランクマッチシングルs26_最終35位_～',
        'https://wkaji.hatenablog.com/entry/2025/02/02/002207',
        'https://yuruchi0412.hatenablog.com/entry/2025/02/01/160814',
        'https://fooooooo628.hateblo.jp/entry/2025/02/02/170706',
        'https://claris-bradbury.hatenablog.com/entry/2025/02/02/【SV_S26_最終10位】特異解のイブ',
        'https://rosupoke.hatenablog.com/entry/2025/02/02/155000', 'https://wld.hatenablog.com/entry/2025/02/02/194432',
        'https://note.com/light_garlic2193/n/n3850a3bd59ca', 'https://ika0000.hatenablog.jp/entry/2025/02/02/153725',
        'https://winday77.hatenablog.com/entry/2025/02/02/174503',
        'https://matorie-poke.hatenablog.jp/entry/2025/02/02/200727',
        'https://kemuri893.hatenablog.com/entry/2025/02/02/182623',
        'https://kttsukihityannm.hatenablog.com/entry/2025/02/02/183755',
        'https://fuyu-pk.hatenablog.com/entry/2025/02/02/163808',
        'https://yon-toro0615.hatenablog.com/entry/2025/02/02/183104',
        'https://yuki4848poke.hatenadiary.jp/entry/2025/02/02/190448',
        'https://uchiko0929.hatenadiary.jp/entry/2025/02/02/183333',
        'https://moryointhebox.hatenablog.com/entry/2025/02/02/180504', 'https://note.com/kamixmas/n/neb734d3d91a9',
        'https://note.com/genos_ggs/n/n5226e4fe079b', 'https://hp0325.blogspot.com/2025/02/sv-s26-430-1912.html',
        'https://aibo-rsn.hatenablog.com/entry/2025/02/02/221344', 'https://note.com/lofty_knot436/n/n056390ff5080',
        'https://azurlog.hatenablog.com/entry/2025/02/01/211534',
        'https://pokemossa.hatenablog.com/entry/2025/02/03/013410',
        'https://tapufini-poke.hatenadiary.com/entry/2025/02/02/090923', 'https://yakkun.com/bbs/party/n6333',
        'https://maromaku0136.hatenablog.com/entry/2025/02/03/【ポケモンSV】未来への飛翔【最終239位】',
        'https://shigure3333.hatenablog.com/entry/2025/02/02/223019',
        'https://re-42uku.hatenablog.com/entry/2025/02/01/192642',
        'https://waimblog.hatenablog.com/entry/2025/02/02/225122',
        'https://enldnfxfdlfl8oh.hatenablog.com/entry/2025/02/03/121809',
        'https://mabo-nebo.hatenablog.com/entry/2025/02/02/220340',
        'https://ta-pokeeeee.hatenablog.com/entry/2025/02/02/232134',
        'https://26.hatenadiary.com/entry/2025/02/02/141331', 'https://madskull.hatenablog.com/entry/2025/02/03/001519',
        'https://maki-poke7.hatenablog.com/entry/2025/02/03/030255', 'https://note.com/bavelpoke/n/n2ce227f8a19e',
        'https://duxia4.hatenablog.com/entry/2025/02/03/184731',
        'https://mikage-poke.hatenablog.com/entry/2025/02/03/202834',
        'https://obsqhgpvnk98139.hatenablog.com/entry/2025/02/03/171226',
        'https://note.com/kon_pokepoke/n/n2c39a4ddef97', 'https://yatorukun.hatenablog.com/entry/2025/01/30/165710',
        'https://saijakupoke.hatenablog.com/entry/2025/01/22/011938',
        'https://kaioo-poke.hatenablog.com/entry/2025/02/03/225510',
        'https://dr-paburofu-325.hatenablog.com/entry/2025/02/03/224909',
        'https://messipoke.hatenablog.com/entry/2025/02/04/122644',
        'https://naoppi24.hatenablog.com/entry/2025/02/04/170252',
        'https://poccyama0393.hatenablog.com/entry/2025/02/03/231041',
        'https://paov8i5qwu0dwvn.hatenablog.com/entry/2025/02/04/151614',
        'https://rkr-syon.hatenablog.com/entry/2025/02/04/144609',
        'https://yukiyukipokemon.hatenablog.com/entry/2025/02/04/164212',
        'https://tutinoko0523.hatenablog.com/entry/monta', 'https://note.com/haru_shaddoll/n/n3cb76430aaa6',
        'https://ribrashiii.hatenablog.com/entry/2025/02/01/225700', 'https://note.com/cute_carp5373/n/n760ac5538d6f',
        'https://mizumo-yadon079.hatenablog.com/entry/2025/02/04/225150',
        'https://shimoken0128.hatenablog.com/entry/2025/02/05/225240',
        'https://note.com/lively_serval576/n/nd3d1530a4bb9', 'https://xx0405xx.hatenablog.com/entry/2025/02/05/220830',
        'https://schwarz5555.hatenablog.jp/entry/2025/02/06/122504',
        'https://hashige.hatenadiary.jp/entry/2025/02/05/200855',
        'https://konchi-poke.hatenablog.com/entry/2025/02/06/224713',
        'https://ma-sa1004.hatenablog.com/entry/2025/02/06/211055',
        'https://piglet445.hatenablog.com/entry/2025/02/06/191234',
        'https://rapid-clover.hatenablog.com/entry/2025/02/08/215307']

hoge = [get_hatenablog_text(url) for url in urls]


embeddings_list = []
model = SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

# for url in urls:
#     text = get_hatenablog_text(url)
#
#     # spacyを用いて文章をセンテンスごとに分割
#     nlp = spacy.load('ja_ginza')
#     sentences = [sent.text for sent in nlp(text).sents]
#
#     # 文章をベクトルに変換
#     embeddings = model.encode(sentences, batch_size=8).detach().numpy()
#
#     embeddings_list.append(
#         {'url': url, 'text': text, 'vector': embeddings},
#     )


with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_list, f)
    print('完了')

# 読み込み
with open("embeddings.pkl", "rb") as f:
    loaded_embeddings = pickle.load(f)

user_input = '運に頼った戦い方'
embeddings2 = model.encode([user_input], batch_size=8)

most_similar = -1
most_idx = -1

# コサイン類似度の計算
for i, loaded_embedding in enumerate(loaded_embeddings):
    cosine_scores = util.cos_sim(loaded_embeddings[i]['vector'], embeddings2)
    best_idx = torch.argmax(cosine_scores)
    best_score = cosine_scores[best_idx]
    if most_similar < best_score:
        most_similar = best_score
        most_idx = i

# 関連性の高い文章を取得
most_relevant = loaded_embeddings[most_idx]

print("最も関連するurl:", most_relevant['url'])
print("最も関連する文章:", most_relevant['text'])
