from flask import Flask, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Model ve tokenizer'ı yükle
model_path = "C:/Users/sonay/saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Duygu analizi yapan fonksiyon
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label_map = {0: "Negatif 😞", 1: "Pozitif 😊", 2: "Nötr 😐"}
    return label_map[prediction]

# Ana sayfa ve sonuç gösterimi
@app.route("/", methods=["GET", "POST"])
def index():
    sonuc = ""
    yorum = ""

    if request.method == "POST":
        yorum = request.form["yorum"]
        sonuc = predict_sentiment(yorum)
        
        # Sonuç kutusu rengine karar ver
        renkler = {"Negatif 😞": "#FF6B6B", "Pozitif 😊": "#4CAF50", "Nötr 😐": "#BDBDBD"}
        kutu_rengi = renkler[sonuc]

        return f"""
        <html>
        <head>
            <title>Kadın Ürün Yorum Analiz Platformu</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #fbc2eb, #a6c1ee);
                    text-align: center;
                    color: #333;
                }}
                h1 {{
                    margin-top: 20px;
                    color: #8e44ad;
                }}
                form {{
                    background-color: #ffffff;
                    border-radius: 10px;
                    padding: 20px;
                    width: 50%;
                    margin: auto;
                    box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
                }}
                textarea, button {{
                    width: 90%;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                button {{
                    background-color: #8e44ad;
                    color: white;
                    border: none;
                    font-size: 16px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #732d91;
                }}
                .result {{
                    margin-top: 20px;
                    padding: 10px;
                    background-color: {kutu_rengi};
                    color: white;
                    border-radius: 8px;
                    display: inline-block;
                    font-size: 16px;
                    width: 40%; /* Daha dar hale getirildi */
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }}
                .emoji {{
                    font-size: 40px; /* Emojiyi küçülttük */
                }}
            </style>
        </head>
        <body>
            <h1>👗 Kadın Ürün Yorum Analiz Platformu</h1>
            <form method="post">
                <label><strong>Yorumunuzu Girin:</strong></label><br>
                <textarea name="yorum" rows="4" placeholder="Yorumunuzu buraya yazın..."></textarea><br>
                <button type="submit">Analiz Et</button>
            </form>
            <div class="result">
                <h3>Yorumunuz:</h3>
                <p>{yorum}</p>
                <h3>Duygu Durumu: {sonuc} <span class="emoji">{sonuc[-2]}</span></h3>
            </div>
        </body>
        </html>
        """
    
    # İlk sayfa yüklemesi
    return f"""
    <html>
    <head>
        <title>Kadın Ürün Yorum Analiz Platformu</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(to right, #fbc2eb, #a6c1ee);
                text-align: center;
                color: #333;
            }}
            h1 {{
                margin-top: 20px;
                color: #8e44ad;
            }}
            form {{
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                width: 50%;
                margin: auto;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
            }}
            textarea, button {{
                width: 90%;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }}
            button {{
                background-color: #8e44ad;
                color: white;
                border: none;
                font-size: 16px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #732d91;
            }}
        </style>
    </head>
    <body>
        <h1>👗 Kadın Ürün Yorum Analiz Platformu</h1>
        <form method="post">
            <label><strong>Yorumunuzu Girin:</strong></label><br>
            <textarea name="yorum" rows="4" placeholder="Yorumunuzu buraya yazın..."></textarea><br>
            <button type="submit">Analiz Et</button>
        </form>
    </body>
    </html>
    """

# Flask sunucusunu başlat
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
