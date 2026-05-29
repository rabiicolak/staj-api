from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Model ve sütunlar
model = joblib.load("model.pkl")
columns = joblib.load("model_columns.pkl")


@app.get("/")
def home():
    return {"message": "API çalışıyor"}


def alan_donustur(tahmin_alan):
    alan_map = {
        "Yapay Zeka Araştırmacısı": "Yapay Zeka",
        "Yapay Zeka Mühendisi": "Yapay Zeka",
        "Veri Bilimci": "Veri Bilimi",
        "Web Geliştirici": "Web",
        "Mobil Uygulama Geliştirici": "Mobil",
        "Bilgi Güvenliği Analisti": "Siber Güvenlik",
        "Bulut Çözümleri Mimarı": "DevOps",
        "Yazılım Mühendisi": "Backend",
        "Veritabanı Yöneticisi": "Backend",
        "Grafik Programcısı": "Oyun Geliştirme",
        "Oyun Geliştirici": "Oyun Geliştirme",
        "Sağlık Bilişimi Uzmanı": "Veri Bilimi"
    }
    return alan_map.get(tahmin_alan, tahmin_alan)


def uygunluk_etiketi(score):
    if score >= 90:
        return "Çok Uygun"
    elif score >= 70:
        return "Uygun"
    elif score >= 50:
        return "Kısmen Uygun"
    else:
        return "Uygun Değil"


def match_score(ogrenci, firma):
    score = 0

    if ogrenci["alan"] == firma["Ilgili_Alan"]:
        score += 40

    if ogrenci["Python"] >= firma["Python"]:
        score += 10

    if ogrenci["Java"] >= firma["Java"]:
        score += 10

    if ogrenci["Csharp"] >= firma["Csharp"]:
        score += 10

    if ogrenci["C++"] >= firma["C++"]:
        score += 10

    if ogrenci["Veritabani"] >= firma["Veritabani"]:
        score += 10

    if ogrenci["GNO"] >= firma["GNO"]:
        score += 10

    return score


@app.post("/predict")
def predict(data: dict):
    # 1) Gelen veriyi modele uygun hale getir
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # 2) Tahmin yap
    tahmin = model.predict(df)[0]

    # 3) Firma alanına dönüştür
    firma_alani = alan_donustur(tahmin)

    # 4) Firma verisini oku
    firma_df = pd.read_excel("firmaVeri.xlsx")
    firma_df["Ilgili_Alan"] = firma_df["Ilgili_Alan"].astype(str).str.strip()
    firma_df["FİRMA"] = firma_df["FİRMA"].astype(str).str.strip()

    # 5) Öğrenci profilini oluştur
    ogrenci = {
        "GNO": data["GNO"],
        "Python": data["Python"],
        "Java": data["Java"],
        "Csharp": data["Csharp"],
        "C++": data["C++"],
        "Veritabani": data["Veritabani"],
        "alan": firma_alani
    }

    # 6) Firmaları puanla
    sonuclar = []
    for _, firma in firma_df.iterrows():
        score = match_score(ogrenci, firma)
        sonuclar.append({
            "firma": firma["FİRMA"],
            "alan": firma["Ilgili_Alan"],
            "score": score,
            "durum": uygunluk_etiketi(score)
        })

    sonuc_df = pd.DataFrame(sonuclar)
    sonuc_df = sonuc_df.sort_values(by="score", ascending=False)

    # 7) JSON dön
    return {
        "tahmin": tahmin,
        "firma_eslesme_alani": firma_alani,
        "onerilen_firmalar": sonuc_df.head(10).to_dict(orient="records")
    }