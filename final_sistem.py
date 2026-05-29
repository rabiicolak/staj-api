#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# AKILLI STAJ VE KARİYER EŞLEŞTİRME SİSTEMİ
# Makine Öğrenmesi + Matching Algoritması
# 1. ÖĞRENCİ VERİSİNİ HAZIRLAMA


def load_student_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # GNO dönüşümü
    df["GNO"] = df["GNO"].replace({
        "Ortalama": 0,
        "İyi": 1,
        "Iyi": 1,
        "Çok iyi": 2,
        "Çok İyi": 2,
        "Cok iyi": 2,
        "Cok İyi": 2,
        "Mükemmel": 3,
        "Mukemmel": 3
    })

    df["GNO"] = pd.to_numeric(df["GNO"])

    # Metin temizliği
    df["Ilgili_Alan"] = df["Ilgili_Alan"].astype(str).str.strip()
    df["Proje"] = df["Proje"].astype(str).str.strip()
    df["Calistigi_Alan"] = df["Calistigi_Alan"].astype(str).str.strip()

    return df


def encode_student_data(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df, columns=["Ilgili_Alan", "Proje"])

    X = df_encoded.drop("Calistigi_Alan", axis=1)
    y = df_encoded["Calistigi_Alan"]

    return df_encoded, X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test


# 2. YENİ ÖĞRENCİ İÇİN TAHMİN


def predict_student_area(model, X_columns, student_info: dict):
    yeni_df = pd.DataFrame([student_info])

    yeni_df["Ilgili_Alan"] = yeni_df["Ilgili_Alan"].astype(str).str.strip()
    yeni_df["Proje"] = yeni_df["Proje"].astype(str).str.strip()

    yeni_encoded = pd.get_dummies(yeni_df, columns=["Ilgili_Alan", "Proje"])
    yeni_encoded = yeni_encoded.reindex(columns=X_columns, fill_value=0)

    tahmin_alan = model.predict(yeni_encoded)[0]
    return tahmin_alan



# 3. FİRMA VERİSİNİ HAZIRLAMA


def load_company_data(file_path: str) -> pd.DataFrame:
    firma_df = pd.read_excel(file_path)

    firma_df["Ilgili_Alan"] = firma_df["Ilgili_Alan"].astype(str).str.strip()
    firma_df["FİRMA"] = firma_df["FİRMA"].astype(str).str.strip()

    return firma_df


# 4. MODEL ÇIKTISINI FİRMA ALANINA ÇEVİRME


def map_predicted_area_to_company_area(predicted_area: str) -> str:
    alan_donusum = {
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

    return alan_donusum.get(predicted_area, predicted_area)


# 5. EŞLEŞTİRME PUANI


def calculate_match_score(student_profile: dict, company_row) -> int:
    score = 0

    # Alan eşleşmesi
    if student_profile["alan"] == company_row["Ilgili_Alan"]:
        score += 40

    # Beceri eşleşmeleri
    if student_profile["Python"] >= company_row["Python"]:
        score += 10

    if student_profile["Java"] >= company_row["Java"]:
        score += 10

    if student_profile["Csharp"] >= company_row["Csharp"]:
        score += 10

    if student_profile["C++"] >= company_row["C++"]:
        score += 10

    if student_profile["Veritabani"] >= company_row["Veritabani"]:
        score += 10

    # GNO eşleşmesi
    if student_profile["GNO"] >= company_row["GNO"]:
        score += 10

    return score


def score_to_label(score: int) -> str:
    if score >= 90:
        return "Çok Uygun"
    elif score >= 70:
        return "Uygun"
    elif score >= 50:
        return "Kısmen Uygun"
    else:
        return "Uygun Değil"


def match_student_to_companies(student_profile: dict, company_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, firma in company_df.iterrows():
        score = calculate_match_score(student_profile, firma)

        results.append({
            "Firma": firma["FİRMA"],
            "Alan": firma["Ilgili_Alan"],
            "Score": score,
            "Durum": score_to_label(score)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    return result_df


# 6. ANA AKIŞ


# Öğrenci verisini yükle
student_df = load_student_data("test1.xlsx")

# Encode et
student_encoded_df, X, y = encode_student_data(student_df)

# Modeli eğit
gb_model, X_train, X_test, y_train, y_test = train_model(X, y)

# Yeni öğrenci bilgisi
new_student = {
    "GNO": 2,
    "Ilgili_Alan": "Yapay Zeka",
    "Proje": "Chatbot Geliştirme",
    "Veritabani": 2,
    "Python": 2,
    "Java": 0,
    "Csharp": 0,
    "C++": 1
}

# Alan tahmini
predicted_area = predict_student_area(gb_model, X.columns, new_student)
print("Tahmin edilen alan:", predicted_area)

# Firma alanına dönüştür
company_area = map_predicted_area_to_company_area(predicted_area)
print("Firma eşleştirmede kullanılacak alan:", company_area)

# Matching için öğrenci profili oluştur
student_profile = {
    "GNO": new_student["GNO"],
    "Python": new_student["Python"],
    "Java": new_student["Java"],
    "Csharp": new_student["Csharp"],
    "C++": new_student["C++"],
    "Veritabani": new_student["Veritabani"],
    "alan": company_area
}

# Firma verisini yükle
company_df = load_company_data("firmaVeri.xlsx")

# Eşleştirme yap
final_matches = match_student_to_companies(student_profile, company_df)

# İlk 10 firmayı göster
print("\nEn uygun firmalar:")
print(final_matches.head(10))


# In[ ]:




