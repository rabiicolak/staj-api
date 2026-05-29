import joblib
from final_sistem import load_student_data, encode_student_data, train_model

student_df = load_student_data("test1.xlsx")
_, X, y = encode_student_data(student_df)

gb_model, X_train, X_test, y_train, y_test = train_model(X, y)

joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model ve sütunlar kaydedildi.")