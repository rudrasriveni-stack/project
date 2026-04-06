import joblib

model = joblib.load("sentiment_model.pkl")

text = input("Enter a review or social media post: ")

prediction = model.predict([text])

print("Predicted Sentiment:", prediction[0])