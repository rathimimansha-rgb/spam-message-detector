
import joblib
import string
from pathlib import Path

# Load the saved model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


# Clean the message using the same preprocessing
# used during model training
def clean_message(message):
    message = message.lower()
    message = message.translate(
        str.maketrans("", "", string.punctuation)
    )
    return message


# Predict whether a message is spam or ham
def predict_spam(message):
    cleaned_message = clean_message(message)

    features = vectorizer.transform([cleaned_message])
    prediction = model.predict(features)[0]

    classes = list(model.classes_)
    spam_index = classes.index(1)
    spam_probability = model.predict_proba(features)[0][spam_index]

    label = "SPAM" if prediction == 1 else "HAM"

    return label, spam_probability


# Run the detector
if __name__ == "__main__":
    print("=== SPAM MESSAGE DETECTOR ===")

    while True:
        message = input(
            "\nEnter a message (or type 'quit' to exit): "
        ).strip()

        if message.lower() == "quit":
            print("Goodbye!")
            break

        if not message:
            print("Please enter a message.")
            continue

        label, probability = predict_spam(message)

        print(f"Prediction: {label}")
        print(f"Spam probability: {probability * 100:.2f}%")
