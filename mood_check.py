from textblob import TextBlob

def analyze_text(user_input):
    blob = TextBlob(user_input)
    # Polarity is from -1 (negative) to 1 (positive)
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0:
        return f"😊 Positive ({sentiment:.2f})"
    elif sentiment < 0:
        return f"😡 Negative ({sentiment:.2f})"
    else:
        return f"😐 Neutral ({sentiment:.2f})"

# Let's test it
test_phrases = [
    "I love this new Linux setup, it's incredibly fast!",
    "This error message is so frustrating and I hate it.",
    "The weather today is exactly as expected."
]

for phrase in test_phrases:
    print(f"Phrase: {phrase}")
    print(f"Result: {analyze_text(phrase)}\n")
