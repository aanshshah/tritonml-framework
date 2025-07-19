"""Quick start example for TritonML."""

from tritonml import deploy

def main():
    """Simple example of deploying a model with TritonML."""
    
    # Deploy a sentiment analysis model
    print("Deploying sentiment analysis model...")
    client = deploy(
        "distilbert-base-uncased-finetuned-sst-2-english",
        server_url="localhost:8000"
    )
    
    # Test with some examples
    texts = [
        "This movie was absolutely fantastic!",
        "I really didn't enjoy this product.",
        "The service was okay, nothing special.",
        "Best purchase I've ever made!",
        "Terrible experience, would not recommend."
    ]
    
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    
    for text in texts:
        sentiment = client.predict(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()