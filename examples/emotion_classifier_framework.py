"""Example: Using TritonML framework to deploy the emotion classifier."""

from tritonml import deploy
from tritonml.tasks import TextClassificationModel


def main():
    """Deploy emotion classifier using the TritonML framework."""

    # Method 1: Simple deployment with auto-detection
    print("=== Method 1: Quick deployment ===")
    client = deploy(
        "cardiffnlp/twitter-roberta-base-emotion",
        server_url="localhost:8000",
        quantize=True,
        optimize=True
    )

    # Test inference
    result = client.infer({
        "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]]
    })
    print(f"Inference result: {result}")

    # Method 2: Using specific model class
    print("\n=== Method 2: Using TextClassificationModel ===")
    model = TextClassificationModel.from_pretrained(
        "cardiffnlp/twitter-roberta-base-emotion",
        model_name="emotion-classifier",
        labels=["anger", "joy", "optimism", "sadness"]
    )

    # Convert and optimize
    model.convert()
    model.quantize(method="dynamic")

    # Deploy
    client = model.deploy()

    # Use high-level predict API
    emotion = model.predict("I love this amazing framework!")
    print(f"Predicted emotion: {emotion}")

    # Get probabilities
    probs = model.predict_proba("This makes me so happy!")
    print(f"Emotion probabilities: {probs}")

    # Method 3: Using specialized EmotionClassifier
    print("\n=== Method 3: Using EmotionClassifier ===")
    from tritonml.tasks.text_classification import EmotionClassifier

    emotion_model = EmotionClassifier.from_pretrained()
    emotion_model.deploy()

    # Batch prediction
    texts = [
        "I'm so angry about this!",
        "This is the best day ever!",
        "Things will get better soon.",
        "I feel so sad and lonely."
    ]

    predictions = emotion_model.predict(texts)
    for text, emotion in zip(texts, predictions):
        print(f"'{text}' -> {emotion}")

    # Benchmark the model
    print("\n=== Benchmarking ===")
    benchmark_results = emotion_model.benchmark(
        test_inputs=["Test input"] * 10,
        batch_sizes=[1, 8, 16, 32]
    )

    for batch_size, metrics in benchmark_results.items():
        print(f"{batch_size}: {metrics}")


if __name__ == "__main__":
    main()
