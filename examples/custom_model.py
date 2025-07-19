"""Example of creating a custom model with TritonML."""

import numpy as np
from pathlib import Path
from tritonml import TritonModel
from tritonml.core.config import TritonConfig
from tritonml.core.converter import ModelConverter


class CustomSentimentModel(TritonModel):
    """Custom sentiment analysis model with additional features."""
    
    def __init__(self, config: TritonConfig):
        super().__init__(config)
        self.sentiment_map = {
            0: "very negative",
            1: "negative", 
            2: "neutral",
            3: "positive",
            4: "very positive"
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load custom model."""
        config = TritonConfig(
            model_name="custom-sentiment",
            input_shapes={"text": [-1]},
            output_shapes={"sentiment": [5], "confidence": [1]},
            **kwargs
        )
        return cls(config)
    
    def preprocess(self, inputs):
        """Custom preprocessing logic."""
        # In real implementation, this would tokenize text
        # For demo, we'll return dummy data
        if isinstance(inputs, str):
            inputs = [inputs]
        
        return {
            "text": np.array([[hash(text) % 1000] for text in inputs])
        }
    
    def postprocess(self, outputs):
        """Custom postprocessing with confidence scores."""
        sentiment_logits = outputs.get("sentiment", outputs.get("output"))
        
        # Get sentiment predictions
        predictions = np.argmax(sentiment_logits, axis=-1)
        confidences = np.max(np.softmax(sentiment_logits, axis=-1), axis=-1)
        
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append({
                "sentiment": self.sentiment_map[pred],
                "confidence": float(conf),
                "score": int(pred) + 1  # 1-5 star rating
            })
        
        return results if len(results) > 1 else results[0]
    
    def _get_converter(self):
        """Return custom converter."""
        # In real implementation, this would return actual converter
        return ModelConverter(self, {})


def main():
    """Demonstrate custom model usage."""
    
    # Create custom model
    model = CustomSentimentModel.from_pretrained("custom-model")
    
    # In real scenario, you would convert and deploy
    # model.convert()
    # model.deploy()
    
    # For demo, show preprocessing and expected output format
    print("Custom Sentiment Analysis Model")
    print("-" * 50)
    
    test_inputs = [
        "This product exceeded all my expectations!",
        "Not worth the money at all.",
        "It's okay, does what it's supposed to do."
    ]
    
    # Show preprocessing
    preprocessed = model.preprocess(test_inputs)
    print(f"Preprocessed inputs shape: {preprocessed['text'].shape}")
    
    # Simulate model output
    dummy_outputs = {
        "sentiment": np.random.randn(3, 5),
        "confidence": np.random.rand(3, 1)
    }
    
    # Show postprocessing
    results = model.postprocess(dummy_outputs)
    
    print("\nResults:")
    for text, result in zip(test_inputs, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} ({result['score']}/5 stars)")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()