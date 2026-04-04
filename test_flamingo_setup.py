import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, BitsAndBytesConfig
import librosa
import numpy as np

def test_flamingo_setup():
    model_id = "nvidia/audio-flamingo-3-hf"
    
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print(f"Loading {model_id} into GPU 0 (RTX 3090)...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Test analysis with dummy audio
        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "Describe the sounds in this audio clip."},
        #             {"type": "audio", "path": "test_audio_16k.wav"},
        #         ],
        #     }
        # ]
        # inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)
        # outputs = model.generate(**inputs, max_new_tokens=200)
        # print(processor.batch_decode(outputs, skip_special_tokens=True))

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    test_flamingo_setup()
