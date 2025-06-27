from transformers import AutoTokenizer, AutoModelForCausalLM
import os

HF_TOKEN = "your_huggingface_token"
model_id = "ibm-granite/granite-3.3-2b-instruct"
model_path = "model/granite-model"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    device_map="auto",
    load_in_8bit=True
)

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
print("✅ Model downloaded and saved to:", model_path)
