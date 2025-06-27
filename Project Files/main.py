import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Load model from local path
model_path = "model/granite-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    output = generator(prompt, max_new_tokens=512, do_sample=False, temperature=0.5)
    return output[0]["generated_text"][len(prompt):].strip()

def handle_feedback(prompt, response, rating, comments):
    print("Prompt:", prompt)
    print("Response:", response)
    print("Rating:", rating)
    print("Comments:", comments)
    return "✅ Thank you for your feedback!"

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 CitizenAI - Ask Public Concerns")
    gr.Markdown("Ask any public safety, legal, or community question.")

    with gr.Row():
        prompt = gr.Textbox(label="Your Question", lines=3)
        response = gr.Textbox(label="CitizenAI Response", lines=5, interactive=True)

    submit_btn = gr.Button("Get Answer")
    submit_btn.click(fn=generate_response, inputs=prompt, outputs=response)

    gr.Markdown("### 📝 Feedback")
    rating = gr.Radio(["👍 Yes", "👎 No"], label="Was this response helpful?")
    comments = gr.Textbox(label="Comments", placeholder="Suggestions or comments?", lines=2)
    feedback_output = gr.Textbox(visible=True, label="Feedback Result", interactive=False)

    submit_feedback = gr.Button("Submit Feedback")
    submit_feedback.click(fn=handle_feedback, inputs=[prompt, response, rating, comments], outputs=feedback_output)

demo.launch(share=True)
