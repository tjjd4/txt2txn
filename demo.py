import gradio as gr
import torch

import sys
print(sys.path)

from src.handler import classify_transaction
from src.swap import convert_transaction
from src.transfer import convert_transfer_intent

# Enable Apple Silicon Metal API
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def generate_response(input_text):
    action_type = classify_transaction(input_text)
    if action_type == 1:
        return convert_transfer_intent(input_text)
    elif action_type == 2:
        return convert_transaction(input_text)
    else:
        return "無法識別交易類型"


with gr.Blocks() as demo:
    gr.Markdown("# Gradio Prompt Generator")
    input_text = gr.Textbox(label="輸入文本", placeholder="在這裡輸入文本...", lines=2)

    response_output = gr.Textbox(label="生成回應")
    generate_response_btn = gr.Button(value="生成回應")

    generate_response_btn.click(fn=generate_response,
                                inputs=input_text,
                                outputs=response_output)

if __name__ == "__main__":
    demo.launch()
