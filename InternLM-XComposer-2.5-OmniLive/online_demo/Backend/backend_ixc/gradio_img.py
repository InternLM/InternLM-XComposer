import os
import torch
import gradio as gr


def load():
    root = '/app/RealTime/backend_ixc/tmp/mem'
    img_pth = torch.load(os.path.join(root, 'temp_lol_img.pth'))
    with open(os.path.join(root, 'prompt.txt')) as fd:
        text = fd.read()
    return img_pth, text


with gr.Blocks() as demo:
    imgs = gr.Gallery(columns=6)
    prompt = gr.Textbox()
    btn = gr.Button()

    btn.click(load, [], [imgs, prompt])

demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7861)

