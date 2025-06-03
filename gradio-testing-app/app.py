import subprocess
import gradio as gr
import sentencepiece as spm

from huggingface_hub import hf_hub_download

model_path = hf_hub_download("AnandTharad/conformer-transformer", "conformer-ks7.pt")
src_model = hf_hub_download("AnandTharad/conformer-transformer", "source.model")
tgt_model = hf_hub_download("AnandTharad/conformer-transformer", "target.model")

# Load SentencePiece models
sp = spm.SentencePieceProcessor(model_file=src_model)
tp = spm.SentencePieceProcessor(model_file=tgt_model)

# Sample sentences
sample_sentences = {
    "How are you?": "How are you?",
    "Where are you going?": "Where are you going?",
    "I love neural machine translation.": "I love neural machine translation.",
    "This is a test sentence.": "This is a test sentence."
}

# Translation function
def translate_text(input_text):
    if not input_text.strip():
        return "Please enter or select a sentence."
    
    tokenized_input = sp.encode(input_text.strip(), out_type=str)
    with open("src.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(tokenized_input) + "\n")

    command = [
        "onmt_translate",
        "-model", model_path,
        "-src", "src.txt",
        "-output", "pred.txt",
        "-gpu", "0",
        "-beam_size", "5",
        "-min_length", "1"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return f"Translation failed: {e}"

    with open("pred.txt", "r", encoding="utf-8") as f:
        raw_output = f.read().strip()
    
    return tp.decode(raw_output.split())

# Merge dropdown and textbox input
def resolve_input(dropdown_val, custom_input):
    return custom_input.strip() if custom_input.strip() else dropdown_val

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 Conformer-Transformer NMT Demo")
    gr.Markdown("Choose a sentence or type your own. Custom input takes priority.")

    dropdown = gr.Dropdown(
        choices=list(sample_sentences.values()),
        value=list(sample_sentences.values())[0],
        label="Sample Sentences"
    )
    textbox = gr.Textbox(
        lines=1,
        label="Or Enter Custom Sentence",
        placeholder="Leave blank to use dropdown selection"
    )
    
    final_input = gr.Textbox(visible=False)  # Hidden box for resolved input
    output_box = gr.Textbox(lines=1, label="Translation Output")
    translate_button = gr.Button("Translate")

    # Resolve input before translating
    dropdown.change(fn=resolve_input, inputs=[dropdown, textbox], outputs=final_input)
    textbox.change(fn=resolve_input, inputs=[dropdown, textbox], outputs=final_input)

    translate_button.click(fn=translate_text, inputs=final_input, outputs=output_box)

# Launch
demo.launch(share=True)
