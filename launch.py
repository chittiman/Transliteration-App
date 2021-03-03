import gradio as gr
from scripts.app_utils import beam_searcher_loader

searcher_dict = {}

languages = ["hindi", "telugu", "tamil","kannada"]

for language in languages:
    searcher_dict[language] = beam_searcher_loader(language)
    
def translit_sentence(sentence,language):
    language = language.lower()
    searcher = searcher_dict[language]
    return searcher.translit_sentence(sentence)

sentence = gr.inputs.Textbox(lines=5, placeholder="Text here..", label = "Sentence")
language = gr.inputs.Radio(["Hindi", "Telugu", "Tamil", "Kannada"], label="Language")
iface = gr.Interface(translit_sentence, [sentence, language],"text")
iface.launch(share = True)
    
