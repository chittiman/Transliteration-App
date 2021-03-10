import gradio as gr
from scripts.app_utils import beam_searcher_loader

searcher_dict = {}

languages = ["hindi", "telugu", "tamil","kannada"]

#Loading beam searcher's for all the languages
for language in languages:
    searcher_dict[language] = beam_searcher_loader(language)

#Selecting the beam searcher for appropriate language and converting the words    
def translit_sentence(sentence,language):
    language = language.lower()
    searcher = searcher_dict[language]
    return searcher.translit_sentence(sentence)

sentence = gr.inputs.Textbox(lines=5, placeholder="Text here..", label = "Sentence")
language = gr.inputs.Radio(["Hindi", "Telugu", "Tamil", "Kannada"], label="Language")
iface = gr.Interface(translit_sentence, [sentence, language],"text")
iface.launch(share = True)
    
