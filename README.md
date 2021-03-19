# Transliteration-App
App for transliterating text from roman script to native script for Hindi,Telugu,Tamil,Kannada languages

<b> What is Transliteration? </b>

Transliteration is <b>conversion of a text from one script to another</b>. 

Using this app, words in roman script can be transliterated to native script for these languages. 

Check the demo below

## Working

The transliteration model is basically a  <b> Encoder - Decoder network based on Luong style Attention </b>. 

### Steps

1) Sentence is broken down into continous sequence of characters

2) Each sequence of characters is further broken down into sequence of alphabets and non-alphabets

3) While the alphabet sequences are transliterated, the non alphabets sequences remains unchanged.

4) When the romanized words (i.e alphabet sequences) are fed to the model, it performs decoding step by step

5) <b>Beam search (with a beam size of 5)</b> is performed over outputs at each time step and finally the word with highest log probability score is selected.

6) Once the results from beam search are obtained, the transliterated words are rejoined in a way to <b>preserve the punctuation</b> and final sentence is returned.

Dataset URL:
[https://github.com/google-research-datasets/dakshina](Dakshina dataset)

For queries contact - <a href="https://twitter.com/chittiman">Chitreddy Sairam</a>

## Demo

<img src="https://github.com/chittiman/Transliteration-App/blob/main/demo.gif?raw=true" width="100%">
