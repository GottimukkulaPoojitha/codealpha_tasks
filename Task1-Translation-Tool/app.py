from flask import Flask, render_template, request, send_file
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ''
    if request.method == 'POST':
        text = request.form['text']
        src_lang = request.form['src_lang']
        dest_lang = request.form['dest_lang']
        result = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        translated_text = result

        # Audio file create cheyyatam
        tts = gTTS(text=translated_text, lang=dest_lang)
        tts.save('static/output.mp3')

    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)