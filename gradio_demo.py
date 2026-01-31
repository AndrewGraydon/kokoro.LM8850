import glob
import os
import re
import subprocess
import gradio as gr
import requests
import time

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]

EXAMPLE_SENTENCES = {
    "en-us": "The future is built by those who believe in their dreams.",
    "en": "Life is a journey meant to be discovered with every step.",
    "es": "La vida es un viaje lleno de momentos que recordar.",
    "fr": "Le bonheur se trouve dans les petites choses de la vie.",
    "hi": "ख़ुशी वहीं मिलती है जहाँ दिल मुस्कुराता है।",
    "it": "La bellezza vive negli attimi que ci sorprendono.",
    "pt-br": "A verdadeira magia está nos detalhes da vida.",
    "ja": "未来は今日の選択で変わる。",
    "zh": "每一个声音，都值得被世界听见。",
}

LANG_CODES = {
    'en': 'American English', 'en': 'British English', 'es': 'Spanish es',
    'fr': 'French fr-fr', 'hi': 'Hindi hi', 'it': 'Italian it', 'pt-br': 'Brazilian Portuguese pt-br',
    'ja': 'Japanese', 'zh': 'Mandarin Chinese',
}
LANG_CODES_REV = {v: k for k, v in LANG_CODES.items()}

def change_voice_and_text(language):
    lang_key = LANG_CODES_REV[language]  # 例：Mandarin Chinese -> zh
    filtered_voices = [v for v in voice_list.keys() if v.startswith(lang_key)]
    
    # 例句填充
    example_text = EXAMPLE_SENTENCES.get(lang_key, "")
    
    if not filtered_voices:
        return gr.update(value=example_text), gr.update(value=None, choices=[])

    return (
        gr.update(value=example_text),               # 更新输入文本
        gr.update(value=filtered_voices[0], choices=filtered_voices) # 更新音色
    )

voice_list = {}

# 加载checkpoints/voices下的所有npy文件的文件名作为key
voice_list = glob.glob("checkpoints/voices_npy/*.npy")
voice_list = {os.path.basename(v).replace(".npy", ""): v for v in voice_list}

def tts(sentence, language, speed, voice):
    resp = requests.post(
        "http://127.0.0.1:28000/tts",
        data={
            "sentence": sentence,
            "language": LANG_CODES_REV[language],
            "speed": str(speed),
            "voice": voice,
        }
    )
    if resp.status_code == 200:
        # 确保 history 目录存在
        os.makedirs("history", exist_ok=True)
        
        save_path = f"history/tts_output_{LANG_CODES_REV[language]}_{voice}_{time.time()}.wav"
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 KOKORO Demo")
    
    with gr.Row():
        with gr.Column():
            sentence = gr.Textbox(label="输入文本",value="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片.")
            language = gr.Dropdown(label="选择语言", choices=list(LANG_CODES_REV.keys()), value="Mandarin Chinese")
            speed = gr.Slider(label="速度", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            voice = gr.Dropdown(
                        label="选择音色",
                        choices=list(v for v in voice_list.keys() if v.startswith(LANG_CODES_REV["Mandarin Chinese"][0])),
                        value=list(v for v in voice_list.keys() if v.startswith(LANG_CODES_REV["Mandarin Chinese"][0]))[0] if voice_list else None,
                        allow_custom_value=True
                    )
            generate = gr.Button("生成音频")
        with gr.Column():
            audio = gr.Audio(label="输出音频")
            
        # 点击生成按钮时，调用服务器端的TTS API
        generate.click(
            fn=tts,
            inputs=[sentence, language, speed, voice],
            outputs=audio,
        )
        language.change(
            fn=change_voice_and_text,
            inputs=language,
            outputs=[sentence, voice],
        )

# 启动
ips = get_all_local_ips()
port = 7861
for ip in ips:
    print(f"* Running on local URL:  http://{ip}:{port}")
ip = "0.0.0.0"
demo.launch(server_name=ip, server_port=port)
