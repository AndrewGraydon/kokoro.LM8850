import argparse
import io
import json
import os
import glob
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import soundfile as sf
from kokoro_ax import Kokoro

# Load voice mappings (using .pt files as standard)
voice_list = {}
voice_candidates = glob.glob("checkpoints/voices/*.pt")
voice_list = {os.path.basename(v).replace(".pt", ""): v for v in voice_candidates}

# Global initialization for persistence
print("Initializing Kokoro TTS engine...")
axmodel_dir = "models"
config = "checkpoints/config.json"
# Ensure we use the .pt path logic from inference_utils
tts = Kokoro(axmodel_dir, config)
print("Kokoro TTS engine initialized.")

class TTSServerHandler(BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        # Update endpoint to /synthesize
        if self.path != "/synthesize":
            self._send_json({"error": "not found"}, 404)
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
             self._send_json({"error": "Content-Type must be application/json"}, 400)
             return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
             self._send_json({"error": "Empty body"}, 400)
             return

        try:
            body = self.rfile.read(length)
            params = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": f"Failed to parse JSON body: {e}"}, 400)
            return

        # Required fields
        sentence = params.get("sentence")
        if not sentence:
            self._send_json({"error": "Field 'sentence' is required"}, 400)
            return

        # Optional params with defaults
        lang = params.get("language", "en")
        
        # Simple heuristic for auto-detection if user passed 'z' or default logic was weird
        if lang == 'z': lang = 'zh'
        if lang == 'a': lang = 'en'
        if lang == 'j': lang = 'ja'
        
        # If still using a default that might be wrong, check text characters
        if lang == 'en' and any(u'\u4e00' <= c <= u'\u9fff' for c in sentence):
             lang = 'zh'


        try:
            sample_rate = int(params.get("sample_rate", 24000))
            speed = float(params.get("speed", 1.0))
        except ValueError:
            self._send_json({"error": "Invalid numeric parameters"}, 400)
            return
        
        voice_name = params.get("voice", "zf_xiaoyi")
        # Support both full path or name
        if voice_name in voice_list:
            voice_path = voice_list[voice_name]
        elif os.path.exists(voice_name):
            voice_path = voice_name
        else:
            self._send_json({"error": f"Voice {voice_name} not found"}, 400)
            return

        print(f"Synthesizing: '{sentence}' ({lang}) | Voice: {voice_name} | Speed: {speed}")

        try:
            # 1. Synthesize Audio
            audio = tts.run(sentence, language=lang, speed=speed, sample_rate=sample_rate, voice=voice_path)
            print(f"Generated audio type: {type(audio)}, shape: {audio.shape}, dtype: {audio.dtype}")
            
            # 2. Convert to WAV in memory
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV")
            wav_bytes = buf.getvalue()

            # 3. Handle Output
            output_path = params.get("outputPath")
            if output_path:
                # Save to file
                with open(output_path, "wb") as f:
                    f.write(wav_bytes)
                self._send_json({"success": True, "path": output_path})
            else:
                # Return Base64
                b64_data = base64.b64encode(wav_bytes).decode('utf-8')
                self._send_json({"success": True, "base64": b64_data})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": f"TTS failed: {str(e)}"}, 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro TTS Server")
    parser.add_argument("--port", type=int, default=8803, help="Port to run the server on")
    args = parser.parse_args()
    
    host = "0.0.0.0"
    port = args.port
    server = HTTPServer((host, port), TTSServerHandler)
    print(f"Kokoro TTS Server listening at http://{host}:{port}")
    server.serve_forever()
