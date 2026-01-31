import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=28000)
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--language", "-l", type=str, choices=["zh", "en", "ja"], default="zh")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--output", type=str, default="tts_output.wav")
args = parser.parse_args()

url = f"http://127.0.0.1:{args.port}/tts"
data = {
    "sentence": args.text,
    "language": args.language,
    "speed": args.speed,
    "sample_rate": "24000",
}

resp = requests.post(url, data=data)
if resp.status_code == 200:
    with open(args.output, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {args.output}")
else:
    print("Error:", resp.status_code, resp.text)
