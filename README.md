# Kokoro TTS for LLM-8850

This repository contains the implementation of Kokoro TTS optimized for the **M5Stack LLM-8850 (Axera AX8850 NPU)**. It provides a high-performance, low-latency speech synthesis API compatible with the Whisplay AI Chatbot.

## Features

*   **NPU Acceleration:** Uses the AXCL Runtime (`AXCLRTExecutionProvider`) to run the 82M parameter model on the NPU.
*   **Persistent Server:** Python HTTP server (`kokoro_svr.py`) maintains the model in memory to avoid initialization overhead (~5s).
*   **API Compatibility:** Provides a JSON-based API compatible with MeloTTS implementations (`POST /synthesize`).
*   **Direct .pt Loading:** Custom patch to load PyTorch `.pt` voice files directly without requiring a full PyTorch installation.
*   **Systemd Integration:** Includes scripts to deploy as a background service.

## Prerequisites

*   **Hardware:** Raspberry Pi 5 + M5Stack LLM-8850.
*   **OS:** Linux (tested on Raspberry Pi OS).
*   **Environment:** Conda (Miniforge) with Python 3.12.
*   **Dependencies:** `axengine` (0.1.3+), `numpy`, `soundfile`, `scipy`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AndrewGraydon/kokoro.LLM8850.git
    cd kokoro.LLM8850
    ```

2.  **Ensure Model Files:**
    Make sure you have pulled the LFS files (models and voices):
    ```bash
    git lfs pull
    ```
    Verify `models/kokoro_part1_96.axmodel` etc. exist.

3.  **Setup Environment:**
    Ensure you have the `kokoro_test` environment or equivalent with `axengine` installed.

    1. Create a virtual environment and activate it 
    ```bash
    conda create -n kokoro_test python=3.12  
    conda activate kokoro  
    ```
    
    2. Install axengine (if not installed)
    ```bash
    hf download AXERA-TECH/PyAXEngine --local-dir PyAXEngine
    cd PyAXEngine
    pip install axengine-0.1.3-py3-none-any.whl
    ```
    
    3. Install project dependencies
    ```bash
    cd kokoro.LLM850
    pip install -r requirements.txt
    
    python -m spacy download en_core_web_sm
    ```
    
## Usage

### 1. Run Manually

Use the provided helper script to start the server on port **8803**:

```bash
./serve.sh
```

**Note:** The first startup takes about 5 seconds to load the models into NPU memory. Subsequent requests are handled instantly.

### 2. Run as a Service (Auto-start)

To run Kokoro TTS in the background and start on boot:

1.  Generate the service file:
    ```bash
    ./startup.sh
    ```
2.  Install and start the service:
    ```bash
    sudo mv kokoro.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable kokoro
    sudo systemctl start kokoro
    ```

3.  Check status:
    ```bash
    sudo systemctl status kokoro
    ```

## API Reference

### Endpoint: `POST /synthesize`

**Request Body (JSON):**

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `sentence` | string | Yes | - | The text to speak. |
| `voice` | string | No | `zf_xiaoyi` | Voice name (e.g., `af_heart`, `am_michael`) or file path. |
| `language` | string | No | `en` | Language code (`en` for English, `z` or `zh` for Chinese, `j` or `ja` for Japanese). |
| `speed` | float | No | `1.0` | Speaking speed. |
| `sample_rate` | int | No | `24000` | Output sample rate. |
| `outputPath` | string | No | - | If provided, saves WAV to this path on the server. If omitted, returns Base64. |

**Example Request:**
```json
{
    "sentence": "Hello, this is a test of the Kokoro TTS engine.",
    "voice": "af_heart",
    "speed": 1.0,
    "language": "en"
}
```

**Response (JSON):**
```json
{
    "success": true,
    "base64": "UklGRi..." 
}
```

## Integration with Whisplay Chatbot

To use this TTS engine with Whisplay:

1.  Edit your Whisplay `.env` file:
    ```ini
    TTS_SERVER=llm8850kokoro
    ```

2.  Ensure the host configuration is present (usually added automatically):
    ```ini
    LLM8850_KOKORO_HOST=http://localhost:8803
    ```

3.  Restart the Whisplay chatbot.

## Troubleshooting

*   **Initialization Error:** If the server fails to start, ensure no other process is using the NPU (check `axcl-smi`).
*   **Memory:** The model uses ~237MB of CMM memory. 
*   **Audio Issues:** If audio is silent or corrupt, check `server.log` for shape mismatches.
*   **Voices:** Available voices are in `checkpoints/voices/`. Use the filename without extension as the `voice` parameter (e.g., `af_heart`).

## License

Apache-2.0
