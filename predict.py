import json
import os
import random
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from cog import BaseModel, BasePredictor, Input, Path
import requests
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_PYTHON = "/root/comfy_env/bin/python"

# Asset Paths
YUE_CKPT_URL = "https://huggingface.co/Comfy-Org/YuE2/resolve/main/checkpoints/yue2_3b_int8_convrot.safetensors"
YUE_CKPT_PATH = (
    "/root/ComfyUI/models/checkpoints/yue2_3b_int8_convrot.safetensors"
)

SHEETSAGE_URL = "https://huggingface.co/Comfy-Org/YuE2/resolve/main/audio_encoders/sheetsage2_bf16.safetensors"
SHEETSAGE_PATH = (
    "/root/ComfyUI/models/audio_encoders/sheetsage2_bf16.safetensors"
)

DEFAULT_STYLE = "1980s japanese city pop, upbeat funk groove, 118 bpm, punchy gated snare, funky slap bassline, sparkling dx7 electric piano, crisp brass section stabs, bright chorus rhythm guitar, clear nostalgic female pop vocals"


DEFAULT_LYRICS = """[intro - sparkling DX7 keys, bright brass stabs, funky slap bass]

[verse]
レインボウ・ブリッジを渡る風 (Reinboo burijji wo wataru kaze)
City lights reflected in your sunglasses
ラジオから流れるサマー・チューン (Rajio kara nagareru samaa chuun)
Chasing twilight as the hour passes

[chorus]
Stay with me, stay with me, 夢の中で (yume no naka de)
Dancing together under neon rain!
Don't say goodbye.
"""


class Output(BaseModel):
  audio: Path
  score_abc: str


class Predictor(BasePredictor):

  def setup(self):
    """Downloads checkpoints and boots isolated ComfyUI server."""
    os.makedirs(os.path.dirname(YUE_CKPT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SHEETSAGE_PATH), exist_ok=True)
    os.makedirs("/root/ComfyUI/input", exist_ok=True)

    # 1. Download YuE2 Model if missing
    if (
        not os.path.exists(YUE_CKPT_PATH)
        or os.path.getsize(YUE_CKPT_PATH) < 3_500_000_000
    ):
      print("Downloading YuE2 INT8 checkpoint (~3.96 GB)...")
      subprocess.run(
          ["wget", "-c", "--progress=dot:giga", YUE_CKPT_URL, "-O", YUE_CKPT_PATH],
          check=True,
      )

    # 2. Download SheetSage2 Audio Encoder if missing
    if (
        not os.path.exists(SHEETSAGE_PATH)
        or os.path.getsize(SHEETSAGE_PATH) < 1_200_000_000
    ):
      print("Downloading SheetSage2 audio encoder (~1.39 GB)...")
      subprocess.run(
          [
              "wget",
              "-c",
              "--progress=dot:giga",
              SHEETSAGE_URL,
              "-O",
              SHEETSAGE_PATH,
          ],
          check=True,
      )

    # 3. Boot background ComfyUI instance
    print("Starting ComfyUI Studio server in isolated virtualenv...")
    cmd = [
        COMFY_PYTHON,
        "/root/ComfyUI/main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--fast",
        "fp16_accumulation",
        "--verbose",
        "INFO",
    ]
    self.comfy_process = subprocess.Popen(cmd)

    ready = False
    for _ in range(60):
      if self.comfy_process.poll() is not None:
        raise RuntimeError(
            f"ComfyUI process exited prematurely with code"
            f" {self.comfy_process.returncode}"
        )
      try:
        res = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=1)
        if res.status_code == 200:
          ready = True
          break
      except Exception:
        time.sleep(1)

    if not ready:
      raise RuntimeError("ComfyUI failed to start within 60 seconds.")
    print("ComfyUI Studio server is online.")

  def predict(
      self,
      reference_audio: Path = Input(
          description=(
              "(Optional) Upload reference song (MP3/WAV/FLAC) to create a"
              " cover/remix. SheetSage2 transcribes its melody to ABC."
          ),
          default=None,
      ),
      cover_mode: str = Input(
          description=(
              "Cover transcription mode: 'melody' extracts the main lead/vocal"
              " hook; 'full' extracts chords + harmony."
          ),
          choices=["melody", "full"],
          default="melody",
      ),
      style: str = Input(
          description="Genre, instruments, mood, tempo, vocal character",
          default=DEFAULT_STYLE,
      ),
      lyrics: str = Input(
          description=(
              "Song lyrics or bracketed musical structure tags ([intro], [solo],"
              " etc.)"
          ),
          default=DEFAULT_LYRICS,
      ),
      audio_format: str = Input(
          description="Audio output format",
          choices=["mp3", "wav", "flac"],
          default="mp3",
      ),
      cfg_scale: float = Input(
          description=(
              "Classifier-Free Guidance. 1.0 = fastest/cheapest single-pass; 1.2"
              " - 1.5 = stronger prompt adherence."
          ),
          ge=1.0,
          le=2.5,
          default=1.2,
      ),
      repetition_penalty: float = Input(
          description=(
              "Penalizes repetitive vocal phrases. Increase to 1.25 - 1.35 if"
              " vocals loop."
          ),
          ge=1.0,
          le=2.0,
          default=1.20,
      ),
      max_duration: float = Input(
          description="Target song duration in seconds",
          ge=15.0,
          le=360.0,
          default=60.0,
      ),
      temperature: float = Input(
          description="Creativity and randomness of generation",
          ge=0.1,
          le=2.0,
          default=1.0,
      ),
      steps: int = Input(
          description="Acoustic diffusion steps (18-22 fast, 32 standard)",
          ge=10,
          le=100,
          default=32,
      ),
      sampler_name: str = Input(
          description="Diffusion solver algorithm",
          choices=["dpm_2", "euler", "dpmpp_2m"],
          default="dpm_2",
      ),
      scheduler: str = Input(
          description="Noise reduction schedule curve",
          choices=["sgm_uniform", "karras", "simple"],
          default="sgm_uniform",
      ),
      seed: int = Input(
          description="Random seed for reproducibility (-1 for random)",
          default=-1,
      ),
  ) -> Output:
    """Runs Studio music generation with dynamic SheetSage2 cover routing."""
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing Studio job with seed: {seed}")

    formatted_lyrics = lyrics.replace("\\n", "\n").strip()

    # Load base studio graph
    with open("workflow_api.json", "r", encoding="utf-8") as f:
      prompt = json.load(f)

    # --- DYNAMIC ROUTING BRANCH ---
    if reference_audio and os.path.exists(str(reference_audio)):
      print("Reference audio detected. Activating SheetSage2 Cover pipeline...")
      # Stage audio file in ComfyUI input folder
      ref_filename = f"ref_{uuid.uuid4().hex[:8]}.wav"
      ref_dest = os.path.join("/root/ComfyUI/input", ref_filename)
      shutil.copyfile(str(reference_audio), ref_dest)

      # Configure Node 20 (LoadAudio) & Node 19 (SheetSage2AudioToABC)
      prompt["20"]["inputs"]["audio"] = ref_filename
      prompt["19"]["inputs"]["mode"] = cover_mode

      # Route SheetSage2 ABC output directly into Node 22 (YuE2GenerateMusic)
      prompt["22"]["inputs"]["abc"] = ["19", 0]

      # Remove Node 23 (Text ABC planner) so it doesn't run unnecessarily
      if "23" in prompt:
        del prompt["23"]
    else:
      print("No reference audio. Running standard text-to-music pipeline...")
      # Route Node 23 (YuE2GenerateABC) into Node 22
      prompt["22"]["inputs"]["abc"] = ["23", 0]
      if "23" in prompt:
        prompt["23"]["inputs"]["style"] = style
        prompt["23"]["inputs"]["lyrics"] = formatted_lyrics
        prompt["23"]["inputs"]["seed"] = seed

      # Delete unused audio cover nodes so ComfyUI doesn't execute them
      for unused_node in ["18", "19", "20"]:
        if unused_node in prompt:
          del prompt[unused_node]

    # Configure Node 22 (YuE2GenerateMusic)
    prompt["22"]["inputs"]["style"] = style
    prompt["22"]["inputs"]["lyrics"] = formatted_lyrics
    prompt["22"]["inputs"]["max_duration"] = max_duration
    prompt["22"]["inputs"]["temperature"] = temperature
    prompt["22"]["inputs"]["repetition_penalty"] = repetition_penalty
    prompt["22"]["inputs"]["cfg_scale"] = cfg_scale
    prompt["22"]["inputs"]["seed"] = seed

    # Configure Node 8 (KSampler)
    prompt["8"]["inputs"]["steps"] = steps
    prompt["8"]["inputs"]["sampler_name"] = sampler_name
    prompt["8"]["inputs"]["scheduler"] = scheduler
    prompt["8"]["inputs"]["cfg"] = cfg_scale
    prompt["8"]["inputs"]["seed"] = seed

    # Submit job to local ComfyUI instance
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    response = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response["prompt_id"]

    # Wait for completion
    while True:
      out = ws.recv()
      if isinstance(out, str):
        message = json.loads(out)
        if message["type"] == "executing":
          data = message["data"]
          if data["node"] is None and data["prompt_id"] == prompt_id:
            break
      else:
        continue
    ws.close()

    # Discover generated audio
    output_root = "/root/ComfyUI/output"
    found_audio = []
    found_text = []

    for root, _, files in os.walk(output_root):
      for f in files:
        full_path = os.path.join(root, f)
        if f.endswith((".flac", ".wav", ".mp3")):
          found_audio.append(full_path)
        elif f.endswith((".abc", ".txt")):
          found_text.append(full_path)

    if not found_audio:
      raise RuntimeError("No audio file was produced.")

    latest_audio = max(found_audio, key=os.path.getmtime)

    abc_text = ""
    if found_text:
      latest_text = max(found_text, key=os.path.getmtime)
      with open(latest_text, "r", encoding="utf-8") as f:
        abc_text = f.read()

    # Transcode audio format
    final_output = latest_audio
    output_dir = os.path.dirname(latest_audio)

    if audio_format == "mp3" and not latest_audio.endswith(".mp3"):
      final_output = os.path.join(output_dir, f"song_{prompt_id}.mp3")
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, "-b:a", "320k", final_output],
          check=True,
      )
    elif audio_format == "wav" and not latest_audio.endswith(".wav"):
      final_output = os.path.join(output_dir, f"song_{prompt_id}.wav")
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, final_output], check=True
      )

    return Output(audio=Path(final_output), score_abc=abc_text)