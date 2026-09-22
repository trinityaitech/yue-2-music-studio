import json
import os
import random
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from typing import Optional
from cog import BaseModel, BasePredictor, Input, Path
import requests

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

DEFAULT_STYLE = (
    "1980s japanese city pop, upbeat funk groove, 118 bpm, punchy gated snare,"
    " funky slap bassline, sparkling dx7 electric piano, crisp brass section"
    " stabs, bright chorus rhythm guitar, clear nostalgic female pop vocals"
)

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

    if (
        not os.path.exists(YUE_CKPT_PATH)
        or os.path.getsize(YUE_CKPT_PATH) < 3_500_000_000
    ):
      print("Downloading YuE2 INT8 checkpoint (~3.96 GB)...")
      subprocess.run(
          ["wget", "-c", "--progress=dot:giga", YUE_CKPT_URL, "-O", YUE_CKPT_PATH],
          check=True,
      )

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
      style: str = Input(
          description="Genre, instruments, mood, tempo, vocal character",
          default=DEFAULT_STYLE,
      ),
      lyrics: str = Input(
          description=(
              "Song lyrics or bracketed musical structure tags ([verse],"
              " [chorus], etc.)"
          ),
          default=DEFAULT_LYRICS,
      ),
      cot: str = Input(
          description=(
              "Symbolic planning mode: 'full' (chords+melody), 'melody' (melody"
              " only), 'off' (direct synthesis without score planning)"
          ),
          choices=["full", "melody", "off"],
          default="full",
      ),
      reference_audio: Optional[Path] = Input(
          description=(
              "(Optional) Upload an audio file (MP3/WAV) to use as a cover/remix"
              " source. Leave blank for standard text-to-music."
          ),
          default=None,
      ),
      cover_mode: str = Input(
          description=(
              "If reference audio is provided: 'melody' extracts lead vocal"
              " hook; 'full' extracts chords + harmony."
          ),
          choices=["melody", "full"],
          default="melody",
      ),
      audio_format: str = Input(
          description="Audio output format",
          choices=["mp3", "wav", "flac"],
          default="mp3",
      ),
      cfg_scale: float = Input(
          description="Text prompt guidance scale (1.0 to 1.5 recommended)",
          ge=1.0,
          le=2.5,
          default=1.2,
      ),
      repetition_penalty: float = Input(
          description="Penalizes repetitive vocal loops. Increase if needed.",
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
    """Runs Studio music generation with robust completion detection."""
    if seed < 0:
      seed = random.randint(0, 2**32 - 1)
    print(f"Executing Studio job with seed: {seed}")

    formatted_lyrics = lyrics.replace("\\n", "\n").strip()

    with open("workflow_api.json", "r", encoding="utf-8") as f:
      prompt = json.load(f)

    # DYNAMIC ROUTING: Audio Cover vs Text Generation
    if reference_audio and os.path.exists(str(reference_audio)):
      print("Reference audio detected. Routing through SheetSage2...")
      ref_filename = f"ref_{uuid.uuid4().hex[:8]}.wav"
      ref_dest = os.path.join("/root/ComfyUI/input", ref_filename)
      shutil.copyfile(str(reference_audio), ref_dest)

      prompt["20"]["inputs"]["audio"] = ref_filename
      prompt["19"]["inputs"]["mode"] = cover_mode
      prompt["22"]["inputs"]["abc"] = ["19", 0]

      if "23" in prompt:
        del prompt["23"]
    else:
      print(f"Running text-to-music pipeline with CoT mode: {cot}")
      prompt["22"]["inputs"]["abc"] = ["23", 0]
      if "23" in prompt:
        prompt["23"]["inputs"]["style"] = style
        prompt["23"]["inputs"]["lyrics"] = formatted_lyrics
        prompt["23"]["inputs"]["mode"] = cot
        prompt["23"]["inputs"]["seed"] = seed

      for unused_node in ["18", "19", "20"]:
        if unused_node in prompt:
          del prompt[unused_node]

    # Parameters for Node 22 (YuE2GenerateMusic)
    prompt["22"]["inputs"]["style"] = style
    prompt["22"]["inputs"]["lyrics"] = formatted_lyrics
    prompt["22"]["inputs"]["mode"] = cot
    prompt["22"]["inputs"]["max_duration"] = max_duration
    prompt["22"]["inputs"]["temperature"] = temperature
    prompt["22"]["inputs"]["repetition_penalty"] = repetition_penalty
    prompt["22"]["inputs"]["cfg_scale"] = cfg_scale
    prompt["22"]["inputs"]["seed"] = seed

    # Parameters for Node 8 (KSampler)
    prompt["8"]["inputs"]["steps"] = steps
    prompt["8"]["inputs"]["sampler_name"] = sampler_name
    prompt["8"]["inputs"]["scheduler"] = scheduler
    prompt["8"]["inputs"]["cfg"] = cfg_scale
    prompt["8"]["inputs"]["seed"] = seed

    # Submit prompt to ComfyUI
    client_id = str(uuid.uuid4())
    payload = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    response = json.loads(urllib.request.urlopen(req).read())
    prompt_id = response["prompt_id"]

    # Robust Polling: Poll /history endpoint directly to prevent hanging WebSocket connections
    max_wait = 720  # 12 minute safety ceiling
    elapsed = 0
    completed = False
    history_outputs = {}

    while elapsed < max_wait:
      try:
        hist_res = requests.get(
            f"http://{COMFY_HOST}/history/{prompt_id}", timeout=3
        )
        if hist_res.status_code == 200:
          hist_json = hist_res.json()
          if prompt_id in hist_json:
            completed = True
            history_outputs = hist_json[prompt_id].get("outputs", {})
            break
      except Exception:
        pass

      time.sleep(2)
      elapsed += 2

    if not completed:
      raise RuntimeError(
          f"ComfyUI generation timed out after {max_wait} seconds."
      )

    # Retrieve generated ABC notation from Stage 1 (Node 23) if available
    abc_text = ""
    if "23" in history_outputs:
      for val in history_outputs["23"].values():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
          abc_text = val[0]
          break

    # Locate generated audio file
    output_root = "/root/ComfyUI/output"
    found_audio = []
    found_text = []

    for root, _, files in os.walk(output_root):
      for f in files:
        full_path = os.path.join(root, f)
        if f.endswith((".flac", ".wav", ".mp3", ".mpga")):
          found_audio.append(full_path)
        elif f.endswith((".abc", ".txt")):
          found_text.append(full_path)

    if not found_audio:
      raise RuntimeError(
          "No audio file was produced in the ComfyUI output directory."
      )

    latest_audio = max(found_audio, key=os.path.getmtime)

    # Fallback score read from disk if history API didn't expose it
    if not abc_text and found_text:
      latest_text = max(found_text, key=os.path.getmtime)
      with open(latest_text, "r", encoding="utf-8") as f:
        abc_text = f.read()

    # Transcode audio format with explicit extension naming
    output_dir = os.path.dirname(latest_audio)

    if audio_format == "mp3":
      final_output = os.path.join(output_dir, f"song_{prompt_id}.mp3")
      subprocess.run(
          [
              "ffmpeg",
              "-y",
              "-i",
              latest_audio,
              "-codec:a",
              "libmp3lame",
              "-b:a",
              "320k",
              final_output,
          ],
          check=True,
      )
    elif audio_format == "wav":
      final_output = os.path.join(output_dir, f"song_{prompt_id}.wav")
      subprocess.run(
          ["ffmpeg", "-y", "-i", latest_audio, final_output],
          check=True,
      )
    else:
      final_output = latest_audio

    return Output(audio=Path(final_output), score_abc=abc_text)