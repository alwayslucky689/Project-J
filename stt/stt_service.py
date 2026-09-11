# stt_service.py - Persistent Parakeet transcription service
#
# Reads WAV file paths from stdin (one per line).
# Writes the transcription to stdout (one line per input).
#
# Run this from inside venv_stt:
#   python stt_service.py
#
# The main assistant launches this as a subprocess and talks to it via pipes.

import sys
import os
import time

# Suppress NeMo's noisy logging
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
os.environ["HYDRA_FULL_ERROR"] = "0"

# Quiet down other loggers
import logging
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")


def log(msg):
    """Print to stderr (safe — doesn't interfere with stdout protocol)."""
    print(msg, file=sys.stderr, flush=True)


def main():
    # Send a READY signal so the parent knows when we're loaded
    log("[stt_service] Starting...")
    
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        log(f"[stt_service] FATAL: cannot import NeMo: {e}")
        sys.exit(1)
    
    log("[stt_service] Loading Parakeet TDT 0.6B v2...")
    start = time.time()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2"
        )
        model.eval()
    except Exception as e:
        log(f"[stt_service] FATAL: model load failed: {e}")
        sys.exit(1)
    
    log(f"[stt_service] Model loaded in {time.time() - start:.1f}s on {model.device}")
    log(f"[stt_service] READY")
    
    # Signal READY to parent via stdout (single line)
    print("READY", flush=True)
    
    # Main loop: read path, transcribe, write result
    for line in sys.stdin:
        path = line.strip()
        
        if not path:
            continue
        
        if path == "__EXIT__":
            log("[stt_service] Received EXIT command, shutting down.")
            break
        
        if not os.path.exists(path):
            print(f"__ERROR__ file not found: {path}", flush=True)
            continue
        
        try:
            t0 = time.time()
            output = model.transcribe([path])
            text = output[0].text.strip()
            elapsed = time.time() - t0
            log(f"[stt_service] Transcribed in {elapsed:.2f}s: {text!r}")
            
            # Sanitize: replace newlines with spaces (protocol requires one line)
            text = text.replace("\n", " ").replace("\r", " ")
            
            print(text, flush=True)
            
        except Exception as e:
            log(f"[stt_service] ERROR transcribing {path}: {e}")
            print(f"__ERROR__ {e}", flush=True)
    
    log("[stt_service] Exiting cleanly.")


if __name__ == "__main__":
    main()