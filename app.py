import modal
import os
import time
from pathlib import Path
from pydantic import BaseModel


class GenerationRequest(BaseModel):
    image: str  # URL to the source portrait image
    driven_video: str  # URL to the driving video
    prompt: str | None = None  # (Optional) text prompt


# Use the new App class instead of Stub
app = modal.App("fantasyportrait-api")

# Define persistent volumes for models and outputs
model_volume = modal.Volume.from_name(
    "fantasyportrait-models", create_if_missing=True
)
output_volume = modal.Volume.from_name(
    "fantasyportrait-outputs", create_if_missing=True
)
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

# Define the custom image with all dependencies
image = (
    # Use the official PyTorch development image which includes nvcc for compiling flash-attn
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")
    # Set environment variable to prevent download timeouts
    .env({
        "HF_HUB_ETAG_TIMEOUT": "60",
        # Help avoid CUDA memory fragmentation on long runs
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Mount the local FantasyPortrait directory into the container.
    # copy=True is required because we run pip install from this directory later.
    .add_local_dir("fantasyportrait", "/root/fantasyportrait", copy=True)
    .apt_install("git", "ffmpeg", "git-lfs", "libmagic1")
    .pip_install(
        "misaki[en]",
        "ninja", 
        "psutil", 
        "packaging",
        "flash_attn==2.7.4.post1",
        # Install other core dependencies
        "fastapi[standard]",  # Required for Modal web endpoints
        "opencv-python-headless",  # infer.py depends on cv2
        "onnx",  # required by fantasyportrait/diffsynth/models/face_utils.py
        "onnxruntime",  # CPU provider is used (gpu_id=None)
        "pydantic",
        "python-magic",
        "huggingface_hub",
        "librosa",
        # Add missing xformers dependency
        "xformers==0.0.28"
    )
    # Install all other dependencies from the FantasyPortrait requirements file.
    .pip_install_from_requirements("fantasyportrait/requirements.txt")
)

# --- CPU-only API Class for w polling ---
@app.cls(
    cpu=1.0,  # Explicitly use CPU-only containers
    image=image.pip_install("python-magic"),  # Lightweight image for API endpoints
    volumes={OUTPUT_DIR: output_volume},  # Only need output volume for reading results
)
class API:
    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    def result(self, call_id: str):
        """
        Poll for video generation results using call_id.
        Returns 202 if still processing, 200 with video if complete.
        """
        import modal
        from fastapi.responses import Response
        import fastapi.responses
        
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            # Try to get result with no timeout
            output_filename = function_call.get(timeout=0)
            
            # Read the file from the volume
            video_bytes = b"".join(output_volume.read_file(output_filename))
            
            # Return the video bytes
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )
        except TimeoutError:
            # Still processing - return HTTP 202 Accepted with no body
            return fastapi.responses.Response(status_code=202)

    @modal.fastapi_endpoint(method="HEAD", requires_proxy_auth=True)
    def result_head(self, call_id: str):
        """
        HEAD request for polling status without downloading video body.
        Returns 202 if still processing, 200 if ready.
        """
        import modal
        import fastapi.responses
        
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            # Try to get result with no timeout
            function_call.get(timeout=0)
            # If successful, return 200 with video headers but no body
            return fastapi.responses.Response(
                status_code=200,
                media_type="video/mp4"
            )
        except TimeoutError:
            # Still processing - return HTTP 202 Accepted with no body
            return fastapi.responses.Response(status_code=202)

# --- GPU Model Class ---
@app.cls(
    gpu="A100-80GB",
    enable_memory_snapshot=True, # new gpu snapshot feature: https://modal.com/blog/gpu-mem-snapshots
    experimental_options={"enable_gpu_snapshot": True},
    image=image,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    scaledown_window=2, #scale down after 2 seconds. default is 60 seconds. for testing, just scale down for now
    timeout=2700,  # 45 minutes timeout for large model downloads and initialization
)
class Model:
    def _download_and_validate(self, url: str, expected_types: list[str]) -> bytes:
        """Download content from URL and validate file type."""
        import magic
        from fastapi import HTTPException
        import urllib.request
        
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from URL {url}: {e}")
        
        # Validate file type
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_buffer(content)
        if detected_mime not in expected_types:
            expected_str = ", ".join(expected_types)
            raise HTTPException(status_code=400, detail=f"Invalid file type. Expected {expected_str}, but got {detected_mime}.")
        
        return content

    @modal.enter()  # Modal handles long initialization appropriately
    def initialize_model(self):
        """Initialize the FantasyPortrait model weights when the container starts."""
        import sys
        from pathlib import Path
        from huggingface_hub import snapshot_download, hf_hub_download

        sys.path.extend(["/root", "/root/fantasyportrait"])
        print("--- Container starting. Initializing model... ---")
        model_root = Path(MODEL_DIR)

        def download_file(repo_id: str, filename: str, local_path: Path, description: str) -> None:
            if local_path.exists():
                print(f"--- {description} already present ---")
                return
            print(f"--- Downloading {description}... ---")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_path.parent)
            print(f"--- {description} downloaded successfully ---")

        def download_repo(repo_id: str, local_dir: Path, check_file: str, description: str) -> None:
            check_path = local_dir / check_file
            if check_path.exists():
                print(f"--- {description} already present ---")
                return
            print(f"--- Downloading {description}... ---")
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
            print(f"--- {description} downloaded successfully ---")

        try:
            download_repo(
                "Wan-AI/Wan2.1-I2V-14B-720P",
                model_root / "Wan2.1-I2V-14B-720P",
                "model_index.json",
                "Wan2.1 base model",
            )
            download_file(
                "acvlab/FantasyPortrait",
                "fantasyportrait_model.ckpt",
                model_root / "fantasyportrait_model.ckpt",
                "FantasyPortrait checkpoint",
            )
            download_file(
                "acvlab/FantasyPortrait",
                "face_landmark.onnx",
                model_root / "face_landmark.onnx",
                "face landmark model",
            )
            download_file(
                "acvlab/FantasyPortrait",
                "face_det.onnx",
                model_root / "face_det.onnx",
                "face detection model",
            )
            download_file(
                "acvlab/FantasyPortrait",
                "pd_fpg.pth",
                model_root / "pd_fpg.pth",
                "expression prior",
            )
            print("--- All required files present. Committing to volume. ---")
            model_volume.commit()
            print("--- Volume committed. ---")
        except Exception as e:
            print(f"--- Initialization failed: {e} ---")
            raise
    @modal.method()
    def _generate_video(self, image: bytes, driven_video: bytes, prompt: str | None = None) -> str:
        """Internal method to generate video using FantasyPortrait and save to the output volume."""
        import sys
        import io
        import tempfile
        import time
        import uuid
        import subprocess
        import os
        from pathlib import Path
        from PIL import Image as PILImage

        sys.path.extend(["/root", "/root/fantasyportrait"])
        t0 = time.time()

        # Save source image
        source_image = PILImage.open(io.BytesIO(image)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            source_image.save(tmp_img.name, "JPEG")
            image_path = tmp_img.name

        # Save driving video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            tmp_vid.write(driven_video)
            driven_video_path = tmp_vid.name

        model_root = Path(MODEL_DIR)
        output_dir = Path(OUTPUT_DIR)
        prev_files = set(output_dir.glob("*.mp4"))

        # Determine frame count from the driven video
        try:
            import cv2
            cap = cv2.VideoCapture(driven_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            frame_count = 0
        print(f"--- Driven video frames detected: {frame_count} ---")

        # Detect source FPS to propagate into segments/stitching (keeps A/V in sync)
        src_fps = 0
        try:
            import cv2 as _cv
            _cap = _cv.VideoCapture(driven_video_path)
            src_fps = float(_cap.get(_cv.CAP_PROP_FPS) or 0)
            _cap.release()
        except Exception:
            src_fps = 0
        if not src_fps or src_fps != src_fps:  # NaN or 0
            src_fps = 25.0
        print(f"--- Source FPS detected: {src_fps} ---")

        # Sliding window parameters
        WINDOW_LEN = 117
        OVERLAP = 17

        def run_infer_segment(start_frame: int, seg_len: int) -> str:
            prev = set(output_dir.glob("*.mp4"))
            cmd = [
                "python", "-u", "/root/fantasyportrait/infer.py",
                "--portrait_checkpoint", str(model_root / "fantasyportrait_model.ckpt"),
                "--alignment_model_path", str(model_root / "face_landmark.onnx"),
                "--det_model_path", str(model_root / "face_det.onnx"),
                "--pd_fpg_model_path", str(model_root / "pd_fpg.pth"),
                "--wan_model_path", str(model_root / "Wan2.1-I2V-14B-720P"),
                "--output_path", str(output_dir),
                "--input_image_path", image_path,
                "--driven_video_path", driven_video_path,
                "--fps", str(int(src_fps)),
                "--start_frame", str(start_frame),
                "--prompt", prompt or "",
                "--scale_image", "True",
                # Reduce resolution and frames to lower VRAM usage
                "--max_size", "480",
                "--num_frames", str(seg_len),
                "--cfg_scale", "1.0",
                "--portrait_scale", "1.0",
                "--portrait_cfg_scale", "4.0",
                "--seed", "42",
                "--no_audio_merge",
            ]
            print(f"--- Launching segment: start={start_frame}, len={seg_len} ---")
            subprocess.run(cmd, check=True)
            new = set(output_dir.glob("*.mp4")) - prev
            if not new:
                raise RuntimeError("Segment generation failed")
            latest = max(new, key=lambda p: p.stat().st_mtime)
            return str(latest)

        segment_paths = []
        if frame_count and frame_count > WINDOW_LEN:
            from fantasyportrait.sliding_window import compute_segments, concat_no_blend
            segments = compute_segments(frame_count, WINDOW_LEN, OVERLAP)
            # Enforce a maximum number of windows to bound runtime
            MAX_WINDOWS = 3
            if len(segments) > MAX_WINDOWS:
                print(f"--- Too many windows ({len(segments)}). Limiting to first {MAX_WINDOWS}. ---")
                segments = segments[:MAX_WINDOWS]
            print(f"--- Using sliding windows: {segments} ---")
            for s, l in segments:
                seg_path = run_infer_segment(s, l)
                segment_paths.append(seg_path)
            stitched_path = output_dir / f"stitched_{uuid.uuid4().hex}.mp4"
            final_video_path = concat_no_blend(segment_paths, OVERLAP, int(src_fps), str(stitched_path))
            # Merge original audio onto stitched video
            from fantasyportrait.sliding_window import merge_audio_to_video
            save_video_path_with_audio = output_dir / f"stitched_{uuid.uuid4().hex}_with_audio.mp4"
            merge_audio_to_video(driven_video_path, final_video_path, str(save_video_path_with_audio))
            final_video_path = str(save_video_path_with_audio)
        else:
            # Single pass (round down to valid length)
            seg_len = frame_count if frame_count and frame_count > 0 else WINDOW_LEN
            if seg_len <= 1:
                seg_len = WINDOW_LEN
            seg_len = seg_len - ((seg_len - 1) % 4)
            only_path = run_infer_segment(0, seg_len)
            final_video_path = only_path

        output_filename = f"fantasyportrait-{uuid.uuid4().hex}.mp4"
        final_path = output_dir / output_filename
        os.rename(str(final_video_path), final_path)
        output_volume.commit()

        # Cleanup temps
        try:
            os.unlink(image_path)
        except Exception:
            pass
        try:
            os.unlink(driven_video_path)
        except Exception:
            pass
        # Keep intermediate segment and stitched files for debugging (no cleanup)
        print(f"--- Generation complete in {time.time()-t0:.2f}s ---")
        return output_filename
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def submit(self, request: "GenerationRequest"):
        """
        Submit a video generation job and return call_id for polling.
        Following Modal's recommended polling pattern for long-running tasks.
        """
        # Download and validate inputs
        image_bytes = self._download_and_validate(request.image, [
            "image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff",
        ])
        driven_bytes = self._download_and_validate(request.driven_video, [
            "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo",
            "video/webm", "video/x-ms-wmv", "video/x-flv"
        ])

        # Spawn the generation job and return call_id
        call = self._generate_video.spawn(
            image_bytes, driven_bytes, request.prompt
        )
        
        return {"call_id": call.object_id}

# --- Local Testing CLI ---
@app.local_entrypoint()
def main(
    image_path: str,
    driven_video_path: str,
    prompt: str = None,
    output_path: str = "outputs/test.mp4",
):
    """
    A local CLI to generate a FantasyPortrait video from local files or URLs.

    Example:
    modal run app.py --image-path "url/to/image.png" --driven-video-path "url/to/video.mp4"
    """
    import base64
    import urllib.request

    print(f"--- Starting generation for {image_path} ---")
    print(f"--- Current working directory: {os.getcwd()} ---")
    print(f"--- Output path: {output_path} ---")

    def _read_input(path: str) -> bytes:
        if path.startswith(("http://", "https://")):
            return urllib.request.urlopen(path).read()
        else:
            with open(path, "rb") as f:
                return f.read()

    # --- Read inputs (validation only happens on remote Modal containers) ---
    image_bytes = _read_input(image_path)
    driven_bytes = _read_input(driven_video_path)

    # --- Run model ---
    model = Model()
    output_filename = model._generate_video.remote(
        image_bytes, driven_bytes, prompt
    )

    # --- Save output ---
    print(f"--- Reading '{output_filename}' from volume... ---")
    video_bytes = b"".join(output_volume.read_file(output_filename))
    
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    
    print(f"🎉 --- Video saved to {output_path} ---") 
