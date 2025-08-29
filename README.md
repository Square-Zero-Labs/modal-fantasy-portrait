# FantasyPortrait on Modal

This project deploys the [Fantasy-AMAP/FantasyPortrait](https://github.com/Fantasy-AMAP/fantasy-portrait) model on [Modal](https://modal.com), providing a high-performance API for animating portrait images with driving videos.

The deployment uses L40S GPUs and caches model files in Modal volumes for efficient inference.

## Prerequisites

1. **Create a Modal Account:** Sign up at [modal.com](https://modal.com).
2. **Install Modal Client:**
   ```bash
   pip install modal
   modal token new
   ```

## Setup

The FantasyPortrait code is included via a `git subtree`:

```bash
git subtree add --prefix fantasyportrait https://github.com/Fantasy-AMAP/fantasy-portrait main --squash
```

To update the subtree:

```bash
git subtree pull --prefix fantasyportrait https://github.com/Fantasy-AMAP/fantasy-portrait main --squash
```

On first run the app downloads required models and stores them in the `fantasyportrait-models` volume:

- [Wan2.1-I2V-14B-720P base model](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)
- [FantasyPortrait expression weights](https://huggingface.co/acvlab/FantasyPortrait)
- `face_landmark.onnx`, `face_det.onnx`, `pd_fpg.pth`

## Deployment

Two Modal volumes are used:

- `fantasyportrait-models` for cached model files
- `fantasyportrait-outputs` for generated videos

Deploy the web endpoint:

```bash
modal deploy app.py
```

Modal builds a container image, downloads the models, and exposes an API for inference.

## Usage

### 1. Web API Endpoint

Send a `POST` request with proxy authentication. The JSON payload accepts:

- `image` (string, required): URL to the source portrait image.
- `driven_video` (string, required): URL to the driving video (MP4).
- `prompt` (string, optional): text description.

```bash
CALL_ID=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $TOKEN_ID" \
  -H "Modal-Secret: $TOKEN_SECRET" \
  -d '{
        "image": "https://example.com/face.jpg",
        "driven_video": "https://example.com/driver.mp4",
        "prompt": "a person smiling"
      }' \
  "https://<username>--fantasyportrait-api-model-submit.modal.run" | jq -r '.call_id')
```

Poll for completion:

```bash
HTTP_STATUS=$(curl -w "%{http_code}" -s --head \
  -H "Modal-Key: $TOKEN_ID" \
  -H "Modal-Secret: $TOKEN_SECRET" \
  "https://<username>--fantasyportrait-api-api-result-head.modal.run?call_id=$CALL_ID")
echo "HTTP $HTTP_STATUS"
```

Retrieve the finished video:

```bash
curl -X GET \
  -H "Modal-Key: $TOKEN_ID" \
  -H "Modal-Secret: $TOKEN_SECRET" \
  --output outputs/api-generated_video.mp4 \
  "https://<username>--fantasyportrait-api-api-result.modal.run?call_id=$CALL_ID"
```

### 2. Local Testing CLI

```bash
modal run app.py --image-path "https://example.com/face.jpg" \
  --driven-video-path "https://example.com/driver.mp4" \
  --prompt "smiling portrait" \
  --output-path outputs/demo.mp4
```

## Resources

- [FantasyPortrait Paper](https://arxiv.org/abs/2507.12956)
- [FantasyPortrait Dataset](https://huggingface.co/datasets/acvlab/FantasyPortrait-Multi-Expr)
- [Model Weights](https://huggingface.co/acvlab/FantasyPortrait)
