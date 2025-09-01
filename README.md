# FantasyPortrait on Modal

This project deploys the [Fantasy-AMAP/FantasyPortrait](https://github.com/Fantasy-AMAP/fantasy-portrait) model on [Modal](https://modal.com), providing a high-performance API for animating portrait images with driving videos.

The deployment caches model files in Modal volumes for efficient inference.

## Prerequisites

1. **Clone this Repository:**

   ```bash
   git clone https://github.com/Square-Zero-Labs/modal-fantasy-portrait
   cd modal-fantasy-portrait
   ```

2. **Create a Modal Account:** Sign up at [modal.com](https://modal.com).

3. **Install Modal Client:**

```bash
pip install modal
modal setup
```

## Deployment

Deploy the web endpoint:

```bash
pip install pydantic
modal deploy app.py
```

Modal builds a container image and exposes an API for inference.
Two Modal volumes are used:

- `fantasyportrait-models` for cached model files
- `fantasyportrait-outputs` for generated videos

## Usage

### Local Testing CLI

```bash
modal run app.py --image-path "https://example.com/portrait.jpg"\
  --driven-video-path "https://example.com/driving-video.mp4"\
  --prompt "The person is talking"\
  --output-path outputs/demo.mp4
```

**Note:** On first run, the app automatically downloads required models and stores them in the `fantasyportrait-models` volume:

- [Wan2.1-I2V-14B-720P base model](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)
- [FantasyPortrait expression weights](https://huggingface.co/acvlab/FantasyPortrait)

### Using the API

First, you'll need to set up authentication.

## Authentication

The API requires proxy authentication tokens.

To create proxy auth tokens, go to your Modal workspace settings and generate a new proxy auth token. Set the token ID and secret as environment variables:

```bash
export TOKEN_ID="your-token-id"
export TOKEN_SECRET="your-token-secret"
```

## Making API Calls

### Submit a Request

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

## Resources

- [Video Tutorial](https://youtu.be/UdiiEXZV-10)
- [FantasyPortrait Paper](https://arxiv.org/abs/2507.12956)
- [FantasyPortrait Repo](https://github.com/Fantasy-AMAP/fantasy-portrait)

## Development Notes

The FantasyPortrait code is included via a `git subtree`:

### Git Subtree Management

When originally added:

```bash
git subtree add --prefix fantasyportrait https://github.com/Fantasy-AMAP/fantasy-portrait main --squash
```

To update the subtree:

```bash
git subtree pull --prefix fantasyportrait https://github.com/Fantasy-AMAP/fantasy-portrait main --squash
```
