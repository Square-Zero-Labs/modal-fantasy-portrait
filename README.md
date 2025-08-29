# FantasyPortrait on Modal

This project deploys the [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) model on [Modal](https://modal.com) to provide a high-performance, scalable API for generating talking head videos from an image and audio files.

The deployment is optimized for efficient inference, leveraging:

- A100 GPUs
- `FusioniX` LoRA optimization
- `flash-attention`
- `teacache`

Note that this does not fully implement the multi-speaking capabilities since that requires downloading separate model weights and we wanted to focus on single person for now.

## Prerequisites

1.  **Create a Modal Account:** Sign up for a free account at [modal.com](https://modal.com).
2.  **Install Modal Client:** Install the Modal client library and set up your authentication token.
    ```bash
    pip install modal
    modal token new
    ```

## Setup

This project uses `git subtree` to incorporate the original `InfiniteTalk` source code. To set this up from a fresh clone, run the following command:

```bash
git subtree add --prefix infinitetalk https://github.com/MeiGen-AI/InfiniteTalk main --squash
```

### Updating the `InfiniteTalk` Source Code

If the original `InfiniteTalk` repository is updated and you want to incorporate those changes into this project, you can pull the updates using the following command:

```bash
git subtree pull --prefix infinitetalk https://github.com/MeiGen-AI/InfiniteTalk main --squash
```

## Deployment

The application consists of a persistent web endpoint for production use and a local CLI for testing. It uses a `Volume` to cache the large model files, ensuring they are only downloaded once. A second `Volume` is used to efficiently handle the video outputs.

To deploy the web endpoint, run the following command from your terminal:

```bash
modal deploy app.py
```

Modal will build a custom container image, download the model weights into a persistent `Volume`, and deploy the application. After a successful deployment, it will provide a public URL for your API endpoint.

The initial deployment will take several minutes as it needs to download the large model files. Subsequent deployments will be much faster as the models are cached in the volume. This version uses a new volume name (`infinitetalk-models`) to accommodate the updated model loading strategy.

## Usage

### 1. Web API Endpoint

The deployed service can be called via a `POST` request with proxy authentication. The API accepts a JSON payload with the following fields:

- `image` (string, required): A URL to the source image (JPEG or PNG).
- `audio1` (string, required): A URL to the first audio file (MP3 or WAV).
- `prompt` (string, optional): A text prompt describing the desired interaction.

The duration of the video is automatically determined by the length of the input audio. When using two audio files, they are processed so that each speaker's audio is temporally separated (the first audio plays, then the second audio plays). The video frame count is calculated to match this combined duration while adhering to the model's `4n+1` frame constraint.

**Key Differences from Original InfiniteTalk:**

- **Dynamic frame calculation**: Frame count is calculated based on audio length instead of using a fixed value
- **LoRA integration**: Uses FusioniX LoRA (`quant_model_int8_FusionX.safetensors`) with quantized models for optimal performance
- **Optimized guidance scales**: Uses `text_guide_scale=1.0` and `audio_guide_scale=2.0` as recommended for LoRA usage
- **Streaming mode**: Automatically uses streaming mode for long video generation

#### Authentication

The API requires proxy authentication tokens.

To create proxy auth tokens, go to your Modal workspace settings and generate a new token. Set the token ID and secret as environment variables:

```bash
export TOKEN_ID="your-token-id"
export TOKEN_SECRET="your-token-secret"
```

**API Usage - Polling Pattern:**

Following [Modal's recommended polling pattern](https://modal.com/docs/guide/webhook-timeouts), we use two endpoints for long-running video generation:

1. **Submit Job** - Starts generation and returns call_id
2. **Poll Results** - Check status and download video when ready

**Step 1: Submit Job**

```bash
# Submit video generation job and capture call_id
CALL_ID=$(curl -s -X POST \
     -H "Content-Type: application/json" \
     -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     -d '{
           "image": "https://replicate.delivery/pbxt/NHF6Y7EeJGK6pp4rDODjJS8m0nk9rj32iuaVQs8IfOl7S4vJ/multi1.png",
           "audio1": "https://replicate.delivery/pbxt/NHF6XifveoBBNUVcYdrkqkiLqq2vDI7g322dYXadTtF4BFZ9/1.WAV",
           "prompt": "A person is talking"
         }' \
     "https://squarezerolabs--infinitetalk-api-model-submit.modal.run" | jq -r '.call_id')

echo "Job submitted with call_id: $CALL_ID"
```

**Step 2: Poll for Results**

```bash
# Poll for results - downloads video on success, shows status on failure
HTTP_STATUS=$(curl -w "%{http_code}" -s --head \
     -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     "https://squarezerolabs--infinitetalk-api-api-result-head.modal.run?call_id=$CALL_ID")
echo "HTTP $HTTP_STATUS"
```

- `202 Accepted` - Shows processing status in terminal only
- `200 OK` - Downloads video to `outputs/generated_video.mp4`

**Step 3: Retrieve Finished Video**

```bash
curl -X GET \
         -H "Modal-Key: $TOKEN_ID" \
         -H "Modal-Secret: $TOKEN_SECRET" \
         --output outputs/api-generated_video.mp4 \
         "https://squarezerolabs--infinitetalk-api-api-result.modal.run?call_id=$CALL_ID"
    echo "Video saved to outputs/api-generated_video.mp4"
```

Replace:

- `squarezerolabs` with your actual Modal username
- `$TOKEN_ID` and `$TOKEN_SECRET` with your proxy auth token credentials

The URL format is `https://[username]--[app-name]-[class-name]-[method-name].modal.run` where the class-name is `model` or `api` and the method-name are the methods defined in app.py.

### 2. Local Testing CLI

For development and testing, you can use the built-in command-line interface to run inference on local files or URLs.

**Example Image to Video Usage:**

One Speaker - multitalk demo example:

```bash
modal run app.py --image-path "https://replicate.delivery/pbxt/NHF6Y7EeJGK6pp4rDODjJS8m0nk9rj32iuaVQs8IfOl7S4vJ/multi1.png" --audio1-path "https://replicate.delivery/pbxt/NHF6XifveoBBNUVcYdrkqkiLqq2vDI7g322dYXadTtF4BFZ9/1.WAV" --prompt "A man talking in a car" --output-path outputs/my_video.mp4
```

One speaker - superhero (1 second):

```bash
modal run app.py --image-path "https://storage.googleapis.com/4public-testing-files4200/superhero.png" --audio1-path "https://replicate.delivery/pbxt/NHF6XifveoBBNUVcYdrkqkiLqq2vDI7g322dYXadTtF4BFZ9/1.WAV" --prompt "super hero speaking" --output-path outputs/superman-one-voice.mp4
```

One speaker - woman reading (6 seconds):

```bash
modal run app.py --image-path "https://storage.googleapis.com/4public-testing-files4200/woman-reading.png" --audio1-path "https://storage.googleapis.com/4public-testing-files4200/anna-k-first-sentence-trimmed.mp3" --prompt "A woman reads a book aloud" --output-path outputs/woman-reading.mp4
```

One speaker - woman reading (30 seconds):

```bash
modal run app.py --image-path "https://storage.googleapis.com/4public-testing-files4200/woman-reading.png" --audio1-path "https://storage.googleapis.com/4public-testing-files4200/anna-k-30-seconds.mp3" --prompt "A woman reads a book aloud" --output-path outputs/woman-reading.mp4
```

**Example Video to Video Usage:**

```bash
modal run app.py --image-path "https://storage.googleapis.com/4public-testing-files4200/dog-8-seconds.mp4" --audio1-path "https://storage.googleapis.com/4public-testing-files4200/anna-k-first-sentence-trimmed.mp3" --prompt "A dog reads a book aloud" --output-path outputs/dog-reading.mp4
```
