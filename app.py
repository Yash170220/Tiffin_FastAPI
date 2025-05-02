from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

# Pydantic schemas
class ImageRequest(BaseModel):
    image_url: str

class AnalysisResponse(BaseModel):
    description: str
    nutrients: dict

# Download image from URL
def download_image(url: str) -> Image.Image:
    try:
        response = httpx.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {e}")

# Analyze image using Gemini
def analyze_image(image: Image.Image) -> AnalysisResponse:
    prompt = (
        "Analyze this food image. First, give a brief description.Just describe the food items you see in the image."
        "An the description should be regarding the food item not what to you see in image not just the image description. "
        "Give atleast 3 sentences for the description"
        "Describe just the food you detect not the image."
        "Then estimate basic nutritional values "
        "Do not the image shows and all things just start to describe"

        "(calories, protein, fat, and carbs). Respond in JSON like this:\n"
        "{\n"
        "  \"description\": \"...\",\n"
        "  \"nutrients\": {\n"
        "    \"calories\": ..., \"protein\": \"...g\", \"fat\": \"...g\", \"carbs\": \"...g\"\n"
        "  }\n"
        "}"
    )
    try:
        result = model.generate_content([prompt, image])
        text = result.text
        start, end = text.find('{'), text.rfind('}') + 1
        import json
        parsed = json.loads(text[start:end])
        return AnalysisResponse(
            description=parsed.get("description", "No description"),
            nutrients=parsed.get("nutrients", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {e}")

# Endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: ImageRequest):
    image = download_image(request.image_url)
    return analyze_image(image)
