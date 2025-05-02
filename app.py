from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

# Allow CORS for all origins (customize in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class ImageRequest(BaseModel):
    image_url: str

class SummaryResponse(BaseModel):
    summary: str

# Download image from URL
def download_image(url: str) -> Image.Image:
    try:
        response = httpx.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {e}")

# Analyze image using Gemini
def analyze_image(image: Image.Image) -> SummaryResponse:
    prompt = (
        "Analyze this food image. First, give a brief description of the food items detected, not the image itself."
        "The description should be at least 3 sentences. Then estimate the basic nutritional values "
        "(calories, protein, fat, and carbs). Respond in JSON format like this:\n"
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
        parsed = json.loads(text[start:end])
        
        description = parsed.get("description", "No description")
        nutrients = parsed.get("nutrients", {})
        nutrient_summary = (
            f"It contains around {nutrients.get('calories', 'N/A')} calories, "
            f"{nutrients.get('protein', 'N/A')} protein, "
            f"{nutrients.get('fat', 'N/A')} fat, and "
            f"{nutrients.get('carbs', 'N/A')} carbohydrates."
        )
        combined_summary = f"{description} {nutrient_summary}"

        return SummaryResponse(summary=combined_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {e}")

# Endpoint
@app.post("/analyze", response_model=SummaryResponse)
async def analyze(request: ImageRequest):
    image = download_image(request.image_url)
    return analyze_image(image)
