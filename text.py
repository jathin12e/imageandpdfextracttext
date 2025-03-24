import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Retrieve OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("\u274c Missing OpenRouter API Key. Please check your .env file.")

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Set Tesseract Path (Update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray)
        return text.strip() if text.strip() else "No readable text found in image."
    except Exception as e:
        return f"Error processing image: {str(e)}"

def search_text_in_image(image, query):
    """Searches for relevant text in the image based on a query."""
    if image is None:
        return "Please upload an image."

    try:
        # Convert NumPy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert PIL Image to OpenCV format
        image = np.array(image)
        
        # Extract text
        extracted_text = extract_text_from_image(image)
        
        if extracted_text == "No readable text found in image.":
            return "No text found in the image. Try uploading a clearer image."
        
        # Use GPT-4o via OpenRouter to find relevant information
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "your-site-url",  # Optional
                "X-Title": "your-site-name",  # Optional
            },
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that extracts relevant text from images."},
                {"role": "user", "content": f" '{query}'.\n\nText: {extracted_text}"}
            ]
        )

        # Check if response is valid
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "Error: Received an empty response from AI."
    except Exception as e:
        return f"Error in AI processing: {str(e)}"

# Gradio Chatbot Interface
interface = gr.Interface(
    fn=search_text_in_image,
    inputs=["image", "text"],
    outputs="text",
    title="\U0001F5BC️ AI Chatbot: Image-Based Text Search",
    description="Upload an image and enter a query. The chatbot will extract and return relevant text."
)

# Run chatbot
if __name__ == "__main__":
    demo.launch()

