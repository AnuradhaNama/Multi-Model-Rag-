import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()


class ImageEmbedder:

    def __init__(self):

        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def describe_image(self, image_url):

        try:

            print(f"\nProcessing image: {image_url}")

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(
                image_url,
                headers=headers,
                timeout=20
            )

            if response.status_code != 200:
                print("Image request failed")
                return None

            image = Image.open(
                BytesIO(response.content)
            )

            print("Image loaded")

            prompt = """
Describe this image clearly.
Mention animals, objects, scenery,
people, colors and important details.
"""

            result = self.model.generate_content(
                [prompt, image]
            )

            description = result.text

            print("Generated description")

            return description

        except Exception as e:

            print("IMAGE ERROR:", e)

            return None


image_embedder = ImageEmbedder()