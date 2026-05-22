import requests
import google.generativeai as genai
import os
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()


class VideoEmbedder:

    def __init__(self):

        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def describe_video(self, video_url):

        try:

            print(f"\nProcessing video: {video_url}")

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(
                video_url,
                headers=headers,
                stream=True,
                timeout=60
            )

            if response.status_code != 200:

                print(
                    f"Video request failed: {response.status_code}"
                )

                return None

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_video:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    temp_video.write(chunk)

                temp_path = temp_video.name

            print("Video downloaded")

            uploaded_file = genai.upload_file(
                temp_path
            )

            print("Waiting for Gemini processing...")

            time.sleep(15)

            prompt = """
Describe the main content of this media
for semantic retrieval.

Keep the description concise and relevant.
"""

            result = self.model.generate_content([
                prompt,
                uploaded_file
            ])

            os.remove(temp_path)

            print("Video description complete")

            return result.text

        except Exception as e:

            print("VIDEO ERROR:", e)

            return None


video_embedder = VideoEmbedder()