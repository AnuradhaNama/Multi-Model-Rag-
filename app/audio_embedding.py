import requests
import google.generativeai as genai
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()


class AudioEmbedder:

    def __init__(self):

        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def transcribe_audio(self, audio_url):

        try:

            print(f"\nProcessing audio: {audio_url}")

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(
                audio_url,
                headers=headers,
                stream=True,
                timeout=30
            )

            if response.status_code != 200:

                print(
                    f"Audio request failed: {response.status_code}"
                )

                return None

            # support ogg/mp3/wav
            suffix = ".mp3"

            if ".ogg" in audio_url:
                suffix = ".ogg"

            elif ".wav" in audio_url:
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_audio:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    temp_audio.write(chunk)

                temp_path = temp_audio.name

            print("Audio downloaded")

            uploaded_file = genai.upload_file(
                temp_path
            )

            prompt = """
Transcribe and describe this audio clearly.
Mention sounds, speech, animal noises,
or important audio events.
"""

            result = self.model.generate_content([
                prompt,
                uploaded_file
            ])

            os.remove(temp_path)

            print("Audio transcription complete")

            return result.text

        except Exception as e:

            print("AUDIO ERROR:", e)

            return None


audio_embedder = AudioEmbedder()