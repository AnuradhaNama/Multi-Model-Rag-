import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()


class GeminiLLM:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def generate_answer(self, query, context):

        try:

            prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{query}
"""

            response = self.model.generate_content(
                prompt
            )

            return response.text

        except Exception as e:

            print("LLM ERROR:", e)

            return "LLM generation failed."


llm = GeminiLLM()