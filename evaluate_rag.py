import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    print("Error: Set GEMINI_API_KEY in .env")

    exit(1)

os.environ["GOOGLE_API_KEY"] = api_key

from app.rag_pipeline import pipeline

from datasets import Dataset

from ragas import evaluate

from ragas.run_config import RunConfig

from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerRelevancy
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


def run_evaluation():

    print("Starting Multimodal RAG Evaluation...")

    # =====================================================
    # TEST QUESTIONS
    # =====================================================

    test_questions = [

        "What are tiger habitats",

        "show tiger images",

        "tiger sounds",

        "tiger videos"
    ]

    # =====================================================
    # GROUND TRUTHS
    # =====================================================

    ground_truths = [

        "Tigers mainly inhabit forests, mangroves, grasslands, and tropical habitats across Asia.",

        "Tiger images usually show orange fur with black stripes and strong muscular bodies.",

        "Tiger sounds include roars, growls, chuffs, and vocal calls.",

        "Tiger videos usually contain tiger movement, roaring, and wildlife environments."
    ]

    questions = []

    answers = []

    contexts = []

    # =====================================================
    # RUN PIPELINE
    # =====================================================

    for q in test_questions:

        print(f"\nQuery: {q}")

        result = pipeline.query(q)

        answer = result.get(
            "answer",
            "No answer generated"
        )

        chunk_texts = []

        # TEXT RESULTS

        text_results = result.get(
            "text_results",
            {}
        )

        if (
            text_results
            and 'documents' in text_results
            and text_results['documents']
        ):

            chunk_texts.extend(
                text_results['documents'][0]
            )

        # IMAGE RESULTS

        image_results = result.get(
            "image_results",
            {}
        )

        if (
            image_results
            and 'documents' in image_results
            and image_results['documents']
        ):

            chunk_texts.extend(
                image_results['documents'][0]
            )

        # AUDIO RESULTS

        audio_results = result.get(
            "audio_results",
            {}
        )

        if (
            audio_results
            and 'documents' in audio_results
            and audio_results['documents']
        ):

            chunk_texts.extend(
                audio_results['documents'][0]
            )

        # VIDEO RESULTS

        video_results = result.get(
            "video_results",
            {}
        )

        if (
            video_results
            and 'documents' in video_results
            and video_results['documents']
        ):

            chunk_texts.extend(
                video_results['documents'][0]
            )

        if not chunk_texts:

            chunk_texts = [
                "No relevant context retrieved"
            ]

        questions.append(q)

        answers.append(answer)

        contexts.append(chunk_texts)

        print("Answer:", answer)

        print(
            "Retrieved Context Sample:",
            chunk_texts[:1]
        )

        time.sleep(5)

    # =====================================================
    # CREATE DATASET
    # =====================================================

    eval_dataset = Dataset.from_dict({

        "question": questions,

        "answer": answers,

        "contexts": contexts,

        "ground_truth": ground_truths
    })

    print(
        "\nRunning RAGAS Evaluation..."
    )

    # =====================================================
    # EVALUATOR LLM
    # =====================================================

    evaluator_llm = ChatGoogleGenerativeAI(

        model="gemini-2.5-flash",

        google_api_key=api_key,

        timeout=180,

        max_retries=5
    )

    # =====================================================
    # EVALUATOR EMBEDDINGS
    # =====================================================

    evaluator_embeddings = (
        HuggingFaceEmbeddings(

            model_name="BAAI/bge-small-en-v1.5"
        )
    )

    run_config = RunConfig(

        max_workers=1,

        timeout=180
    )

    try:

        metrics = [

            Faithfulness(),

            ContextPrecision(),

            ContextRecall(),

            AnswerCorrectness(),

            AnswerRelevancy()
        ]

        result = evaluate(

            dataset=eval_dataset,

            metrics=metrics,

            llm=evaluator_llm,

            embeddings=evaluator_embeddings,

            run_config=run_config
        )

        print("\n================================")

        print("FINAL EVALUATION REPORT")

        print("================================")

        print(result)

    except Exception as e:

        print("Evaluation failed:", e)


if __name__ == "__main__":

    run_evaluation()