import streamlit as st
import requests


FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multimodal RAG",
    layout="wide"
)

st.title("Multimodal RAG System")

st.markdown(
    "Semantic Retrieval using Text, Images, Audio, and Video Embeddings"
)

st.divider()

# =====================================================
# QUERY SECTION
# =====================================================

st.header("Semantic Query")

query = st.text_input(
    "Ask your question"
)

if st.button("Search"):

    if query:

        with st.spinner(
            "Searching vector database..."
        ):

            try:

                response = requests.post(
                    f"{FASTAPI_URL}/query",
                    json={
                        "query": query
                    }
                )

                if response.status_code == 200:

                    result = response.json()

                    query_lower = query.lower()

                    # =================================================
                    # IMAGE RESULTS
                    # =================================================

                    if (
                        "image" in query_lower
                        or "photo" in query_lower
                        or "picture" in query_lower
                    ):

                        st.subheader(
                            "Image Retrieval"
                        )

                        image_results = result.get(
                            'image_results',
                            {}
                        )

                        if (
                            image_results
                            and 'documents' in image_results
                            and image_results['documents']
                            and len(image_results['documents'][0]) > 0
                        ):

                            for idx, doc in enumerate(
                                image_results['documents'][0]
                            ):

                                st.markdown(
                                    f"### Image Result {idx + 1}"
                                )

                                st.write(doc)

                                try:

                                    metadata = image_results[
                                        'metadatas'
                                    ][0][idx]

                                    image_url = metadata.get(
                                        'image_url'
                                    )

                                    if image_url:

                                        st.image(
                                            image_url,
                                            width=500
                                        )

                                except Exception as e:

                                    st.error(e)

                        else:

                            st.info(
                                "No image results"
                            )

                    # =================================================
                    # AUDIO RESULTS
                    # =================================================

                    elif (
                        "audio" in query_lower
                        or "sound" in query_lower
                        or "voice" in query_lower
                    ):

                        st.subheader(
                            "Audio Retrieval"
                        )

                        audio_results = result.get(
                            'audio_results',
                            {}
                        )

                        if (
                            audio_results
                            and 'documents' in audio_results
                            and audio_results['documents']
                            and len(audio_results['documents'][0]) > 0
                        ):

                            for idx, doc in enumerate(
                                audio_results['documents'][0]
                            ):

                                st.markdown(
                                    f"### Audio Result {idx + 1}"
                                )

                                st.write(doc)

                                try:

                                    metadata = audio_results[
                                        'metadatas'
                                    ][0][idx]

                                    audio_url = metadata.get(
                                        'audio_url'
                                    )

                                    if audio_url:

                                        st.audio(
                                            audio_url
                                        )

                                except Exception as e:

                                    st.error(e)

                        else:

                            st.info(
                                "No audio results"
                            )

                    # =================================================
                    # VIDEO RESULTS
                    # =================================================

                    elif (
                        "video" in query_lower
                        or "movie" in query_lower
                        or "clip" in query_lower
                    ):

                        st.subheader(
                            "Video Retrieval"
                        )

                        video_results = result.get(
                            'video_results',
                            {}
                        )

                        if (
                            video_results
                            and 'documents' in video_results
                            and video_results['documents']
                            and len(video_results['documents'][0]) > 0
                        ):

                            for idx, doc in enumerate(
                                video_results['documents'][0]
                            ):

                                st.markdown(
                                    f"### Video Result {idx + 1}"
                                )

                                st.write(doc)

                                try:

                                    metadata = video_results[
                                        'metadatas'
                                    ][0][idx]

                                    video_url = metadata.get(
                                        'video_url'
                                    )

                                    if video_url:

                                        st.video(
                                            video_url
                                        )

                                except Exception as e:

                                    st.error(e)

                        else:

                            st.info(
                                "No video results"
                            )

                    # =================================================
                    # TEXT RESULTS
                    # =================================================

                    else:

                        st.subheader(
                            "Generated Answer"
                        )

                        st.write(
                            result['answer']
                        )

                else:

                    st.error("Query failed")

            except Exception as e:

                st.error(e)