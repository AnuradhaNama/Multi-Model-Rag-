from app.wikipedia_fetcher import fetcher
from app.chunking import chunker
from app.embedding import text_embedder
from app.image_embedding import image_embedder
from app.audio_embedding import audio_embedder
from app.video_embedding import video_embedder
from app.topic_extractor import topic_extractor
from app.vector_store import vector_store
from app.llm import llm
from app.logger import logger


# =====================================================
# STRICT TOPIC FILTER
# =====================================================

def is_topic_relevant(query, title):

    query = query.lower()

    title = title.lower()

    stop_words = {

        "give",
        "me",
        "about",
        "tell",
        "show",
        "videos",
        "video",
        "images",
        "image",
        "photos",
        "photo",
        "pictures",
        "picture",
        "audio",
        "audios",
        "sound",
        "voice",
        "of",
        "the",
        "a",
        "an"
    }

    query_words = [

        word

        for word in query.split()

        if word not in stop_words
    ]

    # STRICT TITLE MATCH

    for word in query_words:

        if word not in title:

            return False

    return True


class RAGPipeline:

    # =====================================================
    # INGESTION
    # =====================================================

    def ingest_topic(self, topic):

        logger.info(
            f"Starting ingestion for: {topic}"
        )

        data = fetcher.fetch_page(topic)

        if not data:
            return "Wikipedia fetch failed"

        # =====================================================
        # TEXT
        # =====================================================

        logger.info(
            "Processing text chunks..."
        )

        chunks = chunker.chunk_text(
            data['text']
        )

        for idx, chunk in enumerate(chunks):

            embedding = text_embedder.embed_text(
                chunk
            )
            print("\nTEXT EMBEDDING VECTOR:")
            print(embedding[:10])
            metadata = {

                'title': data['title'],

                'section': (
                    data['sections'][idx]['section']
                    if idx < len(data['sections'])
                    else ''
                ),

                'subsection': (
                    data['sections'][idx]['subsection']
                    if idx < len(data['sections'])
                    else ''
                ),

                'source': data['source'],

                'type': 'text'
            }

            vector_store.store_text(

                doc_id=f'{topic}_{idx}',

                text=chunk,

                embedding=embedding,

                metadata=metadata
            )

        logger.info(
            "Text embeddings stored"
        )

        # =====================================================
        # IMAGES
        # =====================================================

        logger.info(
            "Processing images..."
        )

        for idx, image_url in enumerate(
            data['images']
        ):

            try:

                logger.info(
                    f"Embedding image {idx}"
                )

                image_description = (
                    image_embedder.describe_image(
                        image_url
                    )
                )

                if not image_description:
                    continue

                image_embedding = (
                    text_embedder.embed_text(
                        image_description
                    )
                )
                print("\nIMAGE EMBEDDING VECTOR:")
                print(image_embedding[:10])
                metadata = {

                    'title': data['title'],

                    'source': data['source'],

                    'type': 'image',

                    'image_url': image_url
                }

                vector_store.store_image(

                    image_id=f'image_{topic}_{idx}',

                    image_text=image_description,

                    embedding=image_embedding,

                    metadata=metadata
                )

                logger.info(
                    "Image stored successfully"
                )

            except Exception as e:

                logger.error(
                    f"IMAGE STORE ERROR: {e}"
                )

        # =====================================================
        # AUDIO
        # =====================================================

        logger.info(
            "Processing audio..."
        )

        for idx, audio_url in enumerate(
            data['audio']
        ):

            try:

                logger.info(
                    f"Embedding audio {idx}"
                )

                transcript = (
                    audio_embedder.transcribe_audio(
                        audio_url
                    )
                )

                if not transcript:
                    continue

                audio_embedding = (
                    text_embedder.embed_text(
                        transcript
                    )
                )
                print("\nAUDIO EMBEDDING VECTOR:")
                print(audio_embedding[:10])
                metadata = {

                    'title': data['title'],

                    'source': data['source'],

                    'type': 'audio',

                    'audio_url': audio_url
                }

                vector_store.store_audio(

                    audio_id=f'audio_{topic}_{idx}',

                    audio_text=transcript,

                    embedding=audio_embedding,

                    metadata=metadata
                )

                logger.info(
                    "Audio stored successfully"
                )

            except Exception as e:

                logger.error(
                    f"AUDIO STORE ERROR: {e}"
                )

        # =====================================================
        # VIDEOS
        # =====================================================

        logger.info(
            "Processing videos..."
        )

        for idx, video_url in enumerate(
            data['videos']
        ):

            try:

                logger.info(
                    f"Embedding video {idx}"
                )

                video_description = (
                    video_embedder.describe_video(
                        video_url
                    )
                )


                if not video_description:
                    continue

                video_embedding = (
                    text_embedder.embed_text(
                        video_description
                    )
                )
                print("\nVIDEO EMBEDDING VECTOR:")
                print(video_embedding[:10])
                metadata = {

                    'title': data['title'],

                    'source': data['source'],

                    'type': 'video',

                    'video_url': video_url
                }

                vector_store.store_video(

                    video_id=f'video_{topic}_{idx}',

                    video_text=video_description,

                    embedding=video_embedding,

                    metadata=metadata
                )

                logger.info(
                    "Video stored successfully"
                )

            except Exception as e:

                logger.error(
                    f"VIDEO STORE ERROR: {e}"
                )

        return "Multimodal Ingestion Complete"

    # =====================================================
    # COLD START
    # =====================================================

    def auto_ingest_if_needed(self, query):

        query_embedding = text_embedder.embed_text(
            query
        )

        results = vector_store.search_text(
            query_embedding
        )

        if (
            not results
            or 'documents' not in results
            or not results['documents']
            or len(results['documents'][0]) == 0
        ):

            logger.info(
                "No data found in DB"
            )

            logger.info(
                "Running cold-start ingestion..."
            )

            topic = topic_extractor.extract_topic(
                query
            )

            logger.info(
                f"Cold-start topic: {topic}"
            )

            self.ingest_topic(topic)

            logger.info(
                "Cold-start ingestion complete"
            )

            return True

        distances = results.get(
            'distances',
            [[]]
        )[0]

        if distances:

            best_distance = min(
                distances
            )

            logger.info(
                f"Best similarity distance: {best_distance}"
            )

            if best_distance > 1.0:

                logger.info(
                    "Triggering cold-start ingestion..."
                )

                topic = topic_extractor.extract_topic(
                    query
                )

                logger.info(
                    f"Cold-start topic: {topic}"
                )

                self.ingest_topic(topic)

                logger.info(
                    "Cold-start ingestion complete"
                )

                return True

        return False

    # =====================================================
    # QUERY
    # =====================================================

    def query(self, query):

        logger.info(
            f"Running query: {query}"
        )

        self.auto_ingest_if_needed(
            query
        )

        query_embedding = text_embedder.embed_text(
            query
        )

        query_lower = query.lower()

        text_results = {}

        image_results = {}

        audio_results = {}

        video_results = {}

        context = ""

        # =====================================================
        # IMAGE SEARCH
        # =====================================================

        if (
            "image" in query_lower
            or "photo" in query_lower
            or "picture" in query_lower
        ):

            raw_image_results = (
                vector_store.search_images(
                    query_embedding
                )
            )

            image_results = {

                'documents': [[]],

                'metadatas': [[]],

                'distances': [[]]
            }

            if (
                raw_image_results
                and 'documents' in raw_image_results
            ):

                docs = raw_image_results[
                    'documents'
                ][0]

                metas = raw_image_results[
                    'metadatas'
                ][0]

                distances = raw_image_results[
                    'distances'
                ][0]

                for doc, meta, dist in zip(
                    docs,
                    metas,
                    distances
                ):

                    title = meta.get(
                        'title',
                        ''
                    )

                    logger.info(
                        f"Image distance: {dist} | title: {title}"
                    )

                    if (
                        dist < 1.4
                        and is_topic_relevant(
                            query,
                            title
                        )
                    ):

                        image_results[
                            'documents'
                        ][0].append(doc)

                        image_results[
                            'metadatas'
                        ][0].append(meta)

                        image_results[
                            'distances'
                        ][0].append(dist)

            if len(image_results['documents'][0]) == 0:

                logger.info(
                    "No relevant image found"
                )

                image_results = {}

        # =====================================================
        # AUDIO SEARCH
        # =====================================================

        elif (
            "audio" in query_lower
            or "sound" in query_lower
            or "voice" in query_lower
        ):

            raw_audio_results = (
                vector_store.search_audio(
                    query_embedding
                )
            )

            audio_results = {

                'documents': [[]],

                'metadatas': [[]],

                'distances': [[]]
            }

            if (
                raw_audio_results
                and 'documents' in raw_audio_results
            ):

                docs = raw_audio_results[
                    'documents'
                ][0]

                metas = raw_audio_results[
                    'metadatas'
                ][0]

                distances = raw_audio_results[
                    'distances'
                ][0]

                for doc, meta, dist in zip(
                    docs,
                    metas,
                    distances
                ):

                    title = meta.get(
                        'title',
                        ''
                    )

                    logger.info(
                        f"Audio distance: {dist} | title: {title}"
                    )

                    if (
                        dist < 1.15
                        and is_topic_relevant(
                            query,
                            title
                        )
                    ):

                        audio_results[
                            'documents'
                        ][0].append(doc)

                        audio_results[
                            'metadatas'
                        ][0].append(meta)

                        audio_results[
                            'distances'
                        ][0].append(dist)

            if len(audio_results['documents'][0]) == 0:

                logger.info(
                    "No relevant audio found"
                )

                audio_results = {}

        # =====================================================
        # VIDEO SEARCH
        # =====================================================

        elif (
            "video" in query_lower
            or "movie" in query_lower
            or "clip" in query_lower
        ):

            raw_video_results = (
                vector_store.search_video(
                    query_embedding
                )
            )

            video_results = {

                'documents': [[]],

                'metadatas': [[]],

                'distances': [[]]
            }

            if (
                raw_video_results
                and 'documents' in raw_video_results
            ):

                docs = raw_video_results[
                    'documents'
                ][0]

                metas = raw_video_results[
                    'metadatas'
                ][0]

                distances = raw_video_results[
                    'distances'
                ][0]

                for doc, meta, dist in zip(
                    docs,
                    metas,
                    distances
                ):

                    title = meta.get(
                        'title',
                        ''
                    )

                    logger.info(
                        f"Video distance: {dist} | title: {title}"
                    )

                    if (
                        dist < 1.5
                        and is_topic_relevant(
                            query,
                            title
                        )
                    ):

                        video_results[
                            'documents'
                        ][0].append(doc)

                        video_results[
                            'metadatas'
                        ][0].append(meta)

                        video_results[
                            'distances'
                        ][0].append(dist)

            if len(video_results['documents'][0]) == 0:

                logger.info(
                    "No relevant video found"
                )

                video_results = {}

        # =====================================================
        # TEXT SEARCH
        # =====================================================

        else:

            raw_text_results = (
                vector_store.search_text(
                    query_embedding
                )
            )

            text_results = {

                'documents': [[]],

                'metadatas': [[]],

                'distances': [[]]
            }

            if (
                raw_text_results
                and 'documents' in raw_text_results
            ):

                docs = raw_text_results[
                    'documents'
                ][0]

                metas = raw_text_results[
                    'metadatas'
                ][0]

                distances = raw_text_results[
                    'distances'
                ][0]

                filtered_chunks = []

                for doc, meta, dist in zip(
                    docs,
                    metas,
                    distances
                ):

                    title = meta.get(
                        'title',
                        ''
                    )

                    logger.info(
                        f"Text distance: {dist} | title: {title}"
                    )

                    if (
                        dist < 1.0
                        and is_topic_relevant(
                            query,
                            title
                        )
                    ):

                        filtered_chunks.append(
                            doc
                        )

                        text_results[
                            'documents'
                        ][0].append(doc)

                        text_results[
                            'metadatas'
                        ][0].append(meta)

                        text_results[
                            'distances'
                        ][0].append(dist)

                context += "\n".join(
                    filtered_chunks
                )

            if not context:

                context = (
                    "No relevant context found."
                )

        # =====================================================
        # GENERATE ANSWER
        # =====================================================

        answer = llm.generate_answer(
            query,
            context
        )

        return {

            'answer': answer,

            'text_results': text_results,

            'image_results': image_results,

            'audio_results': audio_results,

            'video_results': video_results
        }


pipeline = RAGPipeline()