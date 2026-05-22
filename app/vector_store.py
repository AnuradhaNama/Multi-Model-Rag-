import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path='./chroma_db'
        )

        # TEXT COLLECTION

        self.text_collection = self.client.get_or_create_collection(
            name='text_collection'
        )

        # IMAGE COLLECTION

        self.image_collection = self.client.get_or_create_collection(
            name='image_collection'
        )

        # AUDIO COLLECTION

        self.audio_collection = self.client.get_or_create_collection(
            name='audio_collection'
        )

        # VIDEO COLLECTION

        self.video_collection = self.client.get_or_create_collection(
            name='video_collection'
        )

    # TEXT STORAGE

    def store_text(self, doc_id, text, embedding, metadata):

        self.text_collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # IMAGE STORAGE

    def store_image(self, image_id, image_text, embedding, metadata):

        self.image_collection.upsert(
            ids=[image_id],
            documents=[image_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # AUDIO STORAGE

    def store_audio(self, audio_id, audio_text, embedding, metadata):

        self.audio_collection.upsert(
            ids=[audio_id],
            documents=[audio_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # VIDEO STORAGE

    def store_video(self, video_id, video_text, embedding, metadata):

        self.video_collection.upsert(
            ids=[video_id],
            documents=[video_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # TEXT SEARCH

    def search_text(self, embedding):

        return self.text_collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

    # IMAGE SEARCH

    def search_images(self, embedding):

        return self.image_collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

    # AUDIO SEARCH

    def search_audio(self, embedding):

        return self.audio_collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

    # VIDEO SEARCH

    def search_video(self, embedding):

        return self.video_collection.query(
            query_embeddings=[embedding],
            n_results=5
        )


vector_store = VectorStore()