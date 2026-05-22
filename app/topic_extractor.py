import re


class TopicExtractor:

    def extract_topic(self, query):

        query = query.lower()

        # REMOVE COMMON QUESTION PHRASES

        patterns = [

            r"give me about",
            r"tell me about",
            r"what is",
            r"what are",
            r"show me",
            r"give me",
            r"about",
            r"show",
            r"videos of",
            r"images of",
            r"photos of",
            r"pictures of"
        ]

        for pattern in patterns:

            query = re.sub(
                pattern,
                "",
                query
            )

        # REMOVE EXTRA SPACES

        topic = query.strip()

        # CAPITALIZE

        topic = topic.title()

        print(f"Extracted topic: {topic}")

        return topic


topic_extractor = TopicExtractor()