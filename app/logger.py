import logging
import os


# CREATE LOGS FOLDER

if not os.path.exists("logs"):

    os.makedirs("logs")


# LOGGER CONFIG

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),

    handlers=[

        logging.FileHandler(
            "logs/rag_pipeline.log"
        ),

        logging.StreamHandler()
    ]
)


# LOGGER OBJECT

logger = logging.getLogger(
    "MultimodalRAG"
)