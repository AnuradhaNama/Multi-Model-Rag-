from app.rag_pipeline import pipeline


topics = [

    # CULTURE / PLACES

    "Indian culture",
    "Kerala",
    "Tamil Nadu",
    "Araku Valley",
    "Taj Mahal",
    "Red Fort",
    "Mysore Palace",
    "Hampi",
    "Varanasi",
    "Jaipur",

    # ANIMALS

    "Tiger",
    "Elephant",
    "Lion",
    "Peacock",
    "Bengal tiger",
    "Indian elephant",

    # TECHNOLOGY

    "Artificial intelligence",
    "Machine learning",
    "Computer vision",
    "Robotics",

    # SCIENCE

    "Solar System",
    "Moon",
    "Mars",
    "Black hole",
    "Earth"
]


for topic in topics:

    print("\n================================")
    print(f"Ingesting Topic: {topic}")
    print("================================\n")

    pipeline.ingest_topic(topic)

    print(f"\nCompleted: {topic}\n")