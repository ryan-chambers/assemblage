from openai import OpenAI
from config import AppConfig

# Contains shared utils and mode for wine guide

index_name = 'burgundy-notes'

embedding_model = "text-embedding-3-large"

def embedList(chunk: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=AppConfig.openai_api_key)

    res = client.embeddings.create(
      model=embedding_model,
      input=chunk,
    )

    embeds = [r.embedding for r in res.data]
    return embeds

def embedChunk(chunk: str) -> list[float]:
    return embedList([chunk])[0]

class WineNote:
    def __init__(self, note: str, source: str, score: float | None = None):
        """A small container for a retrieved wine note.

        Args:
            note: The text content of the chunk/note.
            source: The original source of the information
            score: Optional relevance score from the vector store/query.
        """
        self.note = note
        self.source = source
        self.score = score

    def __repr__(self):
        return f"WineNote(note={self.note}, source={self.source}, score={self.score})"

class ProducerNotes:
    def __init__(self, producer: str, producer_notes: list[str], wines: list[WineNote], raw: str ):
        """A container for all notes associated with a wine producer/domaine. Includes the raw string used to create it.

        Args:
            producer: The name of the wine producer/domaine.
            notes: A list of WineNote instances associated with this producer.
        """
        self.producer = producer
        self.producer_notes = producer_notes
        self.wines = wines
        self.raw = raw

    def __repr__(self):
        return f"DomaineNotes(producer={self.producer}, notes={self.producer_notes}, wines={self.wines}.\nRaw = {self.raw})"

    def consolidated_note(self) -> str:
        if self.producer_notes:
            return self.producer + '. ' + '. '.join(note.capitalize() for note in self.producer_notes)
        return self.producer

class VillageNotes:
    def __init__(self, village: str, notes: list[str], raw: str):
        self.village = village
        self.notes = notes
        self.raw = raw

    def consolidated_note(self) -> str:
        if self.notes:
            return self.village + '. ' + '. '.join(note.capitalize() for note in self.notes)
        return self.village

    def __repr__(self):
        return f"VillageNotes(village={self.village}, notes={self.notes}. Raw = {self.raw})"
