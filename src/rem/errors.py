class RemError(Exception):
    """Base class for errors rem reports to the user."""


class MemoryNotFound(RemError):
    def __init__(self, memory_id: str):
        super().__init__(f"no memory with id {memory_id}")
        self.memory_id = memory_id


class StoreLockedError(RemError):
    """The vector store is already open in another process."""


class EmbedderMismatchError(RemError):
    """The configured embedder differs from the one the stored memories use."""


class EmbedderUnavailableError(RemError):
    """The embedding model could not be loaded."""
