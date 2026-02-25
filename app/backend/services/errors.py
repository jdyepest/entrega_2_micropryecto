class UpstreamServiceError(RuntimeError):
    """
    Error para servicios externos (Ollama, Gemini, etc).
    Permite retornar un status code distinto de 500 desde Flask.
    """

    def __init__(self, service: str, detail: str, status_code: int = 503):
        super().__init__(detail)
        self.service = service
        self.detail = detail
        self.status_code = status_code

