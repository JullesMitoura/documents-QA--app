class Prompts:
    @staticmethod
    def final_response(context: str, question: str) -> str:
        return (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
