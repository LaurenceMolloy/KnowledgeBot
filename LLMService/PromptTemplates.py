class PromptTemplates:

    @staticmethod
    def get_keywords(text: str, top_n: int) -> str:
        return (
            f"Extract up to {top_n} concise, relevant semantic keywords or short keyphrases from the text below.\n"
            f"Order them with the most relevant first.\n"
            f"Return only a JSON list of strings. No formatting. No commentary. No numbering.\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    @staticmethod
    def get_summary(text: str) -> str:
        return (
            "You must summarise the following text in exactly one sentence of no more than 25 words.\n"
            "Do not write multiple sentences.\n"
            "Use passive voice.\n"
            "Exclude all opinions, reasoning, commentary, or extra phrasing.\n"
            "Output only the sentence. Nothing else. No titles. No formatting. No quotes.\n"
            "Ensure the sentence ends with a full stop.\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )