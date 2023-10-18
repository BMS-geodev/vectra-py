from typing import List
from local_document import LocalDocument
from custom_types import QueryResult, DocumentChunkMetadata, Tokenizer, DocumentTextSection


class LocalDocumentResult(LocalDocument):
    def __init__(self, folder_path: str, id: str, uri: str, chunks, tokenizer: Tokenizer):  # List[QueryResult[DocumentChunkMetadata]]
        super().__init__(folder_path, id, uri)
        self._chunks = chunks
        self._tokenizer = tokenizer

        # Compute average score
        score = 0
        for chunk in self._chunks:
            score += chunk.score
        self._score = score / len(self._chunks)

    @property
    def chunks(self):  # -> List[QueryResult[DocumentChunkMetadata]]
        return self._chunks

    @property
    def score(self) -> float:
        return self._score

    async def render_sections(self, max_tokens: int, max_sections: int) -> List[DocumentTextSection]:
        # Load text from disk
        text = await self.load_text()

        # First check to see if the entire document is less than max_tokens
        tokens = self._tokenizer.encode(text)
        if len(tokens) < max_tokens:
            return [{
                "text": text,
                "token_count": len(tokens),
                "score": 1.0
            }]

        # Otherwise, we need to split the document into sections
        # - Add each chunk to a temp array and filter out any chunk that's longer than max_tokens.
        # - Sort the array by start_pos to arrange chunks in document order.
        # - Generate a new array of sections by combining chunks until the max_tokens is reached for each section.
        # - Generate an aggregate score for each section by averaging the score of each chunk in the section.
        # - Sort the sections by score and limit to max_sections.
        # - For each remaining section, combine adjacent chunks of text.
        # - Dynamically add overlapping chunks of text to each section until the max_tokens is reached.
        chunks = []
        for chunk in self._chunks:
            start_pos = chunk.item.metadata.start_pos
            end_pos = chunk.item.metadata.end_pos
            chunk_text = text[start_pos:end_pos + 1]
            chunk_tokens = self._tokenizer.encode(chunk_text)
            if len(chunk_tokens) <= max_tokens:
                chunks.append({
                    "text": chunk_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "score": chunk.score,
                    "token_count": len(chunk_tokens)
                })

        chunks.sort(key=lambda x: x["start_pos"])

        if not chunks:
            # Take the top chunk and return a subset of its text
            top_chunk = self._chunks[0]
            start_pos = top_chunk.item.metadata.start_pos
            end_pos = top_chunk.item.metadata.end_pos
            chunk_text = text[start_pos:end_pos + 1]
            tokens = self._tokenizer.encode(chunk_text)
            return [{
                "text": self._tokenizer.decode(tokens[:max_tokens]),
                "token_count": max_tokens,
                "score": top_chunk.score
            }]

        sections = []
        current_section = {
            "chunks": [],
            "score": 0,
            "token_count": 0
        }

        for chunk in chunks:
            if current_section["token_count"] + chunk["token_count"] > max_tokens:
                sections.append(current_section.copy())
                current_section = {
                    "chunks": [],
                    "score": 0,
                    "token_count": 0
                }
            current_section["chunks"].append(chunk)
            current_section["score"] += chunk["score"]
            current_section["token_count"] += chunk["token_count"]

        # Normalize section scores
        for section in sections:
            section["score"] /= len(section["chunks"])

        # Sort sections by score and limit to max_sections
        sections.sort(key=lambda x: x["score"], reverse=True)
        if len(sections) > max_sections:
            sections = sections[:max_sections]

        # Combine adjacent chunks of text
        for section in sections:
            i = 0
            while i < len(section["chunks"]) - 1:
                chunk = section["chunks"][i]
                next_chunk = section["chunks"][i + 1]
                if chunk["end_pos"] + 1 == next_chunk["start_pos"]:
                    chunk["text"] += next_chunk["text"]
                    chunk["end_pos"] = next_chunk["end_pos"]
                    chunk["token_count"] += next_chunk["token_count"]
                    section["chunks"].pop(i + 1)
                else:
                    i += 1

        # Add overlapping chunks of text to each section until the max_tokens is reached
        connector = {
            "text": '\n\n...\n\n',
            "start_pos": -1,
            "end_pos": -1,
            "score": 0,
            "token_count": self._tokenizer.encode('\n\n...\n\n')
        }

        for section in sections:
            # Insert connectors between chunks
            if len(section["chunks"]) > 1:
                i = 0
                while i < len(section["chunks"]) - 1:
                    section["chunks"].insert(i + 1, connector)
                    section["token_count"] += connector["token_count"]
                    i += 2

            # Add chunks to the beginning and end of the section until max_tokens is reached
            budget = max_tokens - section["token_count"]
            if budget > 40:
                section_start = section["chunks"][0]["start_pos"]
                section_end = section["chunks"][-1]["end_pos"]
                if section_start > 0:
                    before_text = text[:section_start]
                    before_tokens = self._tokenizer.encode(before_text)
                    before_budget = min(len(before_tokens), budget // 2)
                    chunk = {
                        "text": self._tokenizer.decode(before_tokens[-before_budget:]),
                        "start_pos": section_start - before_budget,
                        "end_pos": section_start - 1,
                        "score": 0,
                        "token_count": before_budget
                    }
                    section["chunks"].insert(0, chunk)
                    section["token_count"] += chunk["token_count"]
                    budget -= chunk["token_count"]

                if section_end < len(text) - 1:
                    after_text = text[section_end + 1:]
                    after_tokens = self._tokenizer.encode(after_text)
                    after_budget = min(len(after_tokens), budget)
                    chunk = {
                        "text": self._tokenizer.decode(after_tokens[:after_budget]),
                        "start_pos": section_end + 1,
                        "end_pos": section_end + after_budget,
                        "score": 0,
                        "token_count": after_budget
                    }
                    section["chunks"].append(chunk)
                    section["token_count"] += chunk["token_count"]
                    budget -= chunk["token_count"]

        # Return final rendered sections
        rendered_sections = []
        for section in sections:
            text = ''
            for chunk in section["chunks"]:
                text += chunk["text"]
            rendered_sections.append({
                "text": text,
                "token_count": section["token_count"],
                "score": section["score"]
            })
        return rendered_sections
