from typing import List, Dict, Any
from .local_document import LocalDocument
from .custom_types import QueryResult, DocumentChunkMetadata, Tokenizer, DocumentTextSection


class LocalDocumentResult(LocalDocument):
    def __init__(self, index: Any, id: str, uri: str, chunks: List[QueryResult[DocumentChunkMetadata]], tokenizer: Tokenizer):
        super().__init__(index, id, uri)
        self._chunks = chunks
        self._tokenizer = tokenizer
        self._score = sum(chunk["score"] for chunk in self._chunks) / len(self._chunks) if self._chunks else 0

    @property
    def chunks(self) -> List[QueryResult[DocumentChunkMetadata]]:
        return self._chunks

    @property
    def score(self) -> float:
        return self._score

    async def render_all_sections(self, max_tokens: int) -> List[DocumentTextSection]:
        text = await self.load_text()
        chunks = self._split_chunks(text, max_tokens)
        sorted_chunks = sorted(chunks, key=lambda x: x["start_pos"])

        sections = self._assemble_sections(sorted_chunks, max_tokens)
        return [
            {"text": ''.join(chunk["text"] for chunk in section["chunks"]),
             "token_count": section["token_count"],
             "score": section["score"]}
            for section in sections
        ]

    async def render_sections(self, max_tokens: int, max_sections: int, overlapping_chunks: bool = True) -> List[DocumentTextSection]:
        text = await self.load_text()
        length = await self.get_length()
        if length <= max_tokens:
            return [{"text": text, "token_count": length, "score": 1.0}]

        chunks = [
            {
                "text": text[chunk['item']['metadata']['start_pos']:chunk['item']['metadata']['end_pos'] + 1],
                "start_pos": chunk['item']['metadata']['start_pos'],
                "end_pos": chunk['item']['metadata']['end_pos'],
                "score": chunk['score'],
                "token_count": len(self._tokenizer.encode(text[chunk['item']['metadata']['start_pos']:chunk['item']['metadata']['end_pos'] + 1]))
            }
            for chunk in self._chunks
            if len(self._tokenizer.encode(text[chunk['item']['metadata']['start_pos']:chunk['item']['metadata']['end_pos'] + 1])) <= max_tokens
        ]
        sorted_chunks = sorted(chunks, key=lambda x: x["start_pos"])

        if not sorted_chunks:
            top_chunk = self._chunks[0]
            chunk_text = text[top_chunk.item.metadata.start_pos:top_chunk.item.metadata.end_pos + 1]
            return [{"text": self._tokenizer.decode(self._tokenizer.encode(chunk_text)[:max_tokens]), "token_count": max_tokens, "score": top_chunk.score}]

        sections = self._assemble_sections(sorted_chunks, max_tokens)
        sections = sorted(sections, key=lambda x: x["score"], reverse=True)[:max_sections]

        self._combine_adjacent_chunks(sections)

        if overlapping_chunks:
            self._add_overlapping_chunks(sections, text, max_tokens)

        return [
            {"text": ''.join(chunk["text"] for chunk in section["chunks"]),
             "token_count": section["token_count"],
             "score": section["score"]}
            for section in sections
        ]

    def _split_chunks(self, text: str, max_tokens: int) -> List[Dict[str, Any]]:
        chunks = []
        for chunk in self._chunks:
            start_pos = chunk.item.metadata.start_pos
            end_pos = chunk.item.metadata.end_pos
            chunk_text = text[start_pos:end_pos + 1]
            tokens = self._tokenizer.encode(chunk_text)
            offset = 0
            while offset < len(tokens):
                chunk_length = min(max_tokens, len(tokens) - offset)
                chunks.append({
                    "text": self._tokenizer.decode(tokens[offset:offset + chunk_length]),
                    "start_pos": start_pos + offset,
                    "end_pos": start_pos + offset + chunk_length - 1,
                    "score": chunk.score,
                    "token_count": chunk_length
                })
                offset += chunk_length
        return chunks

    def _assemble_sections(self, sorted_chunks: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        sections = []
        current_section = {"chunks": [], "score": 0, "token_count": 0}
        for chunk in sorted_chunks:
            if current_section["token_count"] + chunk["token_count"] > max_tokens:
                current_section["score"] /= len(current_section["chunks"])
                sections.append(current_section)
                current_section = {"chunks": [], "score": 0, "token_count": 0}
            current_section["chunks"].append(chunk)
            current_section["score"] += chunk["score"]
            current_section["token_count"] += chunk["token_count"]
        current_section["score"] /= len(current_section["chunks"])
        sections.append(current_section)
        return sections

    def _combine_adjacent_chunks(self, sections: List[Dict[str, Any]]) -> None:
        for section in sections:
            i = 0
            while i < len(section["chunks"]) - 1:
                chunk, next_chunk = section["chunks"][i], section["chunks"][i + 1]
                if chunk["end_pos"] + 1 == next_chunk["start_pos"]:
                    chunk["text"] += next_chunk["text"]
                    chunk["end_pos"] = next_chunk["end_pos"]
                    chunk["token_count"] += next_chunk["token_count"]
                    section["chunks"].pop(i + 1)
                else:
                    i += 1

    def _add_overlapping_chunks(self, sections: List[Dict[str, Any]], text: str, max_tokens: int) -> None:
        connector = {
            "text": '\n\n...\n\n',
            "start_pos": -1,
            "end_pos": -1,
            "score": 0,
            "token_count": len(self._tokenizer.encode('\n\n...\n\n'))
        }

        for section in sections:
            if len(section["chunks"]) > 1:
                i = 0
                while i < len(section["chunks"]) - 1:
                    section["chunks"].insert(i + 1, connector)
                    section["token_count"] += connector["token_count"]
                    i += 2

            budget = max_tokens - section["token_count"]
            if budget > 40:
                self._add_adjacent_text(section, text, budget)

    def _add_adjacent_text(self, section: Dict[str, Any], text: str, budget: int) -> None:
        section_start = section["chunks"][0]["start_pos"]
        section_end = section["chunks"][-1]["end_pos"]

        if section_start > 0:
            before_text = text[:section_start]
            before_tokens = self._tokenizer.encode(before_text)
            before_budget = min(len(before_tokens), budget // 2)
            section["chunks"].insert(0, {
                "text": self._tokenizer.decode(before_tokens[-before_budget:]),
                "start_pos": section_start - before_budget,
                "end_pos": section_start - 1,
                "score": 0,
                "token_count": before_budget
            })
            section["token_count"] += before_budget
            budget -= before_budget

        if section_end < len(text) - 1 and budget > 0:
            after_text = text[section_end + 1:]
            after_tokens = self._tokenizer.encode(after_text)
            after_budget = min(len(after_tokens), budget)
            section["chunks"].append({
                "text": self._tokenizer.decode(after_tokens[:after_budget]),
                "start_pos": section_end + 1,
                "end_pos": section_end + after_budget,
                "score": 0,
                "token_count": after_budget
            })
            section["token_count"] += after_budget
