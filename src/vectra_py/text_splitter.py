from typing import List, Optional

from custom_types import Tokenizer
from gpt3_tokenizer import GPT3Tokenizer
from all_MiniLM_L6_v2_tokenizer import OSSTokenizer

ALPHANUMERIC_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


class TextSplitterConfig:
    def __init__(
        self,
        separators: List[str],
        keep_separators: bool,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Tokenizer,
        doc_type: Optional[str] = None
    ):
        self.separators = separators
        self.keep_separators = keep_separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.doc_type = doc_type


class TextChunk:
    def __init__(self,
                 text: str,
                 tokens: List[int],
                 start_pos: int,
                 end_pos: int,
                 start_overlap: List[int],
                 end_overlap: List[int]):
        self.text = text
        self.tokens = tokens
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_overlap = start_overlap
        self.end_overlap = end_overlap


class TextSplitter:
    def __init__(self, config: Optional[TextSplitterConfig] = None):
        if config is None:
            config = TextSplitterConfig(
                separators=[],
                keep_separators=False,
                chunk_size=400,
                chunk_overlap=40,
                tokenizer=None
            )
        self.config = config
        # Create a default tokenizer if none is provided
        if not self.config.get('tokenizer'):
            print('tokenizer not found. defaulting to GPT3.')
            self.config.tokenizer = GPT3Tokenizer()

        # Use default separators if none are provided
        if not self.config.get('separators') or len(self.config.get('separators')) == 0:
            self.config['separators'] = self.get_separators(self.config['doc_type'])

        # Validate the config settings
        if self.config.get('chunk_size') < 1:
            raise ValueError("chunk_size must be >= 1")
        elif self.config.get('chunk_overlap') < 0:
            raise ValueError("chunk_overlap must be >= 0")
        elif self.config.get('chunk_overlap') > self.config.get('chunk_size'):
            raise ValueError("chunk_overlap must be <= chunk_size")

    def split(self, text: str) -> List[TextChunk]:
        # Get basic chunks
        chunks = self.recursive_split(text, self.config.get('separators'), 0)

        def get_overlap_tokens(tokens: Optional[List[int]] = None) -> List[int]:
            if tokens is not None:
                length = min(len(tokens), self.config.get('chunk_overlap'))
                return tokens[:length]
            else:
                return []

        # Add overlap tokens and text to the start and end of each chunk
        if self.config.get('chunk_overlap') > 0:
            for i in range(1, len(chunks)):
                previous_chunk = chunks[i - 1]
                chunk = chunks[i]
                next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
                chunk.start_overlap = get_overlap_tokens(previous_chunk.tokens[::-1])[::-1]
                chunk.end_overlap = get_overlap_tokens(next_chunk.tokens) if next_chunk else []

        return chunks

    def recursive_split(self, text: str, separators: List[str], start_pos: int) -> List[TextChunk]:
        chunks = []
        if len(text) > 0:
            # Split text into parts
            parts = []
            separator = ''
            next_separators = separators[1:] if len(separators) > 1 else []
            if separators:
                # Split by separator
                separator = separators[0]
                parts = text.split(separator)
            else:
                # Cut text in half
                half = len(text) // 2
                parts = [text[:half], text[half:]]

            # Iterate over parts
            for i in range(len(parts)):
                last_chunk = i == len(parts) - 1
                # Get chunk text and end_pos
                chunk = parts[i]
                end_pos = start_pos + (len(chunk) - 1) + (0 if last_chunk else len(separator))
                if self.config.get('keep_separators') and not last_chunk:
                    chunk += separator

                # Ensure chunk contains text
                if not self.contains_alphanumeric(chunk):
                    continue

                # Optimization to avoid encoding really large chunks
                if len(chunk) / 6 > self.config.get('chunk_size'):
                    # Break the text into smaller chunks
                    sub_chunks = self.recursive_split(chunk, next_separators, start_pos)
                    chunks.extend(sub_chunks)
                else:
                    # Encode chunk text
                    tokens = self.config.get('tokenizer').encode(chunk)
                    if len(tokens) > self.config.get('chunk_size'):
                        # Break the text into smaller chunks
                        sub_chunks = self.recursive_split(chunk, next_separators, start_pos)
                        chunks.extend(sub_chunks)
                    else:
                        # Append chunk to output
                        chunks.append(TextChunk(
                            text=chunk,
                            tokens=tokens,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            start_overlap=[],
                            end_overlap=[],
                        ))
                # Update start_pos
                start_pos = end_pos + 1

        return self.combine_chunks(chunks)

    def combine_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        combined_chunks = []
        current_chunk = None
        current_length = 0
        separator = '' if self.config.get('keep_separators') else ' '
        for i in range(len(chunks)):
            chunk = chunks[i]
            if current_chunk:
                length = len(current_chunk.tokens) + len(chunk.tokens)
                if length > self.config.get('chunk_size'):
                    combined_chunks.append(current_chunk)
                    current_chunk = chunk
                    current_length = len(chunk.tokens)
                else:
                    current_chunk.text += separator + chunk.text
                    current_chunk.tokens.extend(chunk.tokens)
                    current_length += len(chunk.tokens)
            else:
                current_chunk = chunk
                current_length = len(chunk.tokens)

        if current_chunk:
            combined_chunks.append(current_chunk)

        return combined_chunks

    def contains_alphanumeric(self, text: str) -> bool:
        return any(char in ALPHANUMERIC_CHARS for char in text)

    def get_separators(self, doc_type: str = "") -> List[str]:
        separators = {
            "cpp": [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "go": [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "java": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "c#": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "csharp": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "cs": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "ts": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "tsx": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "typescript": [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "js": [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "jsx": [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "javascript": [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "php": [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "proto": [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "python": [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "py": [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "rst": [
                # Split along section titles
                "\n===\n",
                "\n---\n",
                "\n***\n",
                # Split along directive markers
                "\n.. ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "ruby": [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "rust": [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "scala": [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "swift": [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "md": [
                # First, try to split along Markdown headings (starting with level 2)
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n##### ",
                "\n###### ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n\n",
                # Horizontal lines
                "\n\n***\n\n",
                "\n\n---\n\n",
                "\n\n___\n\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                # Github tables
                "<table>",
                # "<tr>",
                # "<td>",
                # "<td ",
                "\n\n",
                "\n",
                " "
            ],
            "latex": [
                # First, try to split along Latex sections
                "\n\\chapter{",
                "\n\\section{",
                "\n\\subsection{",
                "\n\\subsubsection{",

                # Now split by environments
                "\n\\begin{enumerate}",
                "\n\\begin{itemize}",
                "\n\\begin{description}",
                "\n\\begin{list}",
                "\n\\begin{quote}",
                "\n\\begin{quotation}",
                "\n\\begin{verse}",
                "\n\\begin{verbatim}",

                # Now split by math environments
                "\n\\begin{align}",
                "$$",
                "$",

                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ],
            "html": [
                # First, try to split along HTML tags
                "<body>",
                "<div>",
                "<p>",
                "<br>",
                "<li>",
                "<h1>",
                "<h2>",
                "<h3>",
                "<h4>",
                "<h5>",
                "<h6>",
                "<span>",
                "<table>",
                "<tr>",
                "<td>",
                "<th>",
                "<ul>",
                "<ol>",
                "<header>",
                "<footer>",
                "<nav>",
                # Head
                "<head>",
                "<style>",
                "<script>",
                "<meta>",
                "<title>",
                # Normal type of lines
                " "
            ],
            "sol": [
                # Split along compiler informations definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " "
            ]
        }

        return separators.get(doc_type, ["\n\n", "\n", " "])
