import warnings
from collections import defaultdict

import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore")


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 500,
    ):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.min_cluster_size = min_cluster_size
        self.orphan_cluster_size = orphan_cluster_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _cluster_and_process(self, texts, min_size):
        if len(texts) <= 1:
            return texts, texts if len(texts) == 1 else []

        embeddings = self.model.encode(texts, show_progress_bar=False)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_size, metric="euclidean"
        ).fit_predict(embeddings)

        clusters = defaultdict(list)
        orphans = []
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(texts[i])
            else:
                orphans.append(texts[i])

        chunks = []
        for cluster_paras in clusters.values():
            current_chunk = []
            current_tokens = 0

            for para in cluster_paras:
                para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))

                if current_tokens + para_tokens > self.max_tokens and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

            if current_chunk:
                joined = "\n\n".join(current_chunk)
                chunks.extend(self._split_long_paragraph(joined))

        return chunks, orphans

    def _split_long_paragraph(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))

        return chunks

    def create_chunks(self, text_content: str):
        paragraphs = [
            p.strip() for p in text_content.split("\n") if len(p.strip().split()) > 10
        ]
        if not paragraphs:
            return []

        split_paragraphs = []
        for p in paragraphs:
            split_paragraphs.extend(self._split_long_paragraph(p))
        paragraphs = split_paragraphs

        final_chunks, orphans = self._cluster_and_process(
            paragraphs, self.min_cluster_size
        )

        if len(orphans) > 1:
            orphan_chunks, single_orphans = self._cluster_and_process(
                orphans, self.orphan_cluster_size
            )
            final_chunks.extend(orphan_chunks)
            final_chunks.extend(single_orphans)
        elif orphans:
            final_chunks.extend(orphans)

        return final_chunks
