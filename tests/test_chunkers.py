# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import tempfile
import os
from collections import defaultdict

# Third Party
import pytest

# First Party
from instructlab.sdg.utils.chunkers import (
    ContextAwareChunker,
    DocumentChunker,
    FileTypes,
    TextSplitChunker,
)

# Local
from .testdata import testdata


@pytest.fixture
def documents_dir():
    return Path(__file__).parent / "testdata" / "sample_documents"


@pytest.mark.parametrize(
    "filepaths, chunker_type",
    [
        ([Path("document.md")], TextSplitChunker),
        ([Path("document.pdf")], ContextAwareChunker),
    ],
)
def test_chunker_factory(filepaths, chunker_type, documents_dir):
    """Test that the DocumentChunker factory class returns the proper Chunker type"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "",
            "filepaths": filepaths,
        }
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        chunker = DocumentChunker(
            leaf_node=leaf_node,
            taxonomy_path=documents_dir,
            output_dir=temp_dir,
            tokenizer_model_name="instructlab/merlinite-7b-lab",
        )
        assert isinstance(chunker, chunker_type)


def test_chunker_factory_unsupported_filetype(documents_dir):
    """Test that the DocumentChunker factory class fails when provided an unsupported document"""
    leaf_node = [
        {
            "documents": ["Lorem ipsum"],
            "taxonomy_path": "",
            "filepaths": [Path("document.jpg")],
        }
    ]
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                leaf_node=leaf_node,
                taxonomy_path=documents_dir,
                output_dir=temp_dir,
                tokenizer_model_name="instructlab/merlinite-7b-lab",
            )


def test_chunker_factory_empty_filetype(documents_dir):
    """Test that the DocumentChunker factory class fails when provided no document"""
    leaf_node = [
        {
            "documents": [],
            "taxonomy_path": "",
            "filepaths": [],
        }
    ]
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            _ = DocumentChunker(
                leaf_node=leaf_node,
                taxonomy_path=documents_dir,
                output_dir=temp_dir,
                tokenizer_model_name="instructlab/merlinite-7b-lab",
            )

def test_text_split_chunker():
    with tempfile.TemporaryDirectory() as temp_dir:
        # test return of chunked markdown
        server_ctx_size = 4096
        doc_dict = defaultdict(list)
        doc_dict[FileTypes.MD].append(("Lorem ipsum", Path("document.md")))
        document_contents = [d for d, _ in doc_dict[FileTypes.MD]]
        chunker = TextSplitChunker(document_contents,server_ctx_size,1024,temp_dir)
        chunk_markdowns = chunker.chunk_documents()
        assert chunk_markdowns == ["Lorem ipsum"]

        # test empty document contents
        chunker = TextSplitChunker([],server_ctx_size,1024,temp_dir)
        chunk_markdowns = chunker.chunk_documents()
        assert chunk_markdowns == []

        # test ValueError being raised
        with pytest.raises(ValueError):
            server_ctx_size = 1024
            doc_dict = defaultdict(list)
            doc_dict[FileTypes.MD].append(("Lorem ipsum", Path("document.md")))
            document_contents = [d for d, _ in doc_dict[FileTypes.MD]]
            chunker = TextSplitChunker(document_contents,server_ctx_size,1024,temp_dir)
            chunk_markdowns = chunker.chunk_documents()

def test_context_aware_chunker():
    with tempfile.TemporaryDirectory() as output_dir:
        # test empty document contents
        document_paths = []
        filepaths = [Path("document.md")]
        chunker = ContextAwareChunker(document_paths,
                                      filepaths,
                                      output_dir,
                                      tokenizer_model_name="instructlab/merlinite-7b-lab",
                                      chunk_word_count=1024
                                    )
        chunk_markdowns = chunker.chunk_documents()
        assert chunk_markdowns == []

        # test return of chunked markdown
        docs_dir = os.path.dirname(__file__) + "/testdata/sample_documents"
        #file = os.path.join(docs_dir, "phoenix.pdf")
        doc_dirs = [docs_dir]
        doc_dict = defaultdict(list)
        doc_dict[FileTypes.PDF].append(("Lorem ipsum", doc_dirs))
        doc_paths = [d for d, _ in doc_dict[FileTypes.PDF]]
        chunker = ContextAwareChunker(doc_paths,
                                      doc_dirs,
                                      output_dir,
                                      tokenizer_model_name="instructlab/merlinite-7b-lab",
                                      chunk_word_count=1024
                                    )
        #chunk_markdowns = chunker.chunk_documents()
        #print(chunk_markdown)
        #assert chunk_markdowns == ["Lorem ipsum"]
