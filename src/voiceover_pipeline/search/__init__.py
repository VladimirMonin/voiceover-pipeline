"""Offline lexical and (later) semantic search over canonical history.

Plan section 8 stage S08 ships the lexical FTS5 layer only:
:mod:`voiceover_pipeline.search.lexical` owns the derived search form, the
chunking policy, and the query, while :mod:`voiceover_pipeline.search.indexing`
maintains the derived index from SQLite alone. The semantic and hybrid backends
of :mod:`voiceover_pipeline.search.embeddings` are S09 and are not present here.
"""
