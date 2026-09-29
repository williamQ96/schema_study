"""Reuse successful index replay within one verification, bound to actual content.

This is not a persistent admission cache. The verifier and its frozen schemas
remain fixed during a scope; the index, paper, and taxonomy are hashed anew on
every lookup. Failed checks are never cached.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from .common import digest, taxonomy


_successful = ContextVar('successful_index_replays', default=None)
MAX_ENTRIES = 32


@contextmanager
def replay_validation_scope():
    if _successful.get() is not None:
        yield
        return
    token = _successful.set(set())
    try:
        yield
    finally:
        _successful.reset(token)


def cached_index_replay(verifier, index, paper_input, taxonomy_value):
    memo = _successful.get()
    if memo is None:
        return verifier(index, paper_input, taxonomy_value)

    def identity():
        return (verifier, digest(index), digest(paper_input),
                digest(taxonomy() if taxonomy_value is None else taxonomy_value))

    try:
        key = identity()
    except (TypeError, ValueError, RecursionError):
        # Preserve the original validator's behavior for malformed values.
        return verifier(index, paper_input, taxonomy_value)
    if key in memo:
        return []
    errors = verifier(index, paper_input, taxonomy_value)
    if errors == [] and len(memo) < MAX_ENTRIES and key == identity():
        memo.add(key)
    return errors
