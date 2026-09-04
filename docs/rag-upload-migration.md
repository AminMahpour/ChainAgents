# RAG upload storage migration

New thread uploads live under `<persist_directory>/uploads-v2/<sha256>/`,
where `<sha256>` is the SHA-256 digest of the trimmed thread ID. The default
persistence directory is `.rag`, so the new root is `.rag/uploads-v2/`.

The entire older `<persist_directory>/uploads/` tree remains untouched. The new
root is a versioned sibling, because a legacy sanitized thread name can itself
be a SHA-256 digest. Search, ingest, thread cloning, and service restarts never
adopt, index, copy, or migrate files from the legacy tree automatically. Cloning
copies only uploads already stored in the new layout.

To recover an older upload, identify its original thread explicitly and
re-upload the file in that thread. After confirming the new scope is searchable,
archive or remove the legacy directory according to your normal retention
policy. Do not infer a thread ID from a legacy directory name. This also applies
to any hashed directories created under `uploads/` before the versioned root
was introduced; their ownership cannot be distinguished safely from legacy data.
