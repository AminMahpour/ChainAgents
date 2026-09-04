# RAG upload storage migration

Thread uploads now live in directories named with the SHA-256 digest of the
trimmed thread ID. Older directories created from sanitized thread names remain
untouched, but the runtime does not adopt them because different thread IDs may
have shared the same legacy name.

To recover an older upload, identify its original thread explicitly and
re-upload the file in that thread. After confirming the new hashed scope is
searchable, archive or remove the legacy directory according to your normal
retention policy. Do not infer a thread ID from a legacy directory name.
