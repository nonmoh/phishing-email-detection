# Unified data dictionary

- `record_id`: stable row identifier created during conversion.
- `text`: model input text. Subject and body are combined where both exist.
- `subject`, `body`: original fields when available.
- `phishing_label`: `1` phishing, `0` legitimate. Folder names are authoritative for the four-class archive because its LLM CSV `label` field represents a different concept and is not used as the phishing target.
- `origin`: `human` or `llm`.
- `generator`: LLM name when the source provides it; otherwise `unknown_llm`. No generator identity is inferred.
- `register`: currently `unspecified`. The uploaded corpora do not provide adversarial writing-register labels, so no register label is fabricated.
- `source`: source corpus/model identifier.
- `category`: phishing theme if supplied by the source.
- `dataset`: provenance group.
- `license`: known licensing note.
- `text_hash`: SHA-256 of normalized lowercase text, used for exact deduplication.

## Important research constraint
The current uploaded data supports phishing classification, human-vs-LLM analysis, and cross-generator evaluation for GPT-4.1 / DeepSeek 3.2 / LLaMA 3.3 70B. It does **not** by itself support the claimed adversarial-register-shift experiment because no register labels are present. Keep the novelty claim as cross-generator robustness until a controlled register-labelled set is added.
