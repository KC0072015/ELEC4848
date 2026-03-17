"""
src/review_processor.py: Convert attraction review text into a structured dataset.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

from langchain_ollama import ChatOllama

from src.config import OLLAMA_HOST, REVIEW_PROCESSOR_MODEL


@dataclass
class ReviewExtraction:
	category: str | None
	vibes: list[str] = field(default_factory=list)
	environmental_tags: list[str] = field(default_factory=list)
	demographic_tags: list[str] = field(default_factory=list)
	summary_desc: str = "Not specified in reviews."
	facilities_desc: str = "Not specified in reviews."
	tips_desc: str = "Not specified in reviews."


def _load_prompt(prompt_template_file: str) -> str:
	with open(prompt_template_file, "r", encoding="utf-8") as f:
		return f.read()


def _normalize_reviews(reviews: str | Iterable[str]) -> str:
	if isinstance(reviews, str):
		text = reviews.strip()
		return text

	cleaned = [r.strip() for r in reviews if r and r.strip()]
	return "\n\n".join(cleaned)


def _extract_json_block(raw_text: str) -> dict:
	text = raw_text.strip()

	# Handle common fenced output format if the model returns one.
	if text.startswith("```"):
		parts = [p for p in text.split("```") if p.strip()]
		if parts:
			text = parts[0].replace("json", "", 1).strip()

	if text.startswith("{") and text.endswith("}"):
		return json.loads(text)

	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("Model response did not contain a valid JSON object.")

	return json.loads(text[start : end + 1])


def _as_str_list(value: Any) -> list[str]:
	if isinstance(value, list):
		return [str(item).strip() for item in value if str(item).strip()]
	if isinstance(value, str) and value.strip():
		parsed = _parse_json_like(value)
		if isinstance(parsed, list):
			return [str(item).strip() for item in parsed if str(item).strip()]
		return [value.strip()]
	return []


def _parse_json_like(value: str) -> Any:
	text = value.strip()
	if not text:
		return value

	# Try strict JSON first.
	try:
		return json.loads(text)
	except Exception:
		pass

	# Then try Python-like literals such as "['a', 'b']".
	try:
		return ast.literal_eval(text)
	except Exception:
		return value


def _normalize_desc(value: Any, default: str = "Not specified in reviews.") -> str:
	if value is None:
		return default

	if isinstance(value, str):
		parsed = _parse_json_like(value)
		if parsed is not value:
			return _normalize_desc(parsed, default=default)
		cleaned = value.strip()
		return cleaned or default

	if isinstance(value, (list, tuple, set)):
		parts = [str(item).strip() for item in value if str(item).strip()]
		return "; ".join(parts) if parts else default

	if isinstance(value, dict):
		parts = []
		for key, val in value.items():
			k = str(key).strip()
			v = _normalize_desc(val, default="").strip()
			if k and v:
				parts.append(f"{k}: {v}")
		return "; ".join(parts) if parts else default

	cleaned = str(value).strip()
	return cleaned or default


def process_attraction_reviews(
	reviews: str | Iterable[str],
	location_name: str | None = None,
	prompt_template_file: str = "src/review_processor_prompt.txt",
) -> dict[str, Any]:
	"""
	Process attraction reviews into a structured output dictionary.
	"""

	concatenated_reviews_text = _normalize_reviews(reviews)
	if not concatenated_reviews_text:
		raise ValueError("reviews must not be empty")

	prompt_template = _load_prompt(prompt_template_file)
	prompt = prompt_template.format(
		location_name=((location_name or "").strip() or "Unknown location"),
		concatenated_reviews_text=concatenated_reviews_text,
	)

	llm = ChatOllama(
		model=REVIEW_PROCESSOR_MODEL,
		base_url=OLLAMA_HOST,
		temperature=0.0,
	)
	response = llm.invoke(prompt)
	parsed = _extract_json_block(response.content)

	raw_category = parsed.get("category")
	category = str(raw_category).strip() if raw_category is not None else None
	if category == "":
		category = None

	result = ReviewExtraction(
		category=category,
		vibes=_as_str_list(parsed.get("vibes")),
		environmental_tags=_as_str_list(parsed.get("environmental_tags")),
		demographic_tags=_as_str_list(parsed.get("demographic_tags")),
		summary_desc=_normalize_desc(parsed.get("summary_desc")),
		facilities_desc=_normalize_desc(parsed.get("facilities_desc")),
		tips_desc=_normalize_desc(parsed.get("tips_desc")),
	)
	return asdict(result)

