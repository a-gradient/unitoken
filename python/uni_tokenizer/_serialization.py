import json
from pathlib import Path
from typing import Literal, TypedDict, cast


MODEL_CONFIG_FILENAME = "unitoken.json"
MODEL_CONFIG_VERSION = 1


class ModelConfig(TypedDict):
  version: int
  unit: Literal["byte", "unicode"]
  format: Literal["gpt2", "unitoken"]
  vocab_file: str
  merges_file: str
  special_tokens: list[str]
  pat_str: str | None
  unicode_bigrams: list[str] | None
  unicode_bigram_mixed_boundary: Literal["keep", "split"]


def write_model_config(path: Path, config: ModelConfig) -> None:
  path.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def read_model_config(directory: Path) -> ModelConfig:
  config_path = directory / MODEL_CONFIG_FILENAME
  try:
    value = json.loads(config_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"Cannot read unitoken model config at {config_path}: {error}") from error

  if not isinstance(value, dict):
    raise ValueError(f"Invalid unitoken model config at {config_path}: expected an object")
  if value.get("version") != MODEL_CONFIG_VERSION:
    raise ValueError(
      f"Unsupported unitoken model config version {value.get('version')!r}; "
      f"expected {MODEL_CONFIG_VERSION}"
    )

  required = {
    "unit": str,
    "format": str,
    "vocab_file": str,
    "merges_file": str,
    "special_tokens": list,
    "unicode_bigram_mixed_boundary": str,
  }
  for field, field_type in required.items():
    if not isinstance(value.get(field), field_type):
      raise ValueError(f"Invalid unitoken model config field {field!r}")
  if value["unit"] not in ("byte", "unicode"):
    raise ValueError(f"Invalid unitoken model unit {value['unit']!r}")
  if value["format"] not in ("gpt2", "unitoken"):
    raise ValueError(f"Invalid unitoken model format {value['format']!r}")
  for field in ("vocab_file", "merges_file"):
    model_path = Path(value[field])
    if model_path.is_absolute() or ".." in model_path.parts:
      raise ValueError(f"Invalid unitoken model config field {field!r}: expected a relative path")
  if not all(isinstance(token, str) for token in value["special_tokens"]):
    raise ValueError("Invalid unitoken model config field 'special_tokens'")
  if value.get("pat_str") is not None and not isinstance(value["pat_str"], str):
    raise ValueError("Invalid unitoken model config field 'pat_str'")
  if value.get("unicode_bigrams") is not None and (
    not isinstance(value["unicode_bigrams"], list)
    or not all(isinstance(bigram, str) for bigram in value["unicode_bigrams"])
  ):
    raise ValueError("Invalid unitoken model config field 'unicode_bigrams'")
  if value["unicode_bigram_mixed_boundary"] not in ("keep", "split"):
    raise ValueError("Invalid unitoken model config field 'unicode_bigram_mixed_boundary'")

  return cast(ModelConfig, value)
