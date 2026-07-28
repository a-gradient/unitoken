# Python API

The public Python package is organized around one shortcut and four core
objects:

| API | Purpose |
|---|---|
| [`train_bpe()`][ffbpe.training.train_bpe] | Train directly from a string or iterable |
| [`PreTokenizer`][ffbpe.pretokenizer.PreTokenizer] | Count words and Unicode bigrams |
| [`BpeTrainer`][ffbpe.trainer.BpeTrainer] | Train from an explicit word inventory |
| [`BpeModel`][ffbpe.model.BpeModel] | Validate, encode, and save a trained model |
| [`BpeEncoder`][ffbpe.encoder.BpeEncoder] | Load and use model files |

The generated reference follows the typed Python wrappers. Rust extension
implementation classes are intentionally omitted from the public docs.
