import os
from collections import OrderedDict
from threading import RLock
from typing import Optional

from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor

class BaseClient:
    def __init__(self, *args, **kwargs):
        self.rule_id = kwargs.get("rule_id", "GrammaredLanguage")
        print(f"Initialized BaseClient with rule_id: {self.rule_id}")
        self.correction_extractor = ErrantGrammarCorrectionExtractor(rule_id=self.rule_id)
        self._cache_size = int(
            os.getenv("GRAMMARED_LANGUAGE__GRPC_SERVER_CACHE_SIZE", "100000")
        )
        self._sentence_cache: OrderedDict[str, LanguageToolRemoteResult] = OrderedDict()
        self._cache_lock = RLock()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _preprocess(self, text: str) -> str:
        return text
    
    def _predict(self, text: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _pred_postprocess(self, original: str, pred: str, **kwargs) -> LanguageToolRemoteResult:
        matches = self.correction_extractor.extract_replacements(
            original=original, corrected=pred, fix_tokenization=kwargs.get('fix_tokenization', True)
        )
        return LanguageToolRemoteResult(
            language="English",
            languageCode="en-US",
            matches=matches
        )
    
    def _output_postprocess(self, original: str, pred: LanguageToolRemoteResult, **kwargs) -> LanguageToolRemoteResult:
        return pred

    def predict(self, text: str|list[str]) -> LanguageToolRemoteResult|list[LanguageToolRemoteResult]:
        """Predict one text or a model-oriented batch using the sentence cache."""
        if isinstance(text, list):
            return self._predict_batch(text)
        return self._predict_single(text)

    def _predict_single(self, text: str) -> LanguageToolRemoteResult:
        """Predict one text, consulting and updating the shared sentence cache."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        _text = self._preprocess(text)
        corrected_text = self._predict(_text)
        pred = self._pred_postprocess(text, corrected_text)
        result = self._output_postprocess(text, pred)
        self._cache_put(text, result)
        return result

    def _predict_batch(self, texts: list[str]) -> list[LanguageToolRemoteResult]:
        """Predict only unique cache misses in one model batch, preserving order."""
        results: list[Optional[LanguageToolRemoteResult]] = [None] * len(texts)
        misses: list[str] = []
        miss_indexes: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[index] = cached
            elif text in miss_indexes:
                # The first occurrence owns the model result for all duplicates.
                miss_indexes[text].append(index)
            else:
                misses.append(text)
                miss_indexes[text] = [index]

        if not misses:
            return [result for result in results if result is not None]

        _texts = [self._preprocess(text) for text in misses]
        corrected_texts = self._predict(_texts)
        if len(corrected_texts) != len(misses):
            raise RuntimeError(
                f"Batch prediction returned {len(corrected_texts)} outputs "
                f"for {len(misses)} inputs"
            )

        for text, corrected in zip(misses, corrected_texts):
            pred = self._pred_postprocess(text, corrected)
            result = self._output_postprocess(text, pred)
            self._cache_put(text, result)
            for index in miss_indexes[text]:
                results[index] = result

        return [result for result in results if result is not None]

    def _cache_get(self, text: str) -> Optional[LanguageToolRemoteResult]:
        """Get a sentence result and refresh its LRU position.

        Results are intentionally returned by reference, matching the former
        ``lru_cache`` behavior. Callers must treat them as immutable.
        """
        with self._cache_lock:
            result = self._sentence_cache.get(text)
            if result is None:
                self._cache_misses += 1
                return None
            self._sentence_cache.move_to_end(text)
            self._cache_hits += 1
            return result

    def _cache_put(self, text: str, result: LanguageToolRemoteResult) -> None:
        """Store a result, evicting the least recently used entry if necessary."""
        if self._cache_size <= 0:
            return
        with self._cache_lock:
            self._sentence_cache[text] = result
            self._sentence_cache.move_to_end(text)
            while len(self._sentence_cache) > self._cache_size:
                self._sentence_cache.popitem(last=False)

    def cache_stats(self) -> dict[str, int]:
        """Return lightweight cache metrics useful for diagnostics and benchmarks."""
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._sentence_cache),
                "maxsize": self._cache_size,
            }

    def __call__(self, text: str) -> LanguageToolRemoteResult:
        return self.predict(text)
