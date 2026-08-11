import os
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor
from functools import lru_cache

class BaseClient:
    def __init__(self, *args, **kwargs):
        self.rule_id = kwargs.get("rule_id", "GrammaredLanguage")
        print(f"Initialized BaseClient with rule_id: {self.rule_id}")
        self.correction_extractor = ErrantGrammarCorrectionExtractor(rule_id=self.rule_id)
    
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
        """Predict one text or an uncached model-oriented batch of texts."""
        if isinstance(text, list):
            return self._predict_batch(text)
        return self._predict_single(text)

    @lru_cache(maxsize=int(os.getenv('GRAMMARED_LANGUAGE__GRPC_SERVER_CACHE_SIZE', '100000')))
    def _predict_single(self, text: str) -> LanguageToolRemoteResult:
        """Predict and cache a single hashable input text."""
        _text = self._preprocess(text)
        corrected_text = self._predict(_text)
        pred = self._pred_postprocess(text, corrected_text)
        return self._output_postprocess(text, pred)

    def _predict_batch(self, texts: list[str]) -> list[LanguageToolRemoteResult]:
        """Predict a batch without caching the unhashable list input."""
        _texts = [self._preprocess(text) for text in texts]
        corrected_texts = self._predict(_texts)
        if len(corrected_texts) != len(texts):
            raise RuntimeError(
                f"Batch prediction returned {len(corrected_texts)} outputs "
                f"for {len(texts)} inputs"
            )
        pred = [
            self._pred_postprocess(original, corrected)
            for original, corrected in zip(texts, corrected_texts)
        ]
        return [
            self._output_postprocess(original, result)
            for original, result in zip(texts, pred)
        ]

    def __call__(self, text: str) -> LanguageToolRemoteResult:
        return self.predict(text)
