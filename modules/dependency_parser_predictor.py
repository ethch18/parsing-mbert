from typing import Dict, Any, List, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

@Predictor.register('dependency-parser')
class VanillaDependencyParserPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'xx_ent_wiki_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        if self._dataset_reader.use_language_specific_pos: # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)
    
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        words = outputs["words"]
        pos = outputs["pos"]
        heads = outputs["predicted_heads"]
        tags = outputs["predicted_dependencies"]
        return ''.join([('{0}\t{1}\t{1}\t{2}\t{2}\t_\t{3}\t{4}\t_\t_\n'.format(
                i + 1, words[i], pos[i], heads[i], tags[i])) \
                        for i in range(len(words))]) + '\n'

