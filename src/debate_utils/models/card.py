from enum import Enum
from typing import List, Set, Optional, Dict, Iterable
from dataclasses import dataclass
import json
import jsonlines
from io import FileIO
import datasets
import hashlib

TextStyle = Enum('TextStyle', ["UNDERLINED", "EMPHASIS", "BOLD", "HILITE"])


def styleset_to_html(text: str, styles: Set[TextStyle], tag_type="span"):
    return_value = f"<{tag_type} class=\""
    for style in styles:
        match style:
            case TextStyle.UNDERLINED:
                return_value += "underline "
            case TextStyle.EMPHASIS:
                return_value += "emphasis "
            case TextStyle.BOLD:
                return_value += "bold "
            case TextStyle.HILITE:
                return_value += "hilite "
    text = text.replace("\n", "<br/>")
    return f"{return_value[:-1]}\">{text}</{tag_type}>"

@dataclass
class CardInfo:
    source: str
    source_filename: str
    sourece_file_md5sum_hexdigest: str
    competition_year: Optional[int]

@dataclass
class Card:
    paragraphs : List[str]
    paragraphs_styles : List[Set[TextStyle]]
    tag : str
    cite : List[str]
    cite_styles : List[TextStyle]
    additional_info: CardInfo

    def checksum(self) -> int:
        s = "".join(self.paragraphs) + self.tag + "".join(self.cite) + self.additional_info.sourece_file_md5sum_hexdigest
        return hashlib.md5(s.encode()).hexdigest()

    # computed values.
    def text_plain(self) -> str:
        return "".join(self.paragraphs)
    
    def selected_text_plain(self) -> str:
        selected_paragraphs = [p for (p,ss) in zip(self.paragraphs, self.paragraphs_styles) if len(ss) > 0]
        if len(selected_paragraphs) > 0:
            return "".join(selected_paragraphs)
        else:
            assert all([len(style_set) == 0 for style_set in self.paragraphs_styles])
            return self.text_plain()

    
    def cite_plain(self):
        return "".join(self.cite)

    def _check_invariants(self):
        assert len(self.paragraphs_styles) == len(self.paragraphs), \
            "The list of paragraph styles should be equal to the number of paragraphs."
        assert len(self.cite_styles) == len(self.cite), \
            "The list of citation styles should be equal to the length of the citation."

    def to_html(self):
        return_value = f"<span class='tag'>{self.tag}</span><br/>"
        for cite_text, cite_styleset in zip(self.cite, self.cite_Styles):
            return_value += styleset_to_html(cite_text, cite_styleset, tag_type="span")
        for paragraph_text, paragraph_styleset in zip(self.paragraphs, self.paragraphs_styles):
            return_value += styleset_to_html(paragraph_text, paragraph_styleset, tag_type="span")
        return return_value
    
    def to_plaintext(self):
        return self.tag.upper() + "\n" + "".join(self.cite) + "\n" + "".join(self.paragraphs)
    
    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_parsed_json(json.loads(json_str))

    @classmethod
    def from_parsed_json(cls, json_object: dict):
        paragraph_styles = []
        for hi, under, emph in zip(json_object["highlight_labels"], json_object["underline_labels"], json_object["emphasis_labels"]):
            styleset = set()
            if hi:
                styleset.add(TextStyle.HILITE)
            if under:
                styleset.add(TextStyle.UNDERLINED)
            if emph:
                styleset.add(TextStyle.EMPHASIS)
            paragraph_styles.append(styleset)
        # TODO incorporate cite empahsis.
        return Card(
            paragraphs=json_object["run_text"],
            paragraphs_styles=paragraph_styles,
            tag=json_object["tag"],
            cite=[json_object["cite"]], # TODO
            cite_styles=[set()], # TODO
            additional_info=CardInfo(
                source=json_object["additional_info"]["camp_or_other_source"],
                source_filename=json_object["additional_info"]["filename"], 
                sourece_file_md5sum_hexdigest=json_object["additional_info"]["md5sum"],
                competition_year=2024
            )
        )

    @classmethod
    def from_jsonl_file(cls, jsonl_file: FileIO):
        with jsonlines.Reader(jsonl_file) as reader:
            for line in reader:
                yield Card.from_parsed_json(line)
    
    @classmethod
    def from_hf_dataset(cls, dataset_path="hspolicy/ndca_openev_2024_cards") -> Iterable['Card']:
        data = datasets.load_dataset(dataset_path)
        for item in data["train"]:
            yield(cls.from_parsed_json(item))
    
    def __str__(self) -> str:
        return self.tag + " (" + self.cite_plain() + ")"


def sort_cards_into_files(cards: List[Card]) -> Dict[str, List[Card]]:
    files = {}
    for c in cards:
        file_hash = c.additional_info.sourece_file_md5sum_hexdigest
        if file_hash in files.keys():
            files[file_hash].append(c)
        else:
            files[file_hash] = [ c ]
    return files