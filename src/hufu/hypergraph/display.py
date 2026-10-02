"""Optional interactive visualization for dependency/coreference output.

Importing this module is deliberately side-effect free.  The heavyweight
spaCy and FastCoref pipelines are loaded only when the demo is explicitly run.
"""


def main() -> int:
    """Run the historical visualization demo when optional models exist."""

    try:
        import fastcoref  # noqa: F401 - registers the spaCy component.
        import spacy
        from spacy import displacy
    except ImportError as error:
        raise RuntimeError(
            "The visualization demo requires spaCy and FastCoref."
        ) from error

    import os

    from hufu.hypergraph.combine import calc_correfs_str, combine

    spacy_model = os.environ.get("HUFU_SPACY_MODEL", "en_core_web_trf")
    coref_model = os.environ.get("HUFU_COREF_MODEL", "biu-nlp/lingmess-coref")
    nlp_coref = spacy.load(spacy_model)
    nlp_coref.add_pipe(
        "fastcoref",
        config={
            "model_architecture": "LingMessCoref",
            "model_path": coref_model,
            "device": "cpu",
        },
    )
    nlp_parse = spacy.load(spacy_model)
    text = (
        "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. "
        "He also participated in roundtable discussions. That day, the venue "
        "hosted an AI ethics seminar, which featured his keynote and discussions."
    )
    coref_doc = nlp_coref(
        text, component_cfg={"fastcoref": {"resolve_text": True}}
    )
    parsed_doc = nlp_parse(coref_doc._.resolved_text)
    correfs = calc_correfs_str(coref_doc)
    spans_to_merge = combine(parsed_doc, correfs)
    with parsed_doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            retokenizer.merge(span)
    displacy.serve(parsed_doc, style="dep")
    return 0


if __name__ == "__main__":  # pragma: no cover - optional interactive demo.
    raise SystemExit(main())
