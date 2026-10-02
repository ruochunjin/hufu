"""Coarse entity ontology used by matching and hypergraph fusion."""

from enum import Enum, auto

from hufu.hypergraph.linguistic import Entity


class ENT(Enum):
    """Project-level entity types, including abstract and domain categories."""

    PERSON = auto()
    COUNTRY = auto()
    LOC = auto()
    ORG = auto()
    FAC = auto()
    GPE = auto()
    NORP = auto()
    PRODUCT = auto()
    WORK_OF_ART = auto()
    LAW = auto()
    LANGUAGE = auto()
    OCCUPATION = auto()
    EVENT = auto()
    TEMPORAL = auto()
    NUMBER = auto()
    CONCEPT = auto()
    ORGANISM = auto()
    FOOD = auto()
    MEDICAL = auto()
    ANATOMY = auto()
    SUBSTANCE = auto()
    ASTRO = auto()
    AWARD = auto()
    VEHICLE = auto()
    THEORY = auto()
    GROUP = auto()
    FEATURE = auto()
    ECONOMIC = auto()
    SOCIOLOGY = auto()
    PHENOMENON = auto()
    ACTION = auto()
    NOT_ENT = auto()
    @staticmethod
    def from_entity(ent: Entity) -> "ENT":
        """Collapse a spaCy-style entity label into the project ontology."""

        mapping = {
            Entity.PERSON: ENT.PERSON,
            Entity.NORP: ENT.NORP,
            Entity.FAC: ENT.FAC,
            Entity.ORG: ENT.ORG,
            Entity.GPE: ENT.GPE,
            Entity.LOC: ENT.LOC,
            Entity.PRODUCT: ENT.PRODUCT,
            Entity.EVENT: ENT.EVENT,
            Entity.WORK_OF_ART: ENT.WORK_OF_ART,
            Entity.LAW: ENT.LAW,
            Entity.LANGUAGE: ENT.LANGUAGE,
            Entity.DATE: ENT.TEMPORAL,
            Entity.TIME: ENT.TEMPORAL,
            Entity.PERCENT: ENT.NUMBER,
            Entity.MONEY: ENT.NUMBER,
            Entity.QUANTITY: ENT.NUMBER,
            Entity.ORDINAL: ENT.NUMBER,
            Entity.CARDINAL: ENT.NUMBER,
            Entity.NOT_ENTITY: ENT.NOT_ENT,
        }
        return mapping.get(ent, ENT.NOT_ENT)
    def level(self) -> int:
        """Return the hand-authored specificity level used by type comparison."""

        hierarchy = {
            ENT.NOT_ENT: 0,
            ENT.CONCEPT: 1,
            ENT.TEMPORAL: 4,
            ENT.NUMBER: 2,
            ENT.ORGANISM: 3,
            ENT.FOOD: 3,
            ENT.MEDICAL: 3,
            ENT.ANATOMY: 3,
            ENT.SUBSTANCE: 3,
            ENT.ASTRO: 3,
            ENT.AWARD: 3,
            ENT.VEHICLE: 3,
            ENT.PERSON: 4,
            ENT.COUNTRY: 5,
            ENT.LOC: 4,
            ENT.ORG: 4,
            ENT.FAC: 4,
            ENT.GPE: 4,
            ENT.NORP: 4,
            ENT.PRODUCT: 2,
            ENT.WORK_OF_ART: 3,
            ENT.LAW: 4,
            ENT.LANGUAGE: 4,
            ENT.OCCUPATION: 4,
            ENT.EVENT: 2,
            ENT.THEORY: 3,
            ENT.GROUP: 4,
            ENT.FEATURE: 3,
            ENT.ECONOMIC: 3,
            ENT.SOCIOLOGY: 3,
            ENT.PHENOMENON: 3,
            ENT.ACTION: 2
        }
        return hierarchy.get(self, 0)
    @staticmethod
    def from_str(label: str) -> "ENT":
        """Parse an ontology label, falling back to ``NOT_ENT``."""

        try:
            return ENT[label]
        except KeyError:
            return ENT.NOT_ENT
