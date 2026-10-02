"""Entity and relation labels supplied to graph-generation prompts."""

general_entity = [
    "Person", "Location", "Organization", "Event", "Product", "Work", "Time", "Number", "Attribute",
]
health_entity = [
    "Disease", "Symptom", "Drug", "Treatment", "Procedure", "Anatomy", "BodyPart", "Virus", "Bacteria",
]
science_entity = [
    "Force", "Energy", "Field", "Constant", "Equation"
    "Element", "Compound", "Molecule", "Reaction", "Catalyst",
    "Structure", "Cell", "Organism", "Gene", "Pathway", "Individual", "Environment",
    "State", "Property", "Phenomenon", "Law"
]
general_relation_dict = {
    "Contain" : "X belong to Y, X be part of Y, X include Y or X locate at Y, etc.",
    "Fact": "X is Y, X occupation of Y, or The soup tastes delicious, etc.",
    "Action": "X do Y, X happen Y, X cause Y or X affect Y, etc.",
    "Time": "Indicates time or tense, such as X during Y or X while Y, etc.",
    "Causal": "X cause Y, X lead to Y, X result in Y or X because Y, etc.",
    "Step": "X then Y, X next Y, X after Y or X before Y, etc.",
}
general_relation = ["Contain", "Fact", "Action", "Time", "Causal", "Step"]
