"""Extract typed NetworkX graphs from model-generated record streams."""

import json
import logging
import re

import networkx as nx
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import (
    AIMessage,
    merge_message_runs,
)
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from hufu.graphgen.ontology import general_entity, general_relation
from hufu.llm.prompt_graph import (
    graph_attributes_records_msg,
    graph_entity_records_msg,
    graph_records,
    graph_relation_records_msg,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)
class Graph(BaseModel):
        """Structured schema used when a model emits a complete graph object."""

        class Node(BaseModel):
            """Model-emitted entity record."""

            name: str = Field(..., description="The name of the entity")
            type: str = Field(..., description="The type of the entity")
            desc: str = Field(..., description="A description of the entity")
        class Edge(BaseModel):
            """Model-emitted directed relation record."""

            src: str = Field(..., description="The source entity")
            dst: str = Field(..., description="The destination entity")
            type: str = Field(..., description="The type of the relation")
            desc: str = Field(..., description="A description of the relation")
        graph_name: str = Field(..., description="The name of the graph")
        graph_description: str = Field(..., description="The description of the graph")
        entities: list[Node] = Field(..., description="The list of entities")
        relations: list[Edge] = Field(..., description="The list of relations")
def fresh_entity_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str]]]:
    """Parse entity records and return normalized prompt text plus tuples."""

    if not isinstance(msg.content, str):
        raise ValueError("Expected str")
    content = msg.content
    entities = []
    entities_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        match = re.match(r'"entity",*(.+),*(.+),*(.+)', record)
        if match:
            name, type, desc = match.groups()
            name, type, desc = name.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            entities.append(f'("entity", "{name.strip().rstrip()}", "{type.strip().rstrip()}", "{desc.strip().rstrip()}");')
    outs = "\n".join(entities)
    if not len(entities):
        raise ValueError("Expected entities")
    return outs, entities_list
def fresh_relation_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str, str]]]:
    """Parse relation records and return normalized prompt text plus tuples."""

    if not isinstance(msg.content, str):
        raise ValueError("Expected str")
    content = msg.content
    print(f"Relation: {content}")
    relations = []
    relations_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        match = re.match(r'"relationship",(.+),(.+),(.+),(.+)', record)
        if match:
            src, dst, type, desc = match.groups()
            src, dst, type, desc = src.strip().rstrip().strip('"').rstrip('"'), dst.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            relations.append(f'("relationship", "{src}", "{dst}", "{type}", "{desc}");')
            relations_list.append((src, dst, type, desc))
    outs = "\n".join(relations)
    if not len(relations_list):
        raise ValueError("Expected relations")
    return outs, relations_list
def fresh_attribute_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str, str]]]:
    """Parse attribute records and return normalized prompt text plus tuples."""

    if not isinstance(msg.content, str):
        raise ValueError("Expected str")
    content = msg.content
    print(f"Attribute: {content}")
    attributes = []
    attributes_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        match = re.match(r'"attribute",(.+),(.+),(.+),(.+)', record)
        if match:
            key, value, desc, entity = match.groups()
            key, value, desc, entity = key.strip().rstrip().strip('"').rstrip('"'), value.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"'), entity.strip().rstrip().strip('"').rstrip('"')
            attributes.append(f'("attribute", "{key}", "{value}", "{desc}", "{entity}");')
            attributes_list.append((key, value, desc, entity))
    outs = "\n".join(attributes)
    if not len(attributes_list):
        raise ValueError("Expected attributes")
    return outs, attributes_list
@retry(stop=stop_after_attempt(5))
async def build_graph_step_by_step(model: BaseChatModel, title, text, prop, task='popqa') -> nx.DiGraph:
    """Build a graph through separate entity, relation, and attribute prompts."""

    msgs = [
        graph_entity_records_msg.format(entity_types=", ".join(general_entity), input_text=text),
    ]
    merger = merge_message_runs()
    chain = merger | model
    msg = await chain.ainvoke(msgs)
    entities_msg, entities_list = fresh_entity_records(msg)
    msgs = [
        graph_relation_records_msg.format(relation_types=", ".join(general_relation), input_text=text, entities=entities_msg),
    ]
    msg = await chain.ainvoke(msgs)
    relations_msg, relations_list = fresh_relation_records(msg)
    msgs = [
        graph_attributes_records_msg.format(input_text=text, relations=relations_msg),
    ]
    msg = await chain.ainvoke(msgs)
    attributes_msg, attributes_list = fresh_attribute_records(msg)
    graph = nx.DiGraph()
    graph.graph['title'] = title
    graph.graph['text'] = text
    graph.graph['prop'] = prop
    graph.graph['task'] = task
    for node in entities_list:
        name, type, desc = node
        graph.add_node(name, type=type, desc=desc)
    for edge in relations_list:
        src, dst, type, desc = edge
        graph.add_edge(src, dst, type=type, desc=desc)
    for node in attributes_list:
        key, value, desc, entity = node
        graph.add_node(f"{key}={value}", type='Fact', desc=desc)
        if not graph.has_node(entity):
            graph.add_node(entity, type=key, desc=desc)
        graph.add_edge(entity, key, type='attribute', desc=desc)
    return graph
def fresh_records(content: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    """Parse the three supported record kinds from one model response."""

    entities_list: list[tuple[str, str, str]] = []
    relations_list: list[tuple[str, str, str, str]]= []
    attributes_list: list[tuple[str, str, str, str]] = []
    def _out_bars(record: str):
        match1 = re.match(r'\((.*)\);', record)
        match2 = re.match(r'\((.*)\)', record)
        if match1:
            res: str = match1.group(1)
            return res
        elif match2:
            res: str = match2.group(1)
            return res
        else:
            return record
    # The prompt contract is line-oriented and stops at the explicit sentinel;
    # malformed lines are ignored while each recognized record keeps its order.
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        record = _out_bars(record)
        match = re.match(r'"entity",*(.+),*(.+),*(.+)', record)
        if match:
            name, type, desc = match.groups()
            name, type, desc = name.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            entities_list.append((name, type, desc))
        match = re.match(r'"relationship",(.+),(.+),(.+),(.+)', record)
        if match:
            src, dst, type, desc = match.groups()
            src, dst, type, desc = src.strip().rstrip().strip('"').rstrip('"'), dst.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            relations_list.append((src, dst, type, desc))
        match = re.match(r'"attribute",(.+),(.+),(.+),(.+)', record)
        if match:
            key, value, desc, entity = match.groups()
            key, value, desc, entity = key.strip().rstrip().strip('"').rstrip('"'), value.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"'), entity.strip().rstrip().strip('"').rstrip('"')
            attributes_list.append((key, value, desc, entity))
    if not len(entities_list):
        logger.warning(f"[No entities]: {content}")
    if not len(relations_list):
        logger.warning(f"[No relations]: {content}")
    if not len(attributes_list):
        logger.warning(f"[No attributes]: {content}")
    return entities_list, relations_list, attributes_list
async def build_graph(model: BaseLLM, title, text, prop, task='popqa') -> nx.DiGraph:
    """Generate all records in one call and materialize a directed graph."""

    prompt = graph_records.partial(
        entity_types=", ".join(general_entity),
        relation_types=", ".join(general_relation)
    )
    chain = prompt | model
    ans = await chain.ainvoke({
        "input_text": text,
    })
    entities_list, relations_list, attributes_list = fresh_records(ans)
    graph = nx.DiGraph()
    graph.graph['title'] = title
    graph.graph['text'] = text
    graph.graph['prop'] = prop
    graph.graph['task'] = task
    for node in entities_list:
        name, type, desc = node
        graph.add_node(name, type=type, desc=desc)
    for edge in relations_list:
        src, dst, type, desc = edge
        graph.add_edge(src, dst, type=type, desc=desc)
    for node in attributes_list:
        key, value, desc, entity = node
        graph.add_node(f"{key}={value}", type='Fact', desc=desc)
        if not graph.has_node(entity):
            graph.add_node(entity, type=key, desc=desc)
        graph.add_edge(entity, key, type='attribute', desc=desc)
    return graph
def build_graph_batch(model: BaseLLM, prompt_list: list[dict], task='popqa') -> list[nx.DiGraph]:
    """Generate and materialize a batch of task-specific directed graphs."""

    global graph_building
    match task:
        case 'popqa':
            prompt = graph_records.partial(
                entity_types=", ".join(general_entity),
                relation_types=", ".join(general_relation)
            )
        case _:
            raise ValueError(f"Unknown task: {task}")
    chain = prompt | model
    res_list = chain.batch(prompt_list)
    graphs = []
    for res in res_list:
        entities_list, relations_list, attributes_list = fresh_records(res)
        graph = nx.DiGraph()
        graph.graph['task'] = task
        for node in entities_list:
            name, type, desc = node
            graph.add_node(name, type=type, desc=desc)
        for edge in relations_list:
            src, dst, type, desc = edge
            graph.add_edge(src, dst, type=type, desc=desc)
        for node in attributes_list:
            key, value, desc, entity = node
            graph.add_node(f"{key}={value}", type='Fact', desc=desc)
            if not graph.has_node(entity):
                graph.add_node(entity, type=key, desc=desc)
            graph.add_edge(entity, key, type='attribute', desc=desc)
        graphs.append(graph)
    return graphs
def save_graph(graph: nx.DiGraph, path: str) -> None:
    """Serialize a directed graph in NetworkX node-link JSON form."""

    with open(path, 'w') as f:
        data = nx.node_link_data(graph)
        json.dump(data, f, indent=4)
def load_graph(path: str) -> nx.DiGraph:
    """Load a directed graph from NetworkX node-link JSON form."""

    with open(path, 'r') as f:
        data = json.load(f)
        graph = nx.node_link_graph(data)
    return graph
