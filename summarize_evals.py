#!/usr/bin/env python3

import sys
import math
import re
import logging

import yaml

import pandas as pd

from dataclasses import dataclass
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('-d', '--depth', type=int, default=None)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-f', '--flat', action='store_true')
    return ap.parse_args()


class Node:
    def __init__(self, name=None, weight=None, value=None, data=None):
        self.name = name
        self.weight = weight
        self.value = value
        self.data = data
        self.children = defaultdict(lambda: Node())

    def __getitem__(self, key):
        node = self.children[key]
        if node.name is None:
            node.name = key
        return node

    def __setitem__(self, key, value):
        assert isinstance(value, Node)
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def items(self):
        return self.children.items()

    def keys(self):
        return self.children.keys()

    def values(self):
        return self.children.values()

    def propagate_values(self):
        """Recursively compute values from leaves to root as weighted sums."""
        if not self.children:
            if self.value is None:
                raise ValueError(f'leaf value is None for {self.name}')
            return self.value

        child_weights = [c.weight for c in self.children.values()]
        total = sum(child_weights)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Children weights of '{self.name}' sum to {total}, not 1.0 "
                f"(children: {list(self.children.keys())}, weights: {child_weights})"
            )

        self.value = sum(c.weight * c.propagate_values() for c in self.children.values())
        return self.value

    def __repr__(self, prefix='', is_last=True, is_root=True, depth=None):
        if is_root:
            connector = ''
        else:
            connector = '└── ' if is_last else '├── '

        line = f'{prefix}{connector}{self.name}'
        if self.weight is not None and self.weight != 1.0:
            line += f' ({self.weight:.2%})'
        if self.data is not None:
            line += f' {self.data.metric} (n_shot {self.data.n_shot})'
        if self.value is not None:
            line += f' = {self.value:.3}'

        lines = [line]

        if depth is not None:
            if depth <= 0:
                return '\n'.join(lines)
            depth -= 1

        children = list(self.children.values())
        for i, child in enumerate(children):
            if is_root:
                child_prefix = prefix
            else:
                child_prefix = prefix + ('    ' if is_last else '│   ')
            lines.append(child.__repr__(child_prefix, i == len(children) - 1, False, depth))

        return '\n'.join(lines)


def load_task_set(path='configs/sets/comprehensive.yaml'):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    return data


def load_weights(path='configs/weights/code-math.yaml'):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    return data


def load_evals(path='configs/evals'):
    path = Path(path)
    evals = {}
    for f in path.iterdir():
        if not f.is_file() or f.suffix != '.yaml':
            continue
        data = yaml.safe_load(f.read_text())
        assert data['name'] not in evals
        evals[data['name']] = data
    return evals


def get_tree_node(tree, keys):
    node = tree
    for key in keys:
        if key not in node:
            return None
        node = node[key]
    return node


def set_tree_node(tree, keys, node):
    parent = tree
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = node


def add_unique_leaf(tree, keys, node):
    existing = get_tree_node(tree, keys)
    if existing is None:
        set_tree_node(tree, keys, node)
    else:
        raise ValueError(f'''
More than one result for {" ".join(keys)}:
{node.data}
---------- vs ----------
{existing.data}

(you probably want to edit filter_results())
''')


def redistribute_child_weights(node):
    total = sum(c.weight for c in node.children.values())
    if math.isclose(total, 1.0, abs_tol=1e-9):
        return
    else:
        for c in node.children.values():
            c.weight /= total


def assign_lang_subtree_weights(parent, lang, weights):
    subtree = parent[lang]
    subtree.weight = 1/len(parent)
    if subtree.children:    # subtasks
        redistribute_child_weights(subtree)


def assign_task_subtree_weights(parent, task, weights):
    subtree = parent[task]
    subtree.weight = 1/len(parent)
    for lang in subtree:
        assign_lang_subtree_weights(subtree, lang, weights)


def assign_lang_group_subtree_weights(parent, lang_group, weights):
    subtree = parent[lang_group]
    eng_weight = weights['english_weights'][parent.name]
    subtree.weight = (
        eng_weight if lang_group == 'english'
        else 1-eng_weight
    )
    for task in subtree:
        assign_task_subtree_weights(subtree, task, weights)


def assign_task_group_subtree_weights(parent, task_group, weights):
    subtree = parent[task_group]
    subtree.weight = weights['weights'][task_group]
    for lang_group in subtree:
        assign_lang_group_subtree_weights(subtree, lang_group, weights)
    redistribute_child_weights(subtree)


def assign_tree_weights(tree, weights):
    tree.weight = 1.0
    for task_group in tree:
        assign_task_group_subtree_weights(tree, task_group, weights)


def build_tree(checkpoint, results, evals, weights):
    tree = Node(checkpoint)

    # Build dict-of-dicts-like tree organized by task group (e.g. "math"),
    # lang_group ("english" or "multilingual"), task, lang, and optionally
    # subtask.
    for r in results:
        task, lang = r.base_task_name, '-'.join(r.languages)
        if r.languages == ['eng_Latn']:
            lang_group = 'english'
        else:
            lang_group = 'multilingual'
        task_group = evals[task]['category']

        if not 'aggregation' in evals[task]:
            # No subtask structure
            node = Node(lang, value=r.norm_value, data=r)
            keys = [task_group, lang_group, task, lang]
        else:
            subtask = find_task(
                r.full_task_name,
                evals[task]['aggregation']['components'],
                allow_missing=True,
            )
            if subtask is None:
                logging.warning(f'skip {task}')
                continue
            name, weight = subtask['name'], subtask['relative_weight']
            node = Node(name, value=r.norm_value, weight=weight, data=r)
            keys = [task_group, lang_group, task, lang, subtask['name']]
        add_unique_leaf(tree, keys, node)

    assign_tree_weights(tree, weights)
    tree.propagate_values()

    return tree


def log_difference(msg, start, end):
    logging.info(f'{msg}: {start} to {end} ({end/start:.1%})')


def print_flat(node, parents=None, depth=None):
    if parents is None:
        parents = []

    if depth is not None:
        if depth <= 0:
            return
        depth -= 1

    children = list(node.children.values())
    if not children or depth==0:
        print(f'{"\t".join(parents)}\t{node.name}\t{node.value:.3f}')
    else:
        for c in children:
            print_flat(c, parents + [node.name], depth)


@dataclass
class Result:
    checkpoint: str
    full_task_name: str
    base_task_name: str
    metric: str
    filter: str
    n_shot: int
    value: float
    task: dict
    languages: list
    norm_value: float = None

    def __repr__(self):
        # TODO more info
        return f'{self.checkpoint} {self.full_task_name} {"".join(self.languages)} {self.metric} {self.filter} = {self.value:.1%}'


def name_or_regex_matches(element, value):
    if 'name' in element:
        return element['name'] == value
    elif 'regex' in element:
        return re.fullmatch(element['regex'], value) is not None
    else:
        raise ValueError(f'neither name or regex in {element}')


def find_task(name, candidates, allow_missing=False):
    matches = []
    for c in candidates:
        if name_or_regex_matches(c['match'], name):
            matches.append(c)

    if not matches:
        if allow_missing:
            return None
        else:
            raise ValueError(f'no eval matches {name}')
    if len(matches) > 1:
        raise ValueError(f'multiple evals match {name}: {matches}')
    return matches[0]


def resolve_languages(task_name, task):
    matches = []
    for i in task['languages']:
        if task_name in i['tasks']:
            if 'language' in i:
                matches.append([i['language']])
            elif 'source_language' in i and 'target_language' in i:
                matches.append([i['source_language'], i['target_language']])
            else:
                raise ValueError(f'failed to resolve language for {task_name}')

    if not matches:
        raise ValueError(f'no language matches {task_name}')
    if len(matches) > 1:
        raise ValueError(f'multiple languages match {task_name}')
    return matches[0]


def normalize_value(value, task):
    value /= task['score']['scale']    # e.g. [0-100] to [0-1]

    minimum = task['normalize']['min']
    maximum = task['normalize']['max']
    value = (value - minimum) / (maximum - minimum)

    if not 0 <= value <= 1:
        raise ValueError(f'normalized value {value} for {task["name"]} not in [0,1]')

    return value


def parse_results(df, evals):
    results = []
    for _, row in df.iterrows():
        full_task_name = row['task']
        task = find_task(full_task_name, evals.values())
        base_task_name = task['name']
        languages = resolve_languages(full_task_name, task)

        results.append(Result(
            checkpoint=row['checkpoint'],
            full_task_name=full_task_name,
            base_task_name=base_task_name,
            metric=row['metric'],
            filter=row['filter'],
            n_shot=row['n_shot'],
            value=row['value'],
            task=task,
            languages=languages
        ))
    return results


def filter_by_task(results, task_set):
    """Filter list of Result objects to those included in task set."""
    included = {
        t['name'] for t in task_set['evals']
    }
    filtered = [
        r for r in results
        if r.base_task_name in included
    ]
    return filtered


def filter_by_metric(results, task_set):
    """Filter list of Result objects to those with the preferred metric."""
    # TODO: implement task_set exceptions to eval defaults
    filtered = [
        r for r in results
        if r.metric == r.task['metric']
    ]
    return filtered


def filter_by_filter(results, task_set):
    """Filter list of Result objects to those with the preferred filter."""
    filtered = [
        r for r in results
        if r.filter == r.task['metric_filter']
    ]
    return filtered


def filter_by_n_shot(results, task_set):
    """Filter list of Result objects to those with preferred n_shot value."""
    # TODO: implement task_set exceptions to eval defaults
    filtered = [
        r for r in results
        if 'shots' in r.task and r.n_shot == r.task['shots']
    ]
    return filtered


def filter_by_language(results, task_set):
    """Filter list of Result objects to remove excluded languages."""
    exclude = task_set['exclude_languages']
    filtered = [
        r for r in results
        if not any(l in r.languages for l in exclude)
    ]

    # Implement exceptions defined in the task set
    exclude_by_task = {}
    for t in task_set['evals']:
        exclude_by_task[t['name']] = t.get('exclude_languages', [])
    filtered = [
        r for r in filtered
        if not any(l in r.languages for l in exclude_by_task[r.base_task_name])
    ]

    return filtered


def filter_by_select(results, task_set):
    """Filter list of Result objects to only selected (sub)tasks."""
    filtered = [
        r for r in results
        if 'select' not in r.task or
        name_or_regex_matches(r.task['select'], r.full_task_name)
    ]
    return filtered


def filter_results(results, task_set):
    initial_unique_tasks = { r.base_task_name for r in results }

    count = len(results)
    results = filter_by_metric(results, task_set)
    log_difference('metric filter', count, len(results))

    count = len(results)
    results = filter_by_filter(results, task_set)
    log_difference('filter filter', count, len(results))

    count = len(results)
    results = filter_by_n_shot(results, task_set)
    log_difference('n_shot filter', count, len(results))

    count = len(results)
    results = filter_by_language(results, task_set)
    log_difference('language filter', count, len(results))

    count = len(results)
    results = filter_by_task(results, task_set)
    log_difference('task filter', count, len(results))

    count = len(results)
    results = filter_by_select(results, task_set)
    log_difference('select filter', count, len(results))

    unique_tasks = { r.base_task_name for r in results }
    task_diff = initial_unique_tasks - unique_tasks
    if task_diff:
        logging.warning(f'removed all results for tasks {task_diff}')

    return results


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    task_set = load_task_set()
    evals = load_evals()
    weights = load_weights()

    df = pd.read_csv(args.csv, keep_default_na=False)

    results = parse_results(df, evals)
    results = filter_results(results, task_set)

    for r in results:
        r.norm_value = normalize_value(r.value, r.task)

    for checkpoint, group in groupby(results, key=lambda r: r.checkpoint):
        tree = build_tree(checkpoint, group, evals, weights)

        if not args.flat:
            print(tree.__repr__(depth=args.depth))
        else:
            print_flat(tree, depth=args.depth)


if __name__ == '__main__':
    main()
