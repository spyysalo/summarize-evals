#!/usr/bin/env python3

import sys
import math
import logging

import yaml

import pandas as pd

from collections import defaultdict
from argparse import ArgumentParser

from tasks import get_task_group, parse_task, PREFERRED_METRIC, RANDOM_BASELINE


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('-d', '--depth', type=int, default=None)
    ap.add_argument('-w', '--weights', default='weights/uniform.yaml')
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap.parse_args()


# Globals (sorry), loaded from YAML
GROUP_WEIGHT, ENGLISH_WEIGHT, MULTILINGUAL_WEIGHT = None, None, None


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

    def propagate(self):
        """Recursively compute values from leaves to root as weighted sums."""
        if not self.children:
            return self.value

        child_weights = [c.weight for c in self.children.values()]
        total = sum(child_weights)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Children weights of '{self.name}' sum to {total}, not 1.0 "
                f"(children: {list(self.children.keys())}, weights: {child_weights})"
            )

        self.value = sum(c.weight * c.propagate() for c in self.children.values())
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


def build_tree(checkpoint, results):
    tree = Node(checkpoint)

    # Build dict-of-dicts-like tree organized by task group (e.g. "math"),
    # lang_group ("english" or "multilingual"), task, and lang.
    for idx, row in results.iterrows():
        task, subtask, lang = row.task, row.subtask, row.lang
        if not pd.isna(subtask):
            raise NotImplementedError('subtask support')

        task_group = get_task_group(task)
        lang_group = 'english' if lang in ('eng_Latn', 'code') else 'multilingual'

        if lang in tree[task_group][lang_group][task]:
            raise ValueError(f'''
More than one result for {task_group} {task} {lang}:
{row}
---------- vs ----------
{tree[task_group][lang_group][task][lang].data}

(you probably want to edit filter_results())
''')
        node = Node(lang, value=row.norm_value, data=row)
        tree[task_group][lang_group][task][lang] = node

    # Assign weights
    tree.weight = 1.0
    for task_group, stree in list(tree.items()):
        stree.weight = GROUP_WEIGHT[task_group]
        num_lang_group = len(stree)
        for lang_group, sstree in list(stree.items()):
            if num_lang_group == 1:
                weight = 1.0
            else:
                assert num_lang_group == 2
                weight = ENGLISH_WEIGHT if lang_group == 'english' \
                    else MULTILINGUAL_WEIGHT
            sstree.weight = weight
            num_task = len(sstree)
            for task, ssstree in list(sstree.items()):
                ssstree.weight = 1.0/num_task
                num_lang = len(ssstree)
                for lang, sssstree in list(ssstree.items()):
                    sssstree.weight = 1.0/num_lang

    tree.propagate()

    return tree


def group_results(df):
    # Add missing weighted average for polymath
    weights = { 'low': 1/15, 'medium': 2/15, 'high': 4/15, 'top': 8/15 }

    # Columns that identify a group
    group_cols = [
        'checkpoint',
        'lang',
        'n_shot',
        'harness',
        'backend',
        'metric',
        'filter',
    ]

    # Compute the weighted average per group for the 'polymath' task
    avg_rows = (
        df[df['task'] == 'polymath']
        .assign(weighted=lambda d: d['subtask'].map(weights) * d['value'])
        .groupby(group_cols, as_index=False)['weighted']
        .sum()
        .assign(
            value=lambda d: d['weighted'],
            task='polymath',
            subtask=None,
            is_group='yes',
            n_samples=0,    # TODO
        )
        .drop(columns='weighted')
    )

    # Concat without forcing column alignment — missing columns become NaN
    df = pd.concat([df, avg_rows], ignore_index=True)

    return df


def remove_rows(df, columns, values):
    """Remove rows whose values given columns match any item in values."""
    mask = df.set_index(list(columns)).index.isin(values)
    return df[~mask]


def log_difference(msg, start, end):
    logging.info(f'{msg}: {start} to {end} ({end/start:.1%})')


def filter_results(df):
    initial_unique_tasks = set(df['task'].unique())

    # Filter out rows not having the preferred metric for the task
    count = len(df)
    df = df[df['metric'] == df['task'].map(PREFERRED_METRIC)]
    log_difference('metric filter', count, len(df))

    # Filter out rows with non-preferred filter values
    count = len(df)
    df = remove_rows(df, ('task', 'filter'), {
        ('gsm8k', 'strict-match'),
        ('mgsm_native_cot', 'strict-match'),
    })
    log_difference('filter filter', count, len(df))

    # Filter out rows with non-preferred n_shot values
    count = len(df)
    df = remove_rows(df, ('task', 'n_shot'), {
        ('arc_challenge', 10),
        ('hellaswag', 0),
        ('piqa', 0),
    })
    log_difference('n_shot filter', count, len(df))

    # Filter out unsupported languages
    count = len(df)
    df = df[~df['lang'].isin({
        'kat_Geor',
    })]
    log_difference('language filter', count, len(df))

    # Filter out redundant tasks
    count = len(df)
    df = df[~df['task'].isin({
        'global_piqa_prompted',
    })]
    log_difference('redundant task filter', count, len(df))

    # Filter out English results from translated tasks where the
    # original English task is also included
    count = len(df)
    df = remove_rows(df, ('task', 'lang'), {
        ('global_mmlu_full', 'eng_Latn'),
        ('global_mgsm', 'eng_Latn'),
        ('mgsm_native_cot', 'eng_Latn'),
        ('global_piqa_completions', 'eng_Latn'),
        ('xcsqa', 'eng_Latn'),
    })
    log_difference('en in multilingual filter', count, len(df))

    # Filter out any subtask results (got too complicated)
    count = len(df)
    df = df[df['subtask'].isna()]
    log_difference('subtask filter', count, len(df))

    unique_tasks = set(df['task'].unique())
    task_diff = initial_unique_tasks - unique_tasks
    if task_diff:
        logging.warning(f'removed all results for tasks {task_diff}')

    return df


def normalize_scores(df):
    # Normalize translation metric scores from [0, 100] to [0, 1]
    for metric in ['bleu', 'bleu_1', 'bleu_4', 'chrf', 'chrf++']:
        df.loc[df['metric'] == metric, 'value'] /= 100.0

    # Normalize squadv2 f1 scores from [0, 100] to [0, 1]
    for metric in ['f1', 'best_f1']:
        df.loc[(df['task'] == 'squadv2') & (df['metric'] == metric), 'value'] /= 100.0

    # Check all (task, metric) pairs have a baseline
    task_metric = set(zip(df['task'], df['metric']))
    missing = task_metric - RANDOM_BASELINE.keys()
    if missing:
        raise ValueError(f'Missing random baseline for {missing}')

    # Normalize wrt random baseline
    baselines = df.apply(
        lambda r: RANDOM_BASELINE[(r['task'], r['metric'])], axis=1
    )
    df['norm_value'] = (df['value'] - baselines) / (1.0 - baselines)

    # Check that all normalized values are in [0, 1]
    out_of_range = df[(df['norm_value'] < 0) | (df['norm_value'] > 1)]
    if len(out_of_range) > 0:
        print('Values out of [0, 1] range:')
        print(out_of_range[['task', 'metric', 'value', 'norm_value']])
        raise ValueError(f'{len(out_of_range)} normalized values outside [0, 1]')
    return df


def load_weights(args):
    global GROUP_WEIGHT, ENGLISH_WEIGHT, MULTILINGUAL_WEIGHT

    with open(args.weights) as f:
        weights = yaml.safe_load(f)

    GROUP_WEIGHT = weights['group_weight']
    ENGLISH_WEIGHT = weights['language_weight']['english']
    MULTILINGUAL_WEIGHT = weights['language_weight']['multilingual']

    assert abs(sum(GROUP_WEIGHT.values()) - 1.0) < 1e-9
    assert ENGLISH_WEIGHT + MULTILINGUAL_WEIGHT == 1.0


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    load_weights(args)

    df = pd.read_csv(args.csv)

    # Parse task into (main) task, subtask, and language
    df[['task', 'subtask', 'lang']] = (
        df['task'].map(parse_task).apply(pd.Series)
    )

    # Add missing grouped results
    df = group_results(df)

    # Only keep result for preferred metric, filter, etc. for each task
    df = filter_results(df)

    # Normalize scores to [0,1] w/random baseline
    df = normalize_scores(df)

    # Group by checkpoint
    results = { c: g for c, g in df.groupby('checkpoint') }

    # Build and display trees
    for checkpoint, group in results.items():
        tree = build_tree(checkpoint, group)
        print(tree.__repr__(depth=args.depth))


if __name__ == '__main__':
    main()
