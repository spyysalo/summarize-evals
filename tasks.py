import re

from langcodes import normalize_langcode

# Grouping of tasks into top-level categories
GROUP_TASKS = {
    'code': [
        'HumanEval',
        'LiveCodeBench',
        'bigbench_cs_algorithms_generate_until',
        'mbpp',
    ],
    'math': [
        'MATH500',
        'bigbench_dyck_languages_generate_until',
        'bigbench_operators_generate_until',
        'bigbench_repeat_copy_logic_generate_until',
        'gsm8k',
        'global_mgsm',
        'mgsm_native_cot',
    ],
    'reasoning': [
        'AIME24',
        'AIME25',
        'AMC23',
        'JEEBench',
        'agieval_lsat_ar',
        'polymath',
    ],
    'translation': [
        'flores200',
        'opensubtitles_multi40',
    ],
    'knowledge': [
        'GPQADiamond',
        'arc_challenge',
        'arc_challenge_mt',
        'arc_easy',
        'global_mmlu_full',
        'include_base_44',
        'jeopardy',
        'mmlu',
        'openbookqa',
        'bigbench_qa_wikidata_generate_until',
    ],
    'commonsense': [
        'commonsense_qa',
        'copa',
        'global_piqa_completions',
        'global_piqa_prompted',
        'hellaswag',
        'piqa',
        'social_iqa',
        'winogrande',
        'wsc273',
        'xcsqa',
        'xcopa',
    ],
    'reading': [
        'belebele',
        'boolq',
        'coqa',
        'lambada_openai',
        'sib200',
        'squadv2',
    ],
    'language': [
        'bigbench_language_identification_multiple_choice',
        'multiblimp',
    ],
    'instruction': [
        'ifeval',
    ],
}


TASK_GROUP = {    # invert
    task: group
    for group, tasks in GROUP_TASKS.items()
    for task in tasks
}


# Regular expressions for parsing a task label
# (e.g. global_mmlu_full_de_philosophy) into a task
# (e.g. global_mmlu_full), a language (e.g. de), and and optional
# subtask (e.g. philosophy). The first group is assumed to be the
# task, the second the language, and the optional third group the
# subtask.
TASK_LANG_RES = [
    r'^(arc_challenge_mt)_([a-z]{2})$',
    r'^(belebele)_([a-z]{3}_[A-Z][a-z]{3})$',
    r'^(flores200):([a-z]{3}_[A-Z][a-z]{3}-[a-z]{3}_[A-Z][a-z]{3})$',
    r'^(global_mgsm)_([a-z]{2})$',
    r'^(global_mmlu_full)_([a-z]{2})(?:_(.*))?$',
    r'^(global_piqa_completions)_([a-z]{3}_[a-z]{4}(?:_[a-z]{4})?)$',
    r'^(global_piqa_prompted)_([a-z]{3}_[a-z]{4}(?:_[a-z]{4})?)$',
    r'^(hellaswag)_([a-z]{2})$',
    r'^(include_base_44)_([a-z ]+)(?:_(.*))?$',
    r'^(mgsm_native_cot)_([a-z]{2})$',
    r'^(multiblimp)_([a-z]{3})$',
    r'^(opensubtitles_multi40)_([a-z]{2}_to_[a-z]{2})$',
    r'^(polymath)_([a-z]{2})_(low|medium|high|top)$',
    r'^(sib200)_([a-z]{3}_[A-Z][a-z]{3})$',
    r'^(xcopa):([a-z]{2}$)',
    r'^(xcsqa)_([a-z]{3}_[A-Z][a-z]{3})$',
    r'^(mmlu)()(?:_(.*))?$',    # subtask but no lang
]


# Language for monolingual evals where the language is not part of the
# task label
LANG_TASKS = {
    'eng_Latn': [
        'AIME24',
        'AIME25',
        'AMC23',
        'GPQADiamond',
        'JEEBench',
        'MATH500',
        'agieval_lsat_ar',
        'arc_challenge',
        'arc_easy',
        'bigbench_dyck_languages_generate_until',
        'bigbench_language_identification_multiple_choice',
        'bigbench_operators_generate_until',
        'bigbench_qa_wikidata_generate_until',
        'bigbench_repeat_copy_logic_generate_until',
        'boolq',
        'commonsense_qa',
        'copa',
        'coqa',
        'gsm8k',
        'hellaswag',
        'ifeval',
        'jeopardy',
        'lambada_openai',
        'openbookqa',
        'piqa',
        'social_iqa',
        'squadv2',
        'winogrande',
        'wsc273',
    ],
    'code': [
        'HumanEval',
        'LiveCodeBench',
        'bigbench_cs_algorithms_generate_until',
        'mbpp',
    ],
}


TASK_LANG = {    # invert
    task: lang
    for lang, tasks in LANG_TASKS.items()
    for task in tasks
}


PREFERRED_METRIC = {
    'AIME24': 'accuracy_avg',
    'AIME25': 'accuracy_avg',
    'AMC23': 'accuracy_avg',
    'GPQADiamond': 'accuracy_avg',
    'HumanEval': 'python_pass@1',    # NB: not including sh_pass@1
    'JEEBench': 'accuracy_avg',
    'LiveCodeBench': 'accuracy_avg',
    'MATH500': 'accuracy',
    'agieval_lsat_ar': 'acc_norm',
    'arc_challenge': 'acc_norm',
    'arc_challenge_mt': 'acc_norm',
    'arc_easy': 'acc_norm',
    'belebele': 'acc_norm',
    'bigbench_cs_algorithms_generate_until': 'exact_match',
    'bigbench_dyck_languages_generate_until': 'exact_match',
    'bigbench_language_identification_multiple_choice': 'acc',
    'bigbench_operators_generate_until': 'exact_match',
    'bigbench_qa_wikidata_generate_until': 'exact_match',
    'bigbench_repeat_copy_logic_generate_until': 'exact_match',
    'boolq': 'acc',
    'commonsense_qa': 'acc',
    'copa': 'acc',
    'coqa': 'f1',
    'flores200': 'bleu',    # NB: original paper uses chrf++
    'global_mgsm': 'exact_match',
    'global_mmlu_full': 'acc',
    'global_piqa_completions': 'acc_norm',
    'global_piqa_prompted': 'exact_match',
    'gsm8k': 'exact_match',
    'hellaswag': 'acc_norm',
    'ifeval': 'prompt_level_strict_acc',
    'include_base_44': 'acc',
    'jeopardy': 'exact_match',
    'lambada_openai': 'acc',
    'mbpp': 'pass_at_1',
    'mgsm_native_cot': 'exact_match',
    'mmlu': 'acc',
    'multiblimp': 'acc_norm',
    'openbookqa': 'acc_norm',
    'opensubtitles_multi40': 'bleu',
    'piqa': 'acc_norm',
    'polymath': 'exact_match',
    'sib200': 'acc',
    'social_iqa': 'acc',
    'squadv2': 'best_f1',
    'winogrande': 'acc',
    'wsc273': 'acc',
    'xcopa': 'acc',
    'xcsqa': 'acc_norm',
}


# See also "Random baseline" in
# https://github.com/mlfoundations/dclm/blob/main/eval/eval_meta_data.csv
RANDOM_BASELINE = {
    # Generation
    ('AMC23', 'accuracy_avg'): 0,
    ('AIME24', 'accuracy_avg'): 0,
    ('AIME25', 'accuracy_avg'): 0,
    ('HumanEval', 'python_pass@1'): 0,
    ('LiveCodeBench', 'accuracy_avg'): 0,
    ('MATH500', 'accuracy'): 0,
    ('bigbench_cs_algorithms_generate_until', 'exact_match'): 0,
    ('bigbench_dyck_languages_generate_until', 'exact_match'): 0,
    ('bigbench_operators_generate_until', 'exact_match'): 0,
    ('bigbench_qa_wikidata_generate_until', 'exact_match'): 0,
    ('bigbench_repeat_copy_logic_generate_until', 'exact_match'): 0,
    ('coqa', 'f1'): 0,
    ('flores200', 'bleu'): 0,
    ('global_mgsm', 'exact_match'): 0,
    ('global_piqa_prompted', 'exact_match'): 0,
    ('gsm8k', 'exact_match'): 0,
    ('ifeval', 'prompt_level_strict_acc'): 0,
    ('jeopardy', 'exact_match'): 0,
    ('lambada_openai', 'acc'): 0,
    ('mbpp', 'pass_at_1'): 0,
    ('mgsm_native_cot', 'exact_match'): 0,
    ('opensubtitles_multi40', 'bleu'): 0,
    ('polymath', 'exact_match'): 0,
    ('squadv2', 'best_f1'): 0,
    # 2-choice
    ('boolq', 'acc'): 1/2,    # NB: DCLM-core has 0.62
    ('copa', 'acc'): 1/2,
    ('global_piqa_completions', 'acc_norm'): 1/2,
    ('multiblimp', 'acc_norm'): 1/2,
    ('piqa', 'acc_norm'): 1/2,
    ('winogrande', 'acc'): 1/2,
    ('wsc273', 'acc'): 1/2,
    ('xcopa', 'acc'): 1/2,
    # 3-choice
    ('social_iqa', 'acc'): 1/3,
    # 4-choice
    ('GPQADiamond', 'accuracy_avg'): 1/4,
    ('arc_challenge', 'acc_norm'): 1/4,
    ('arc_challenge_mt', 'acc_norm'): 1/4,
    ('arc_easy', 'acc_norm'): 1/4,
    ('belebele', 'acc_norm'): 1/4,
    ('global_mmlu_full', 'acc'): 1/4,
    ('hellaswag', 'acc_norm'): 1/4,
    ('include_base_44', 'acc'): 1/4,
    ('mmlu', 'acc'): 1/4,
    ('openbookqa', 'acc_norm'): 1/4,
    # 5-choice
    ('agieval_lsat_ar', 'acc_norm'): 1/5,    # NB: DCLM-core has 0.25
    ('commonsense_qa', 'acc'): 1/5,
    ('xcsqa', 'acc_norm'): 1/5,
    # 7-choice
    ('sib200', 'acc'): 1/7,
    # 11-choice (NB: DCLM-core has 0.25)
    ('bigbench_language_identification_multiple_choice', 'acc'): 1/11,
    # Mixed
    ('JEEBench', 'accuracy_avg'): 0.1055,    # https://aclanthology.org/2023.emnlp-main.468.pdf#page=5 table 2
}
assert all((k, v) in RANDOM_BASELINE for k, v in PREFERRED_METRIC.items()), \
    f'missing random baseline for {set(PREFERRED_METRIC.items())-set(RANDOM_BASELINE.keys())}'


def get_task_group(task):
    if task not in TASK_GROUP:
        raise ValueError(f'unknown task {task}')
    return TASK_GROUP[task]


def parse_task(label):
    """Given task label (e.g. "global_mmlu_full_de_philosophy"),
    return (task, subtask, lang) triple (e.g. ("global_mmlu_full",
    "philosophy", "deu_Latn"). Subtask is None iff the label has no
    subtask.
    """
    # Tasks where the label doesn't include language or subtask
    if label in TASK_LANG:
        return (label, None, TASK_LANG[label])

    # Other tasks
    matches = []
    for r in TASK_LANG_RES:
        m = re.match(r, label)
        if m:
            matches.append(m.groups())

    if not matches:
        raise ValueError(f'failed to parse label {label}')
    elif len(matches) > 2:
        raise ValueError(f'multiple matches for {label}: {matches}')

    if len(matches[0]) == 2:   # no subtask
        task, lang, subtask = (*matches[0], None)
    else:
        assert len(matches[0]) == 3
        task, lang, subtask = matches[0]

    # edge case: subtask but no language
    if lang == '':
        assert task in ('mmlu',), label
        lang = 'eng_Latn'

    # normalize language code(s)
    lang = normalize_langcode(lang)

    return task, subtask, lang
