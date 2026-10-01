# Weights of top-level groups
GROUP_WEIGHT = {
    'code': 1/9,
    'math': 1/9,
    'reasoning': 1/9,
    'translation': 1/9,
    'knowledge': 1/9,
    'commonsense': 1/9,
    'reading': 1/9,
    'language': 1/9,
    'instruction': 1/9,
}
assert sum(GROUP_WEIGHT.values()) == 1


# Language group weights
ENGLISH_WEIGHT = 0.5
MULTILINGUAL_WEIGHT = 1 - ENGLISH_WEIGHT
