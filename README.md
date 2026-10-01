# summarize-evals

Special-purpose tools to summarize LLM evaluation results

## Quickstart

```
python3 summarize_evals.py example-data/v2zloss_86k.flag-evals-436.tasks.csv 
```

output should look like

```
v2zloss_86k = 0.417
├── reasoning (11.11%) = 0.106
│   └── english = 0.106
│       ├── AIME24 (20.00%) = 0.107
│       │   └── eng_Latn accuracy_avg (n_shot 0) = 0.107
│       ├── AIME25 (20.00%) = 0.1
│       │   └── eng_Latn accuracy_avg (n_shot 0) = 0.1
│       ├── AMC23 (20.00%) = 0.144
│       │   └── eng_Latn accuracy_avg (n_shot 0) = 0.144
│       ├── JEEBench (20.00%) = 0.14
│       │   └── eng_Latn accuracy_avg (n_shot 0) = 0.14
│       └── agieval_lsat_ar (20.00%) = 0.038
│           └── eng_Latn acc_norm (n_shot 3) = 0.038
├── knowledge (11.11%) = 0.429
│   ├── english (50.00%) = 0.499
│   │   ├── GPQADiamond (12.50%) = 0.0595
[...]
```
