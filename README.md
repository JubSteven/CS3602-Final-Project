# nlp-dev

The course project for SJTU CS3602 targeting Chinese SLU problem. 

## Formulation of the SLU Problem

Given an input sentence of spoken language (denoted by concantenation of static words $[w_1,\dots, w_n]$), we need to formulate an output tuple $(a, s, v)$, where $a$ denotes the **act** of the sentence, $s$ denotes the **slot** of the sentence and $v$ denotes the **value** of the semantic label. The following is an example

> 我想定一张从北京飞到上海的机票

The target output would be `inform(城市=上海)`, where `inform` is the act, `城市` is the slot and `上海` is the label(value). Note that all the acts $a\in A$ and value $v\in V$, where $A,V$ are *preset* datasets. Note that for a given sentence, it is possible to have **more than one** $(a,s,v)$ pairs. As an example, we have the following.

```json
{
    "utt_id": 1,
    "manual_transcript": "取消导航",
    "asr_1best": "取消导航",
    "semantic": [
        [
            "inform",
            "操作",
            "取消"
        ],
        [
            "inform",
            "对象",
            "导航"
        ]
    ]
}
```

## Analysis of the given code

The given code approaches the task using a BiLSTM with PosTagging to deal with the problem. Here I will give a detailed explanation of the baseline, so that we get to know the structure of the repo as well as the task better. In the process, some modifications to the given code will also be documented.

`slu_baseline.py` is the main code. Since we will be modifying the model but use the same dataset, I will try to make full use of the *complicated* data preprocessing part. The class `Example` defined in `example.py` can process the data. 

### Data initialization

The `load_dataset` method returns `exmaples`, which is a list of `Example()` object initialized by `di-ui` and `utt`, the following is an example
```yaml
0-0
{'utt_id': 1, 'manual_transcript': '导航到凯里大十字', 'asr_1best': '导航到凯里大十字', 'semantic': [['inform', '操作', '导航'], ['inform', '终点名称', '凯里大十字']]}
```

For each `Example()` object, it has several attributes. Using the previous example, I have listed some key attributes

```yaml
ex: the utt parameter mentioned previous, a dict containing the data associated with one sentence

did: the id of the data

utt: ex['asr_1best'], gets the sentence part of the data

slot: {'inform-操作': '导航', 'inform-终点名称': '凯里大十字'}

tags: ['B-inform-操作', 'I-inform-操作', 'O', 'B-inform-终点名称', 'I-inform-终点名称', 'I-inform-终点名称', 'I-inform-终点名称', 'I-inform-终点名称']

'O' serves as a separator, 'B' symbols the start and 'I' symbols mid-word. This is used for POS-Tagging later.

slotvalue: ['inform-操作-导航', 'inform-终点名称-凯里大十字']

tag_id: [30, 31, 1, 14, 15, 15, 15, 15]. 

Use the convert_tag_to_idx function to convert slotvalue into corresponding index using existing vocabulary. 

(Only consider the act and slot part, no value as it is not in the dataset)
```

**Note:** here $(a,s,v)$ is considered as a whole, not separately. So essentially we are estimating a function $f$ such that $f := w\rightarrow y=(a,s,v)$


where $y$ is a discrete variable. It remains to be seen whether we should break them apart. For now, I leave it as it is.

### Model

The model is relatively simple, defined in `model/slu_baseline_tagging.py`. The forward function deals with the inputs in batch. The word embeddings (`nn.Embedding`) are loaded directly from `word2vec-768.txt`. The model structure can be read simply, so I will focus on the tricky part where data is processed (continuing in the order of `slu_baseline.py`).

First, all the data is loaded into `train_dataset`, and `cur_dataset` is a batch subset of the full dataset. Recall that `train_dataset` is a list of `Example()` objects. `from_example_list` takes in the the subset of `Example()` objects, and returns `current_batch` as a `Batch` object (defined in `utils/batch.py`) with many more attributes, including `utt`, `did`, `lengths`, `lables`, etc. After that, it is fed into the model described previously.

The output of the model is a `torch.Tensor` of shape `[32, 26, 74]`, where `32` is the `batch_size`. The `decode` function will use `SLUTagging.decode()` method to iterate over the dataset again using the trained model, decode the result (transfer the probability into solid indices) and finally use `Example.evaluator()` to calculate the metrics required.

## Summary

Currently, the structure of the code is acceptable, and it contains many parts that fits the current data. It would be ideal if we only need to modify the model part, so that we don't need to bother reading and decoding the data a lot.

