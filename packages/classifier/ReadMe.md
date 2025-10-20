### Anirud: text/titles/page_number/captions for images/links (maybe arrows/connectors overlap with person 3)

### Jerry: charts/tables/any kind of graph/shapes (basic)

### Javad: logos/images/icons (maybe smart art like infographics)/ differentiation between badge/icon/brand logo


### Tools that could be useful:
- DOC2PPT dataset (dataset of slide images)
- Label Studio (for manually labeling training data)

### Instructions for using Dataset:
- download all json files from github link below
- run three files for retrieving dataset
```bash
python extract_and_annotate.py
```
```bash
python generate_metadata.py
```
```bash
python organize_full_dataset.py
```

#### Resources:
- https://github.com/nttmdlab-nlp/SlideVQA/tree/main/annotations/bbox
- https://huggingface.co/datasets/NTT-hil-insight/SlideVQA

