# DocXLayout


### Run demo with DLA34 model
1. Model is saved in `model_path/`
2. Run
```
python main.py
```
3. Results are saved in `result.json` and visual results are saved in `demo/`.

- The structure of `result.json` as below.
```
root
├── subfileds
│   ├── full column
│   │   └── layouts
│   │       ├── text
│   │       └── text
│   ├── subcolumn
│   │   └── layouts
│   │       ├── text
│   │       └── text
│   ├── others
│   │   └── layouts
│   │       ├── text
│   │       └── text
├── layouts
│   ├── text
│   └── text
```

Subfileds are hierarchical outputs, and layouts are plain results within reading order.
