
## requirements

- python
- anaconda

downloaded .json-files from https://github.com/BestBuyAPIs/open-data-set 


## Run in cmd first time:

```

conda create --name venv_rag python=3.9 -y

conda activate venv_rag

pip install -r requirements.txt

conda install jupyter -y
python -m ipykernel install --user --name venv_rag --display-name "Python 3.9 (venv_rag)"

```


## How to run

### Activate enviorment:
```

conda activate venv_rag

```


### Run main

Run all tasks (default):
```
python main.py
```

Run only embedding:
```
python main.py --task embedding
```

Overwrite the index during embedding:
```
python main.py --task embedding --keep_index False
```

Run only conversation:
```
python main.py --task conversation
```



