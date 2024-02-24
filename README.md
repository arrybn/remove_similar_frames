# Setup and activate virtual environment
## with nox
```nox -s create_env```

```source .nox/create_env/bin/activate```

## without nox


# Run
## with nox
```nox -s similar_frames_remover -- -h``` to get help

```nox -s similar_frames_remover -- {path to directory with images}``` to process the directory with images

## without nox
```python -m similar_frames_remover -h``` to get help
```python -m similar_frames_remover {path to directory with images}``` to process the directory with images

# Lint
## with nox
```nox -t style```