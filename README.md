# Setup and activate virtual environment
## with nox
```nox -s create_env```

```source .nox/create_env/bin/activate```

## without nox


# Run
## with nox
```nox -s remove_similar_frames -- -h``` to get help

```nox -s remove_similar_frames -- {path to directory with images}``` to process the directory with images

## without nox
```python -m remove_similar_frames -h``` to get help
```python -m remove_similar_frames {path to directory with images}``` to process the directory with images

# Lint
## with nox
```nox -t style```