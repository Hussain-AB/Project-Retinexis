# Project-Retinexis
Deep Learning for Non-Invasive Systemic Disease Detection from Retinal Images

## Testing

### Environment setup

Create and activate a local virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `python -m pytest` rather than bare `pytest` so the test runner uses the
same interpreter as the installed dependencies.

### Fast default test suite

Run the lightweight default suite:

```bash
python -m pytest -q
```

This default path excludes resource-intensive model tests so normal development
feedback stays fast.

### Slow model tests

Run only the slow model tests:

```bash
python -m pytest -o addopts='' --run-slow-model -m slow_model -q
```

Run the full suite including both the fast tests and the slow model tests:

```bash
python -m pytest -o addopts='' --run-slow-model -q
```

`-o addopts=''` clears the default `pytest.ini` marker filter, and
`--run-slow-model` allows the slow model test files to be collected.

## Coverage

### Fast coverage

Recommended for routine local development:

```bash
python -m pytest --cov=src --cov-config=.coveragerc --cov-report=term-missing:skip-covered -q
```

### Fuller coverage

Includes the slow model tests and can take significantly longer:

```bash
python -m pytest -o addopts='' --run-slow-model --cov=src --cov-config=.coveragerc --cov-report=term-missing:skip-covered -q
```

### Combined two-stage coverage

Recommended when you want a fuller report without making the default fast suite
your only feedback loop:

```bash
coverage erase
python -m pytest --cov=src --cov-config=.coveragerc --cov-report= -q
python -m pytest -o addopts='' --run-slow-model -m slow_model --cov=src --cov-config=.coveragerc --cov-append --cov-report= -q
coverage report -m
coverage html
```

Open the HTML report with:

```bash
open htmlcov/index.html
```
