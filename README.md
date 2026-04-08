# Project-Retinexis
Deep Learning for Non-Invasive Systemic Disease Detection from Retinal Images

## Team

**Team Name:** Retinexis

**Members:**
- Faisal Almofadhi
- Frederick De Rivaz
- Wesam Alsaqr
- Abdalla Alhajeri
- Ali Alharmoodi
- Omar Abdulaal
- Layan Alsubhi
- Talia Almarzouki
- Arnav Gupta
- Hussain Albeshri

**Repository:** https://github.com/Hussain-AB/Project-Retinexis.git

**Deployed App:** https://huggingface.co/spaces/talia-marzouki/Project-Retinexis

**Models HF Repo:** https://huggingface.co/Faisasa12/Project-Retinexis/tree/main

## Models

Trained models are deployed on Hugging Face and require no local setup. Visit the deployed app above to run inference directly in your browser.

## Running Locally

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then run training directly via:

```bash
python scripts/train.py --config <config_path> --fold <0-4>
```

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
