# FastTrackPy
[![PyPI](https://img.shields.io/pypi/v/fasttrackpy)](https://pypi.org/project/fasttrackpy/) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fasttrackpy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fasttrackpy) [![Python CI](https://github.com/JoFrhwld/fasttrackpy/actions/workflows/test-and-run.yml/badge.svg)](https://github.com/JoFrhwld/fasttrackpy/actions/workflows/test-and-run.yml) [![codecov](https://codecov.io/gh/FastTrackiverse/fasttrackpy/graph/badge.svg?token=GOAWY4B5C8)](https://codecov.io/gh/FastTrackiverse/fasttrackpy) <a href="https://codeclimate.com/github/JoFrhwld/fasttrackpy/maintainability"><img src="https://api.codeclimate.com/v1/badges/6725fded174b21a3c59f/maintainability" /></a> [![DOI](https://zenodo.org/badge/580169086.svg)](https://zenodo.org/badge/latestdoi/580169086)


A python implementation of the FastTrack method

## Installation

```bash
pip install fasttrackpy
```

This will make the command line executable `fasttrack` available, along with its subcommands:

- `audio`
- `audio-textgrid`
- `corpus`

## Getting help

For any of the fasttrack subcommands, add the `--help` flag to
print the help info. You can also visit [the docs](https://fasttrackiverse.github.io/fasttrackpy/usage/getting_started.html).

## Usage

For a single audio file containing a vowel-like sound:

```bash
fasttrack audio --file audio.wav \
    --output formants.csv
```

For a paired audio file and textgrid with intervals defining
target audio to process:

```bash
fasttrack audio-textgrid --audio audio.wav \
    --textgrid audio.TextGrid \
    --output formants.csv
```

For a corpus directory of paired audio files and textgrid

```bash
fasttrack corpus --corpus dir/ \
    --output formants.csv
```