# VS2F2D
**V**ery **S**imple **F**unction **F**it **to** **D**ata

:memo: **Note:** This all is created in few minutes.

<img src="https://github.com/user-attachments/assets/121bec23-ac10-4d84-8ee2-31499fa3811d" height="300">

## Features
- Fit functions
  - linear
  - quadratic
  - root
  - exponencial
  - reciprocal
  - logarithmic
  - sine
- Plot function as dots to `.csv`
- Set delimiter and decimal chars
- Print fitted function expression
- Cut data with ui (`trim.py`)
## How to run
Create and activate python environment
```console
python -m venv env
. env/bin/activate
```
Install all requirements
```console
pip install -r requirements.txt
```
Run
```console
python fit_function.py
```
Example with arguments
```console
python fit_function.py -F LED_2.csv -x V -y mA -f exponential
```
