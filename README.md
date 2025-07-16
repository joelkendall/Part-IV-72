# Part-IV-72
A dependency view of software evolution

## Setup

1. Python 3.12.2 or compatible verison

2. Create virtual environment
    ```
    python -m venv .venv
    ```

3. Run the virtual environment

    Windows:
    ```
    .venv\Scripts\activate
    ```

    Mac/Linux
    ```
    source .venv/bin/activate
    ```

4. Install dependencies
    ```
    pip install -r requirements.txt
    ```
    or

    ```
    pip install -e .
    ```

## Running Script

1. Navigate to src directory

2. Run the script
    ```
    python TSVReader.py <path to directory containing tsv files> 
    ```
3. Optional args

    Exclude java.lang dependencies
    ```
    --oj
    ```

    Exclude all java dependencies
    ```
    --oja
    ```

    eg.
    ```
    python TSVReader.py data/junit-depfiles/junit-depfiles --oj
    ```

## Running GUI

1. Navigate to src directory

2. Run GUIMain.py via IDE or Command Line



    
