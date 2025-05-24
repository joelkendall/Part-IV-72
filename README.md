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

## Running Script

1. Navigate to root directory

2. Run the script
    ```
    python src/TSVReader.py
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
    python src/TSVReader.py --oj
    ```


    
