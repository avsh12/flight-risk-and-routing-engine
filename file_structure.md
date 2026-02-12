.
├── config
│   ├── config.yaml
│   ├── data_dependency.yaml
│   └── schema.yaml
├── dags
├── data
│   ├── airlines.csv
│   ├── airports.csv
│   ├── bronze
│   ├── flights.csv
│   ├── gold
│   └── silver
├── docker
│   ├── Dockerfile.etl
│   └── Dockerfile.ml
├── file_structure.md
├── logs
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── etl
│   │   ├── airports.py
│   │   ├── date_encoding.py
│   │   ├── flight
│   │   │   ├── clean.py
│   │   │   ├── extract.py
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── extract.cpython-312.pyc
│   │   │   │   └── __init__.cpython-312.pyc
│   │   │   ├── schema_validation.py
│   │   │   └── transform.py
│   │   ├── index_categories.py
│   │   ├── __init__.py
│   │   ├── join.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-312.pyc
│   │   ├── scale.py
│   │   └── weather
│   │       ├── clean.py
│   │       ├── extract.py
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── extract.cpython-312.pyc
│   │       │   └── __init__.cpython-312.pyc
│   │       └── transform.py
│   ├── flight_risk.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── models
│   │   └── __init__.py
│   └── utils
│       ├── config_loader.py
│       ├── constants.py
│       ├── db_client.py
│       ├── __init__.py
│       ├── loaders.py
│       ├── __pycache__
│       │   ├── constants.cpython-312.pyc
│       │   ├── __init__.cpython-312.pyc
│       │   └── loaders.cpython-312.pyc
│       └── settings.py
├── structure.md
└── tests
    ├── etl_pipeline.py
    ├── flight_delay.ipynb
    ├── test_etl.py
    └── test_models.py

21 directories, 52 files
