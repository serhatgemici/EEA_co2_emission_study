## 🚖 CO2 emissions prediction by passenger vehicles

An ML implemetation for our Data Scientist training project at Liora that predicts CO2 emissions for a given set of specs of a passenger vehicle.
The project uses the free and openly published [European Enivronment Agency (EEA) CO2 emissions from new passenger cars dataset](https://co2cars.apps.eea.europa.eu/?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2025%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D) (year = 2024, status = final, Country = France) to train and validate the models, and (year = 2025, status = provisional, Country = France) data for testing.

## Installation

The project uses [uv as package manager](https://docs.astral.sh/uv/). To install the python environment and all the dependencies of the project, simply execute the following command on the root directory of the project in terminal:

```bash
% uv sync
```

Once the installation is completed, use the `jul_26_bds_in_co2` python environment inside the jupyter notebooks.

2025 Data is provisional and is subject to changes. For a smooth start, [please download the data folder](https://drive.google.com/file/d/1XJg47GvBsXUMxc4bFmNEzE2JYm6rfXK3/view?usp=sharing), extract it and replace the data folder inside the root folder with it.

## Running the Streamlit app

The Streamlit app contains the live report and a "try for yourself" section for on-the-fly prediction demonstration.
App files are in the `src/streamlit` directory. In order to run the Streamlit app simply follow these instructions in Terminal in project root folder:

```bash
% cd src/streamlit
% uv run streamlit run app.py
```

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
