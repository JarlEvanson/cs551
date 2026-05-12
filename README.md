# Usage

## Setup

Run the following commands to initialize the project. This utilizes `uv`, but `pip` works too.

```bash
uv pip install -r requirements.txt
uv pip install -r NTLFlowLyzer/requirements.txt

cd NTLFlowLyzer
python setup.py install
```

You will also need to obtain the IDS 2017 dataset (link: `https://www.unb.ca/cic/datasets/ids-2017.html`).

## Training and Analysis

Run the main project script to process the dataset, train the SSL encoder, and save the models:

```bash
python project.py
```

This will generate .joblib and .pth files representing the trained models, encoders, and preprocessors.

An interactive version is available using Juypter Notebook and `project.ipynb`.

## Deployment

Launch the streamlit application.

```bash
streamlit run deploy.py
```

Once the app is running:

- Upload a `.pcap` file
- View the pre-processing logs from NTLFlowLyzer.
- View the flow-by-flow and model-by-model predictions from each model.
- Select a specific flow and model to view the SHAP plot explaining the classification.
