# NeuroAI

NeuroAI is a data-driven web application designed for the analysis, processing, and machine learning inference of neurophysiological data (such as EEG signals). It provides a user-friendly Streamlit dashboard for uploading data, visualizing signals, analyzing history, and performing automated inference using deep learning models.

## Features

- **Interactive Dashboard:** Built with Streamlit and Plotly for rich, interactive data visualization.
- **Data Upload & Management:** Upload and manage raw neurological data files securely.
- **Signal Processing Pipeline:** Preprocessing pipelines for cleaning and preparing neurological data using MNE and SciPy.
- **Machine Learning Inference:** Integrated PyTorch models for performing inference and predictions on processed signals.
- **History & Tracking:** Keep track of past analysis and model predictions.
- **Admin Interface:** Dedicated admin controls for managing the application state and configurations.

## Project Structure

```text
NEUROAI/
├── admin/          # Administrative scripts and tools
├── app/            # Streamlit application (frontend)
│   ├── pages/      # Streamlit pages (Upload, Dashboard, History, Admin)
│   └── storage.py  # Storage utilities for the app
├── data/           # Raw and processed datasets
├── deployment/     # Deployment configurations and scripts
├── docs/           # Project documentation
├── inference/      # Machine learning inference modules
├── models/         # Saved PyTorch models
├── outputs/        # Output reports and exported results
├── tests/          # Unit and integration tests
├── training/       # Model training scripts and dataset loaders
└── utils/          # General utility functions (logger, plotting, etc.)
```

## Technologies Used

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Neuroscience Data Processing:** [MNE-Python](https://mne.tools/)
- **Machine Learning:** PyTorch, Scikit-Learn
- **Scientific Computing:** SciPy

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd NEUROAI
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit web application, run the following command from the root of the project:

```bash
streamlit run app/pages/Upload.py
```
*(Note: Adjust the main entry point to your primary Streamlit file if you have a wrapper in the root directory)*

## Usage

1. Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Navigate to the **Upload** page to securely upload your `.mat`, `.dat`, or other supported neuro data files.
3. Use the **Dashboard** to visualize the signals and view model inference results.
4. Check the **History** tab to review previous analyses.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
