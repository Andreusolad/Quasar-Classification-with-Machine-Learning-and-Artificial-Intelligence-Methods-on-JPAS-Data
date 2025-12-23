# Quasar Classification Using Machine Learning on JPAS Photometric Data

A machine learning pipeline for classifying quasars from galaxies and stars using photometric data from the J-PAS (Javalambre Physics of the Accelerating Universe Astrophysical Survey). This project implements and compares multiple classification algorithms including Random Forest, Stochastic Gradient Descent (SGD), and Multi-Layer Perceptron (MLP) neural networks.

## Project Overview

Quasars are among the most distant and luminous objects in the universe, making their identification crucial for cosmological studies. This project uses supervised learning techniques to classify astronomical objects based on their spectral features, achieving strong performance metrics:

- **Random Forest**: F1-score ~0.96, Precision ~0.97
- **SGD Classifier**: Optimized with hyperparameter tuning
- **Neural Network (MLP)**: Trained with class balancing for improved quasar detection

### Key Features

- Multi-algorithm comparison (Random Forest, SGD, MLP)
- Comprehensive hyperparameter tuning using GridSearchCV and Optuna
- Feature importance analysis to identify key spectral characteristics
- Handling of class imbalance through weighting strategies
- Evaluation on both mock and real JPAS data

## Technical Details

### Data Processing
- **Input**: 57-band photometric data from J-PAS survey
- **Features**: Flux measurements, flux errors, coordinates (RA/DEC), morphological type, and auxiliary photometry
- **Classes**: Quasar (QSO), Galaxy, Star
- **Preprocessing**: Null value handling, feature scaling (StandardScaler), class balancing

### Models Implemented

1. **Random Forest Classifier**
   - Hyperparameter optimization via grid search
   - Feature importance analysis using Mean Decrease in Impurity (MDI)
   - Class weighting to handle imbalanced data

2. **Stochastic Gradient Descent (SGD)**
   - Multiple loss functions tested (log_loss, hinge, modified_huber)
   - L1/L2/ElasticNet regularization
   - Adaptive learning rate with early stopping

3. **Multi-Layer Perceptron (MLP)**
   - Architecture: 2 hidden layers (100, 50 neurons)
   - Oversampling strategy for minority class (quasars)
   - Weight-based feature importance analysis

### Evaluation Metrics
- Precision (Purity)
- Recall (Completeness)
- F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion matrices

## Project Structure
```
├── classifier_pef2.ipynb      # Main notebook with full pipeline
├── data/                      # Dataset directory (not included)
├── README.md                  # Project documentation
```

### Data Availability

**Important**: The J-PAS photometric dataset used in this project is **not publicly available** and is not included in this repository due to collaboration agreements. To replicate this work:

- Request access through the [J-PAS collaboration](https://www.j-pas.org/)
- Use your own authorized astronomical survey data
- Adapt the preprocessing pipeline to your dataset format

## Requirements
```
python>=3.8
scikit-learn
numpy
pandas
matplotlib
astropy
optuna
```

## Usage

1. **Data Preparation** (if you have authorized access):
```python
   # Place J-PAS files in data/ directory:
   # - Mock_train_FLUX+NOISE.npy
   # - Mock_train_PROPS.csv
   # - Mock_valid_FLUX+NOISE.npy
   # - Mock_valid_PROPS.csv
   # - etc.
```

2. **Run the Classification Pipeline**:
   Open and execute `classifier_pef2.ipynb` sequentially:
   - Data loading and preprocessing
   - Model training and hyperparameter tuning
   - Model evaluation and comparison
   - Feature importance analysis

3. **Experiment with Models**:
   Modify hyperparameters in the training cells to optimize performance for your specific dataset.

## Results Highlights

- Successfully classified quasars with **>96% F1-score** using Random Forest
- Identified key spectral features through importance analysis
- Demonstrated effective handling of class imbalance (quasars represent ~5% of data)
- Validated performance on real J-PAS observations

## Future Improvements

- Implement deep learning architectures (CNNs for spectral data)
- Add cross-validation for more robust performance estimates
- Explore ensemble methods combining multiple classifiers
- Optimize for real-time classification on large datasets

## About the Author

Physics Engineering student with strong interests in astrophysics, machine learning, and data science. Experienced in applying ML techniques to complex scientific problems, with a focus on space technology and financial applications.

**Skills**: Python, Machine Learning (scikit-learn, TensorFlow), Data Analysis, Astrophysics, MATLAB

## Contact

**Andreu Solà**  
📧 andreusolad@gmail.com | andreu.sola@estudiantat.upc.edu  
💼 [LinkedIn] www.linkedin.com/in/andreu-solà-i-dagas 
🔗 [GitHub] https://github.com/Andreusolad

---

*This project was developed as part of my Physics Engineering degree, applying machine learning techniques to astrophysical data analysis.*
