# Water Potability Prediction Project

![Water Potability](https://img.shields.io/badge/Water%20Potability-Prediction%20Project-blue)

This repository contains a water potability prediction project, where the goal is to predict whether water from different water bodies is safe for human consumption or not. Access to safe drinking water is crucial for public health, and this project focuses on using machine learning techniques to assess the potability of water based on various water quality metrics.

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Project Highlights](#project-highlights)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)

## About

Access to safe drinking water is a fundamental human right and a critical aspect of health and development. This project aims to address the important issue of water quality assessment by utilizing machine learning to predict whether water is safe for human consumption.

## Dataset

The `water_potability.csv` file contains water quality metrics for 3276 different water bodies. The dataset includes the following features:

1. pH value
2. Hardness
3. Total dissolved solids (TDS)
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic carbon
8. Trihalomethanes (THMs)
9. Turbidity
10. Potability (target variable)

The 'Potability' label indicates whether water is safe for human consumption (Potable) or not (Not Potable).

## Project Highlights

- Leveraged machine learning techniques to predict water potability.
- Analyzed key water quality metrics to assess safety for consumption.
- Developed a predictive model based on water quality features.
- Provided insights to help address the issue of safe drinking water.

## Features

- Data preprocessing and exploratory data analysis (EDA).
- Feature selection and engineering.
- Model selection and training.
- Model evaluation and performance metrics.

## Getting Started

To get started with this project, follow these steps:

### Installation

Clone the repository:

```bash
git clone https://github.com/zenithmacwan/water-potability-prediction.git
cd water-potability-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the water potability prediction script:

```bash
python predict_potability.py
```

## Model Evaluation

The performance of the water potability prediction model is evaluated using various metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. These metrics provide insights into the model's ability to predict water potability accurately.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.



---

By Zenith Macwan(https://github.com/zenithmacwan)
