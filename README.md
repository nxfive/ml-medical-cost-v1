# First ML Project – Medical Cost Prediction

## 🚀 Project Overview

This was my first machine learning project, created to combine new tools into a functional end-to-end workflow. The primary goal was simple: **make it work**.

It demonstrates:

- Model serving with BentoML
- Experiment tracking using MLflow
- Deployment to a private VPS via GitLab CI/CD

The project reflects a functional approach to ML deployment rather than polished engineering—perfect for tracking my growth over time.

## 🧠 Model & Optimization

- The model was chosen based on its performance during initial analysis in a notebook.
- Further optimization was done with **Optuna**, with hyperparameters stored in params.yaml.

The same configuration file also contains training and serving parameters.

## 📊 Data Handling

- No feature transformations are applied.
- Categorical values like "yes"/"no" are left as-is because the model handles them effectively.

## ⚡ Why Share This Project?

This project serves as a **baseline** to track my growth in ML and engineering skills. <br>The **second version** of this project demonstrates clear improvements in:

- **Code structure & modularity** – implemented clean architecture, following SOLID principles and architectural patterns

- **Data preprocessing & feature engineering** – features are prepared in a way that all models can handle them effectively

- **Deployment & APIs** – added a FastAPI endpoint; the API serves the best-performing model via BentoML

- **Model training & optimization** – trains all models explored in the notebook, while optimizing only the best-performing one

- **Code quality & maintainability** – added type annotations, and docstrings for every function/method

Sharing this **first version** helps visualize how my workflow and engineering practices have evolved over time.

Check out the next version of this project <br >
[![Version 2](https://img.shields.io/badge/version-2-blue)](https://github.com/nxfive/ml-medical-cost-v2)
