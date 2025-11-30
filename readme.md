# 🤖 AI Job Market Analysis & Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)

## Project Overview
This project provides a comprehensive analysis of the global **AI Job Market**. By analyzing thousands of job postings, we aim to uncover trends related to salaries, in-demand skills, hiring volume, and geographic hubs.

The project consists of:
1.  **Data Analysis Pipeline:** A Jupyter Notebook for data cleaning, feature engineering, and exploratory data analysis (EDA).
2.  **Interactive Dashboard:** A Streamlit web application allowing users to explore the data dynamically.

## Key Insights
Based on the analysis of the dataset, we identified several key trends:
* **Balanced Market:** Hiring is evenly distributed across various industries and company sizes, indicating a healthy ecosystem.
* **Entry-Level Opportunities:** There is a significant volume of jobs for entry-level talent, debunking the myth that AI is for seniors only.
* **Location Matters:** Geography is the primary driver for salary variance, more so than company size.
* **Hiring Seasonality:** **Q1** and **August** show the highest hiring activity.

## 🛠️ Tech Stack
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (MultiLabelBinarizer for skills processing)
* **Dashboarding:** Streamlit
* **Data Source:** KaggleHub

## 📂 Project Structure
AI Job Market Analysis/
├── app.py                   # Streamlit Dashboard Application
├── main_notebook.ipynb      # Analysis Notebook (Cleaning & EDA)
├── main_notebook.html       # Analysis Notebook (Cleaning & EDA) "Web Page with Code"
├── requirements.txt         # Project Dependencies
├── README.md                # Project Documentation
└── data/
    ├── araw_ai_job_market.csv      # Original Dataset
    ├── state_map.json        
    └── ai_job_market_enhanced.csv  # Processed Dataset

---
**Author:** George Yacoub Fayez
**LinkedIn:** https://www.linkedin.com/in/george-yacope