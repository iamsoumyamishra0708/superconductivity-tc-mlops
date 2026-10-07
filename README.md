<div align="center">

# ⚡ Superconductor Critical Temperature ($T_c$) Predictor
### *Advanced MLOps Pipeline with FastAPI, Streamlit, Docker & AWS EC2*

[![CI/CD Pipeline](https://github.com/iamsoumyamishra7/superconductor-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/iamsoumyamishra7/superconductor-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws)](https://aws.amazon.com/)

</div>

---

## 🚀 Overview
An end-to-end production-grade **MLOps application** designed to predict the Critical Temperature ($T_c$) of superconducting materials using advanced Machine Learning (`XGBoost Regressor`). The system features a fully automated **CI/CD pipeline via GitHub Actions**, containerized multi-service architecture using **Docker & Docker Compose**, and cloud deployment on **AWS EC2**.

---

## 🛠️ Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend API** | FastAPI, Uvicorn | High-performance asynchronous inference REST API |
| **Frontend UI** | Streamlit, Plotly | Interactive dashboard with real-time telemetry & gauge metrics |
| **ML Engine** | Scikit-Learn, XGBoost | Regressor model trained on multi-dimensional material matrices |
| **Containerization** | Docker, Docker Compose | Isolated multi-container deployment |
| **CI/CD Pipeline** | GitHub Actions | Automated build, push to Docker Hub, and EC2 deployment |
| **Cloud Infrastructure** | AWS EC2 (Ubuntu) | Scalable cloud hosting environment |

---

## 🌟 Key Features
- **⚡ Real-Time Inference:** Instant critical temperature predictions with interactive gauge visualizations.
- **📊 Advanced Analytics:** Multi-dimensional feature analysis via interactive bar and radar charts.
- **📜 Session Logs & Export:** Track prediction history within the session and export reports directly as CSV.
- **🔄 Fully Automated CI/CD:** Zero-touch deployments where every `git push` triggers automated testing, container building, and cloud updating.
- **🔒 Secure & Scalable:** Session-persisted endpoints, health-check telemetry, and robust error handling.

---

## 📂 Project Directory Structure
```text
superconductor-mlops/
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions CI/CD configuration
├── frontend/
│   └── app.py                # Streamlit UI dashboard
├── backend/
│   └── main.py               # FastAPI inference server & model loader
├── Dockerfile                # Multi-stage optimized container build
├── docker-compose.yml        # Service orchestration file
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation