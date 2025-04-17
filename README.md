# DesignFlow
 A tool to simplify design automation for microfluidic co-flow droplet generation.
Here is the link of the Web-based app: https://designcoflow.streamlit.app/
# DesignFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)  
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-orange.svg)](https://designcoflow.streamlit.app/)

A Streamlit‑based tool to simplify design automation for microfluidic co‑flow droplet generation. DesignFlow combines:

- **Forward Prediction:** Estimate droplet diameter and generation rate from channel geometry, flow rates, and fluid properties.  
- **Reverse Design:** Compute channel geometry ratios needed to achieve target droplet size and rate.  

Whether you’re exploring design space or optimizing workflows, DesignFlow provides an interactive, no‑code interface to accelerate your microfluidic designs.

---

## 📖 Table of Contents

1. [Demo](#demo)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation & Setup](#installation--setup)  
5. [Running the App](#running-the-app)  
6. [Usage Guide](#usage-guide)  
7. [Project Structure](#project-structure)  
8. [Configuration & Customization](#configuration--customization)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Contact & Acknowledgments](#contact--acknowledgments)  

---

## 🚀 Demo

Check out the live app here:  
👉 https://designcoflow.streamlit.app/

---

## ✨ Features

- **Forward Prediction**  
  - Input: channel dimensions (width, height), fluid properties (viscosity, interfacial tension), flow rates.  
  - Output: predicted droplet diameter & generation rate.

- **Reverse Design (Geometry Optimizer)**  
  - Input: desired droplet diameter & rate.  
  - Output: optimized channel width/height ratios and flow settings.

- **Interactive UI**  
  - Built with Streamlit for fast, in‑browser interaction—no coding required.

- **Modular Codebase**  
  - Core algorithms in `model_lib/`, UI pages under `pages/`, and configuration in `.streamlit/`.

- **Dev Container Support**  
  - `.devcontainer/` folder for an easy VS Code Remote‑Containers setup.

---

## 🛠️ Prerequisites

- **Python:** 3.8 or newer  
- **Git:** for cloning the repository  
- **(Optional) VS Code + Dev Containers:** if you want a reproducible development environment

---

## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AlirezaSamari/DesignFlow.git
   cd DesignFlow

