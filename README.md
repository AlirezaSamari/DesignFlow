# DesignFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)  
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-orange.svg)](https://designcoflow.streamlit.app/)

A Streamlit‑based tool to simplify design automation for microfluidic co‑flow droplet generation. DesignFlow combines:

- **Forward Prediction:** Estimate droplet diameter and generation rate from channel geometry, flow rates, and fluid properties.  
- **Reverse Design:** Compute channel geometry ratios needed to achieve target droplet size and rate.  

Whether you’re exploring design space or optimizing workflows, DesignFlow provides an interactive, no‑code interface to accelerate your microfluidic device designs.

---

## 🚀 Check out the live app here:  
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

