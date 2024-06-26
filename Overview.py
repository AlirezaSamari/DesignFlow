import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="DesignFlow",
    page_icon="💧",
    layout="wide",
)

st.title('DesignFlow')
st.write("Welcome to **DesignFlow**, a software tool developed during my thesis to simplify design automation for microfluidic co-flow droplet generation. DesignFlow aims to make design processes more efficient by providing easy-to-use features and reliable performance. Whether you’re working on optimizing designs or managing workflows, DesignFlow offers practical solutions to help you achieve your goals. Explore the straightforward, user-friendly approach of DesignFlow and enhance your design efficiency.")

st.markdown("""
The image below illustrates the geometry of the co-flow system used in our design automation tool. 
We use specific aspect ratios and dimensionless numbers as key features in our predictions:

- **$ \\displaystyle l^* = \\frac{l_d}{l_{Design}} $**: Ratio of droplet size to design length.
- **$ \\displaystyle R^* = \\frac{R}{l_{Design}} $**: Dimensionless radius of curvature.
- **$ \\displaystyle \\mu^* = \\frac{\\mu_d}{\\mu_c} $**: Ratio of droplet viscosity to continuous phase viscosity.
- **$ \\displaystyle Q^* = \\frac{Q_d}{Q_c} $**: Ratio of droplet flow rate to continuous phase flow rate.
- **$ \\displaystyle Ca_c = \\frac{\\mu_c u_c}{\\sigma} $**: Capillary number of the continuous phase.
- **$ \\displaystyle We_d = \\frac{\\rho_d u_d^2 l_{Design}}{\\sigma} $**: Weber number of the droplet phase.
- **$ \\displaystyle St = \\frac{fl_{Design}}{u_c} $**: Strouhal number of the flow.

""")

st.image(Image.open('Images/GEO.png'), caption='Geometric Representation')

st.write('For inquiries or more information, you can reach out to me and follow my work through the platforms below:')
st.write('[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlirezaSamari)')
st.write('[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alireza-samari-255819159/)')
st.markdown('<p style="font-size: 18px; color: #008CBA;"><strong>Email:</strong> alirexasamari@gmail.com</p>', unsafe_allow_html=True)
