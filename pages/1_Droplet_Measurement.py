import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config( 
    page_title="Droplet Measurement", 
    page_icon="🌀", 
    layout="wide"
) 

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0.5):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(output_size, output_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc_residual = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc_residual.weight)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        residual = self.fc_residual(residual)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_size, output_size, residual_blocks_neurons, dropout_p=0.5):
        super(ResNet, self).__init__()
        layers = []
        prev_layer_size = input_size
        for neurons in residual_blocks_neurons:
            residual_block = ResidualBlock(prev_layer_size, neurons, dropout_p)
            layers.append(residual_block)
            prev_layer_size = neurons

        output_layer = nn.Linear(prev_layer_size, output_size)
        nn.init.xavier_uniform_(output_layer.weight)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


with open('model_lib/R_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_size = 5
output_size = 1
resnet_model_R = ResNet(input_size, output_size, residual_blocks_neurons=[32, 16, 8], dropout_p=0.0)
resnet_model_R.load_state_dict(torch.load('model_lib/resnet_R_model.pth'))
resnet_model_R.eval()

resnet_model_St = ResNet(input_size, output_size, residual_blocks_neurons=[32, 16, 8], dropout_p=0.0)
resnet_model_St.load_state_dict(torch.load('model_lib/resnet_St_model.pth'))
resnet_model_St.eval()

def predict_R_St(input_data):
    feature_names = ['l_ratio', 'mu_ratio', 'we_d', 'ca_c', 'Q_ratio']
    df_input = pd.DataFrame(input_data, columns=feature_names)
    X_scaled_input = scaler.transform(df_input)
    X_input_tensor = torch.tensor(X_scaled_input, dtype=torch.float32)
    with torch.no_grad():
        prediction_R = resnet_model_R(X_input_tensor)
        prediction_St = resnet_model_St(X_input_tensor)
    return prediction_R.numpy(), prediction_St.numpy()
st.title("Flow Dynamics")
def synchronized_input(label, min_val, max_val, initial_val, step, format_str):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        slider_val = st.slider(
            label,
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(initial_val), 
            step=float(step), 
            format=format_str,
            key=f"slider_{label}"
        )
    
    with col2:
        number_val = st.number_input(
            ' ', 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=slider_val, 
            step=float(step),
            format=format_str,
            key=f"number_{label}"
        )
    
    if number_val != slider_val:
        slider_val = number_val
        number_val = slider_val

    return slider_val

l_ratio = synchronized_input(r"$l^*$", 0.196, 0.75, 0.196, 0.001, "%.3f")
mu_ratio = synchronized_input(r"$\mu^*$", 0.017, 0.033, 0.017, 0.0001, "%.4f")
we_d = synchronized_input(r"$We_d$", 0.00016, 0.1020, 0.00016, 0.00001, "%.5f")
ca_c = synchronized_input(r"$Ca_c$", 0.1, 3.9, 0.1, 0.001, "%.3f")
Q_ratio = synchronized_input(r"$Q^*$", 0.00196, 0.149, 0.00196, 0.0001, "%.4f")
l_design = synchronized_input(r"$l_{Design}\; (\mu m)$", 82.6, 233.0, 82.6, 0.1, "%.1f")

if st.button(r"Predict $R^*$, $St$"):
    input_data = np.array([[l_ratio, mu_ratio, we_d, ca_c, Q_ratio]])
    prediction_R_star, prediction_St = predict_R_St(input_data)
    st.session_state.R_star = prediction_R_star[0][0]
    st.session_state.St = prediction_St[0][0]

if "R_star" in st.session_state and "St" in st.session_state:
    st.write(f"$R^*$ = {st.session_state.R_star:.4f}")
    st.write(f"$St$ = {st.session_state.St:.4f}")

    st.markdown("<h3 style='color: #87CEEB;'>Channel Geometry</h3>", unsafe_allow_html=True)

    def plot_channel_geometry(l_design, l_ratio, R_star, Q_ratio, St):

        R = R_star * l_design
        big_channel_width = 10 * l_design
        big_channel_height = l_design
        small_channel_width = big_channel_width / 20
        small_channel_height = l_ratio * l_design

        distance_star = 4 * Q_ratio / ((np.pi * R_star ** 2) * St)
        distance = distance_star * l_design

        num_droplets = int((big_channel_width - (small_channel_width + R)) / distance) + 1
        droplet_positions = [small_channel_width + 1.6 * R]
        for i in range(1, num_droplets):
            droplet_positions.append(droplet_positions[0] + i * distance)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)   
        ax.plot([0, big_channel_width], [0, 0], 'w-', lw=2)
        ax.plot([0, 0], [0, big_channel_height], 'w--', lw=1)
        ax.plot([0, big_channel_width], [big_channel_height, big_channel_height], 'w-', lw=2)
        ax.plot([big_channel_width, big_channel_width], [0, big_channel_height], 'w--', lw=1)

        small_channel_x = 0
        small_channel_y = (big_channel_height - small_channel_height) / 2
        ax.plot([small_channel_x, small_channel_x + small_channel_width], [small_channel_y, small_channel_y], 'w-', lw=2)
        ax.plot([small_channel_x, small_channel_x], [small_channel_y, small_channel_y + small_channel_height], 'w--', lw=1, linestyle='--')
        ax.plot([small_channel_x, small_channel_x + small_channel_width], [small_channel_y + small_channel_height, small_channel_y + small_channel_height], 'w-', lw=2)
        # ax.plot([small_channel_x + small_channel_width, small_channel_x + small_channel_width], [small_channel_y, small_channel_y + small_channel_height], 'k-', lw=2, linestyle='--')

        droplet_start_y = small_channel_y + small_channel_height / 2
        for pos in droplet_positions:
            ax.add_patch(plt.Circle((pos, droplet_start_y), R, color='#87CEEB', alpha=1))

        st.write(f"$l_d$ = {l_design*l_ratio:.1f} μm")
        st.write(f"$R$ = {R:.1f} μm")
        ax.set_aspect('equal')
        ax.set_ylabel('Width (μm)', color='white')
        ax.set_xlabel('Length (μm)', color='white')
        ax.grid(False)
        ax.tick_params(axis='both', colors='white')

        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        plt.tight_layout()
        plt.grid(False)
        st.pyplot(fig)

    plot_channel_geometry(l_design, l_ratio, st.session_state.R_star, Q_ratio, st.session_state.St)

