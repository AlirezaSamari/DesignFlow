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
    page_title="Design Automation", 
    page_icon="📐", 
    layout="wide"
) 

# Define the ResidualBlock and ResNet classes (for L* prediction)
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

# Load the scaler for L* prediction
with open('model_lib/L_scaler.pkl', 'rb') as f:
    scaler_l = pickle.load(f)

# Load the L* model
input_size_l = 5
output_size_l = 1
resnet_model = ResNet(input_size_l, output_size_l, residual_blocks_neurons=[32, 16, 8], dropout_p=0.5)
resnet_model.load_state_dict(torch.load('model_lib/resnet_l_model.pth'))
resnet_model.eval()

# Define the prediction function for L* (or L_ratio)
def predict_l_ratio(input_data):
    feature_names_l = ['l', 'mu_ratio', 'Q_ratio', 'st', 'R']
    df_input = pd.DataFrame(input_data, columns=feature_names_l)
    X_scaled_input = scaler_l.transform(df_input)
    X_input_tensor = torch.tensor(X_scaled_input, dtype=torch.float32)
    with torch.no_grad():
        prediction = resnet_model(X_input_tensor)
    return prediction.numpy()

# Define the FourierSeries and FourierModel classes (for Q^*_range prediction)
class FourierSeries(nn.Module):
    def __init__(self, input_size, num_harmonics):
        super(FourierSeries, self).__init__()
        self.input_size = input_size
        self.num_harmonics = num_harmonics
        self.coefficients = nn.Parameter(torch.randn(self.input_size, self.num_harmonics) * 0.1)
        self.amplitude_an = nn.Parameter(torch.randn(1, self.input_size, self.num_harmonics) * 0.1)
        self.amplitude_bn = nn.Parameter(torch.randn(1, self.input_size, self.num_harmonics) * 0.1)
        self.bias = nn.Parameter(torch.randn(1, self.input_size))

    def forward(self, x):
        batch_size = x.size(0)
        gain = torch.zeros(batch_size, self.input_size)
        for i in range(batch_size):
            harmonic_fn = self.amplitude_an * torch.cos(torch.matmul(x[i].unsqueeze(0), self.coefficients)) + \
                          self.amplitude_bn * torch.sin(torch.matmul(x[i].unsqueeze(0), self.coefficients))
            harmonic_fn = harmonic_fn + self.bias.unsqueeze(2)
            gain[i] = torch.sum(harmonic_fn, dim=2)
        return gain

class FourierModel(nn.Module):
    def __init__(self, input_size, num_harmonics, output_size):
        super(FourierModel, self).__init__()
        self.fourier = FourierSeries(input_size, num_harmonics)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        fourier_output = torch.nn.functional.relu(self.fourier(x))
        output = self.linear(fourier_output)
        return output

# Load the scaler for Q^*_range prediction
with open('model_lib/Q_scaler.pkl', 'rb') as f:
    scaler_q = pickle.load(f)

# Load the Q^*_range model
input_size_q = 3
output_size_q = 7  # 7 classes for classification
fourier_model = FourierModel(input_size_q, 60, output_size_q)
fourier_model.load_state_dict(torch.load('model_lib/fourier_classification_q_ratio_model.pth'))
fourier_model.eval()

# Define the prediction function for Q^*_range
def predict_q_range(input_data):
    feature_names_q = ['mu_ratio', 'st', 'R']
    df_input = pd.DataFrame(input_data, columns=feature_names_q)
    X_scaled_input = scaler_q.transform(df_input)
    X_input_tensor = torch.tensor(X_scaled_input, dtype=torch.float32)
    with torch.no_grad():
        output = fourier_model(X_input_tensor)
        probabilities = F.softmax(output, dim=1)
    return probabilities.numpy()

# Streamlit app title
st.title("Design Automation")

# Initialize session state if not already done
if "q_range" not in st.session_state:
    st.session_state.q_range = None
if "l_ratio" not in st.session_state:
    st.session_state.l_ratio = None

# Create synchronized slider and number input for each feature
def synchronized_input(label, min_val, max_val, initial_val, step, format_str):
    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed
    
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
    
    # Ensure synchronization
    if number_val != slider_val:
        slider_val = number_val
        number_val = slider_val

    return slider_val

mu_ratio = synchronized_input(r"$\mu^*$", 0.017, 0.033, 0.017, 0.0001, "%.4f")
st_val = synchronized_input(r"$St$", 0.015, 8.164, 0.015, 0.001, "%.3f")
R_val = synchronized_input(r"$R^*$", 0.03, 0.84, 0.03, 0.01, "%.2f")

# Step 2: Predict Q^*_range
q_dict = {0:r'0.0020 <$Q^*_{Range}$< 0.0071',
 1:r'0.0071 <$Q^*_{Range}$< 0.0107',
 2:r'0.0107 <$Q^*_{Range}$< 0.0138',
 3:r'0.0138 <$Q^*_{Range}$< 0.0179',
 4:r'0.0179 <$Q^*_{Range}$< 0.0245',
 5:r'0.0245 <$Q^*_{Range}$< 0.0330',
 6:r'0.0330 <$Q^*_{Range}$< 0.1490'}

if st.button(r"Predict $Q^*_{range}$"):
    input_data_q = np.array([[mu_ratio, st_val, R_val]])
    probabilities = predict_q_range(input_data_q)

    # Find the index of the maximum probability
    max_prob_index = np.argmax(probabilities[0])

    # Get the corresponding class label
    st.session_state.q_range = q_dict[max_prob_index]  # Store in session state

# Display the Q^*_range prediction if available
if st.session_state.q_range:
    st.write(f"{st.session_state.q_range}")

l_design = synchronized_input(r"$l_{Design}\; (\mu m)$", 82.6, 233.0, 82.6, 0.1, "%.1f")
Q_ratio = synchronized_input(r"$Q^*$", 0.002, 0.149,0.002, 0.001, "%.3f")

# Step 4: Predict l*
if st.button(r"Predict $l^*$"):
    # Convert $l_{Design}$ from micrometers to meters for L* prediction
    l_design_m = l_design * 1e-6
    input_data_l = np.array([[l_design_m, mu_ratio, Q_ratio, st_val, R_val]])
    prediction_l = predict_l_ratio(input_data_l)
    st.session_state.l_ratio = prediction_l[0][0]
# Display the l* prediction if available
if st.session_state.l_ratio is not None:
    st.write(f"$l^*$ = {st.session_state.l_ratio:.4f}")

    # Plot the channel geometry
    st.markdown("<h3 style='color: #87CEEB;'>Channel Geometry</h3>", unsafe_allow_html=True)

    def plot_channel_geometry(l_design, l_ratio, R_star, Q_ratio, st_number):
        
        R = R_star * l_design
        big_channel_width = 10 * l_design
        big_channel_height = l_design
        small_channel_width = big_channel_width / 20
        small_channel_height = l_ratio * l_design

        distance_star = 4 * Q_ratio / ((np.pi * R_star ** 2) * st_number)
        distance = distance_star * l_design

        num_droplets = int((big_channel_width - (small_channel_width + R)) / distance) + 1
        droplet_positions = [small_channel_width + 1.6 * R]
        for i in range(1, num_droplets):
            droplet_positions.append(droplet_positions[0] + i * distance)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0)  # Make the figure background transparent
        ax.patch.set_alpha(0)   
        ax.plot([0, big_channel_width], [0, 0], 'w-', lw=2)
        ax.plot([0, 0], [0, big_channel_height], 'w--', lw=1)
        ax.plot([0, big_channel_width], [big_channel_height, big_channel_height], 'w-', lw=2)
        ax.plot([big_channel_width, big_channel_width], [0, big_channel_height], 'w--', lw=1)

        small_channel_x = 0
        small_channel_y = (big_channel_height - small_channel_height) / 2
        ax.plot([small_channel_x, small_channel_x + small_channel_width], [small_channel_y, small_channel_y], 'w-', lw=2)
        ax.plot([small_channel_x, small_channel_x], [small_channel_y, small_channel_y + small_channel_height], 'w--', lw=1)
        ax.plot([small_channel_x, small_channel_x + small_channel_width], [small_channel_y + small_channel_height, small_channel_y + small_channel_height], 'w-', lw=2)
        # ax.plot([small_channel_x + small_channel_width, small_channel_x + small_channel_width], [small_channel_y, small_channel_y + small_channel_height], 'k-', lw=2, linestyle='--')

        droplet_start_y = small_channel_y + small_channel_height / 2
        for pos in droplet_positions:
            ax.add_patch(plt.Circle((pos, droplet_start_y), R, color='#87CEEB', alpha=1))

        st.write(f"$l_d$ = {st.session_state.l_ratio*l_design:.1f} μm")
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

    plot_channel_geometry(l_design, st.session_state.l_ratio, R_val, Q_ratio, st_val)

