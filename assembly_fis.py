import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Assembly FIS Dashboard", layout="wide")
st.title("🏭 Module 2: Automated Assembly Error Identification")
st.markdown("### Fuzzy Inference System for Mechanical Fault Detection")

# --- 1. SYSTEM DESIGN (TASK 1) ---
# Define Universes
torque = ctrl.Antecedent(np.arange(0, 11, 1), 'torque')
alignment = ctrl.Antecedent(np.arange(0, 6, 1), 'alignment')
acoustic = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'acoustic')
severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

# Membership Functions
torque['low'] = fuzz.trapmf(torque.universe, [0, 0, 2, 4])
torque['medium'] = fuzz.trimf(torque.universe, [2, 5, 8])
torque['high'] = fuzz.trapmf(torque.universe, [6, 8, 10, 10])

alignment['small'] = fuzz.trapmf(alignment.universe, [0, 0, 1, 2])
alignment['moderate'] = fuzz.trimf(alignment.universe, [1, 2.5, 4])
alignment['large'] = fuzz.trapmf(alignment.universe, [3, 4, 5, 5])

acoustic['normal'] = fuzz.trapmf(acoustic.universe, [0, 0, 0.2, 0.4])
acoustic['suspicious'] = fuzz.trimf(acoustic.universe, [0.2, 0.5, 0.8])
acoustic['severe'] = fuzz.trapmf(acoustic.universe, [0.6, 0.8, 1, 1])

severity['low'] = fuzz.trimf(severity.universe, [0, 25, 50])
severity['medium'] = fuzz.trimf(severity.universe, [25, 50, 75])
severity['high'] = fuzz.trimf(severity.universe, [50, 75, 100])

# --- 2. RULE BASE (TASK 2: 12 RULES) ---
rules = [
    ctrl.Rule(torque['low'] & alignment['small'] & acoustic['normal'], severity['low']),
    ctrl.Rule(torque['high'] & alignment['large'], severity['high']),  # Key Required Rule
    ctrl.Rule(torque['high'] | alignment['large'] | acoustic['severe'], severity['high']),
    ctrl.Rule(torque['medium'] & alignment['moderate'], severity['high']),
    ctrl.Rule(torque['medium'] & acoustic['suspicious'], severity['medium']),
    ctrl.Rule(alignment['moderate'] & acoustic['suspicious'], severity['medium']),
    ctrl.Rule(torque['low'] & alignment['moderate'], severity['medium']),
    ctrl.Rule(torque['medium'] & alignment['small'], severity['medium']),
    ctrl.Rule(acoustic['severe'], severity['high']),
    ctrl.Rule(torque['high'], severity['high']),
    ctrl.Rule(alignment['large'], severity['high']),
    ctrl.Rule(torque['low'] & alignment['small'] & acoustic['suspicious'], severity['low'])
]

# Create Control System
assembly_ctrl = ctrl.ControlSystem(rules)
assembly_sim = ctrl.ControlSystemSimulation(assembly_ctrl)

# --- 3. DASHBOARD UI (TASK 4) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔧 Sensor Inputs")
    t_input = st.slider("Torque Deviation (Nm)", 0.0, 10.0, 1.0)
    a_input = st.slider("Alignment Error (mm)", 0.0, 5.0, 0.5)
    s_input = st.slider("Acoustic Index (0-1)", 0.0, 1.0, 0.1)
    
    # Run Simulation
    assembly_sim.input['torque'] = t_input
    assembly_sim.input['alignment'] = a_input
    assembly_sim.input['acoustic'] = s_input
    assembly_sim.compute()
    result = assembly_sim.output['severity']

    # Display Result
    st.subheader(f"Fault Severity: {result:.2f}%")
    if result < 40:
        st.success("STATUS: PASS (Normal)")
    elif result < 70:
        st.warning("STATUS: CHECK (Rework Likely)")
    else:
        st.error("STATUS: REJECT (Immediate Action)")

with col2:
    st.header("📈 Visualization")
    tab1, tab2 = st.tabs(["Membership Functions", "Surface Plot"])
    
    with tab1:
        fig, ax = plt.subplots()
        torque.view(sim=assembly_sim)
        st.pyplot(plt.gcf())
        st.write("Current Torque input visualized on the fuzzy sets.")

    with tab2:
        import plotly.graph_objects as go
        
        # Create data for surface
        x_range = np.linspace(0, 10, 20)
        y_range = np.linspace(0, 5, 20)
        x, y = np.meshgrid(x_range, y_range)
        z = np.zeros_like(x)
        
        for i in range(20):
            for j in range(20):
                assembly_sim.input['torque'] = x[i,j]
                assembly_sim.input['alignment'] = y[i,j]
                assembly_sim.input['acoustic'] = 0.5 
                assembly_sim.compute()
                z[i,j] = assembly_sim.output['severity']

        fig = go.Figure(data=[go.Surface(z=z, x=x_range, y=y_range, colorscale='Viridis')])
        fig.update_layout(title='3D Fault Severity Surface', autosize=False,
                          width=600, height=600,
                          margin=dict(l=65, r=50, b=65, t=90),
                          scene=dict(xaxis_title='Torque', yaxis_title='Alignment', zaxis_title='Severity'))
        st.plotly_chart(fig)

# --- 4. TEST CASES (TASK 3) ---
st.header("📋 Simulation Test Cases")
test_data = [
    [0.5, 0.2, 0.1, "Perfect Assembly"],
    [8.5, 4.5, 0.2, "Major Failure (Both)"],
    [9.0, 0.5, 0.1, "Over-Torque"],
    [1.0, 4.8, 0.1, "Misalignment"],
    [4.5, 2.5, 0.5, "Minor Wear/Loose"],
    [1.0, 0.5, 0.9, "Internal Grinding Noise"],
    [3.0, 1.0, 0.1, "Near-Perfect"],
    [7.0, 2.0, 0.4, "Torque Warning"],
    [2.0, 3.5, 0.2, "Alignment Warning"],
    [5.0, 4.0, 0.8, "Compound Critical Error"]
]

results_table = []
for td in test_data:
    assembly_sim.input['torque'] = td[0]
    assembly_sim.input['alignment'] = td[1]
    assembly_sim.input['acoustic'] = td[2]
    assembly_sim.compute()
    results_table.append({"Torque": td[0], "Align": td[1], "Acoustic": td[2], "Severity %": round(assembly_sim.output['severity'], 2), "Case": td[3]})
with st.expander("📜 View FIS Rule Base (Task 2)"):
    st.write("1. If (Torque is Low) AND (Alignment is Small) AND (Acoustic is Normal) THEN (Severity is Low)")
    st.write("2. If (Torque is High) AND (Alignment is Large) THEN (Severity is High) **[REQUIRED CASE]**")
    st.write("3. If (Torque is High) OR (Alignment is Large) OR (Acoustic is Severe) THEN (Severity is High)")
    st.write("4. If (Torque is Medium) AND (Alignment is Moderate) THEN (Severity is High)")
    st.write("5. If (Torque is Medium) AND (Acoustic is Suspicious) THEN (Severity is Medium)")
    st.write("6. If (Alignment is Moderate) AND (Acoustic is Suspicious) THEN (Severity is Medium)")
    st.write("7. If (Torque is Low) AND (Alignment is Moderate) THEN (Severity is Medium)")
    st.write("8. If (Torque is Medium) AND (Alignment is Small) THEN (Severity is Medium)")
    st.write("9. If (Acoustic is Severe) THEN (Severity is High)")
    st.write("10. If (Torque is High) THEN (Severity is High)")
    st.write("11. If (Alignment is Large) THEN (Severity is High)")
    st.write("12. If (Torque is Low) AND (Alignment is Small) AND (Acoustic is Suspicious) THEN (Severity is Low)")
st.table(results_table)