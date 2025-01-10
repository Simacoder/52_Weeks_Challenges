# Real-Time Network Traffic Analysis Dashboard

A Streamlit-based application for real-time monitoring and analysis of network traffic. The dashboard provides insights into network packets, including protocol distribution, traffic timelines, and geographical mapping of IPs. It also includes anomaly detection and traffic alerts.

# Features
**Real-Time Packet Capture:**
- Capture live network traffic using Scapy.
**Data Visualization:**
- Protocol distribution (Pie chart).
**Traffic timeline (Line chart).**
- Top source IP addresses (Bar chart).
**Geographical Mapping:**
- Visualize source IP locations on an interactive map.
**Anomaly Detection:**
- Uses an Isolation Forest model to detect unusual traffic patterns.
**Traffic Alerts:**
- Identifies high traffic from specific IPs and excessive UDP traffic.

# Prerequisites
- Python 3.7 or higher
- Administrator/root privileges (required for packet capturing)
  
# Installation

1. **Clone the Repository:**

```bash
    git clone https://github.com/Simacoder/52_weeks_challenges.git
    cd Week_2

```
2. **Install Dependencies:**
   ```bash
                pip install -r requirements.txt

   ```

3. **Install Scapy:** Scapy is used for packet capturing. Ensure it is installed correctly:
```bash
        pip install scapy
```
4. **Set Up Geolocation API:**

- Sign up for an IP geolocation service like IPStack.
- Replace YOUR_API_KEY in the script with your API key.

# Usage
1.  **Run the Application:**  Execute the following command in the project directory:
```bash
    streamlit run network.py

```
2. **Access the Dashboard:** Open your browser and navigate to: 
```bash
    http://localhost:8501
    
```

# Capture Network Traffic:

- The application starts capturing packets in real-time.
- Visualizations and metrics update dynamically as packets are processed.
  
# Project Structure
```bash
    Week_2/
├── network.py   # Main application script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file

```
# Contributing
Contributions are welcome! Feel free to submit issues or pull requests

# License
This project is licensed under the MIT License. 

# AUTHOR
- Simanga Mchunu