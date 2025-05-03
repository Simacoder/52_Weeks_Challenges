import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from scapy.all import sniff, IP
import threading
import logging
from sklearn.ensemble import IsolationForest
import requests

# Configuring logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Geolocation API Key (Replace with your API key)
GEO_API_KEY = "API_KEY"  #

class PacketProcessor:
    def __init__(self):
        """Initialize the PacketProcessor with an empty data list."""
        self.data = []
        self.lock = threading.Lock()

    def process_packet(self, packet):
        """Process a single packet and extract relevant information."""
        try:
            if packet.haslayer(IP):
                packet_data = {
                    'timestamp': datetime.now(),
                    'source': packet[IP].src,
                    'destination': packet[IP].dst,
                    'protocol': packet[IP].proto,
                    'size': len(packet)
                }
                with self.lock:
                    self.data.append(packet_data)
        except Exception as e:
            logger.error(f"Error processing packet: {e}")

    def get_dataframe(self):
        """Return the processed data as a Pandas DataFrame."""
        with self.lock:
            if not self.data:
                return pd.DataFrame(columns=['timestamp', 'source', 'destination', 'protocol', 'size'])
            return pd.DataFrame(self.data)

def packet_sniffer(processor: PacketProcessor, iface: str = None):
    """Capture packets and pass them to the PacketProcessor."""
    sniff(
        iface=iface,
        prn=processor.process_packet,
        store=False
    )

def train_anomaly_detector(data):
    """Train an Isolation Forest model for anomaly detection."""
    if len(data) == 0:
        return None
    features = data[['size']].fillna(0)  # Use 'size' as an example feature
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    return model

def detect_anomalies(model, data):
    """Use the model to detect anomalies."""
    if model is None or len(data) == 0:
        data['anomaly'] = 0
        return data
    features = data[['size']].fillna(0)
    data['anomaly'] = model.predict(features)
    return data

def get_geolocation(ip):
    """Fetch geolocation data for an IP address."""
    try:
        response = requests.get(f"http://api.ipstack.com/{ip}?access_key={GEO_API_KEY}")
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching geolocation: {e}")
        return None

def create_map(df):
    """Create a geographical map for IPs."""
    if df.empty:
        st.warning("No data available for mapping.")
        return
    if 'source' not in df.columns:
        st.warning("Source IP data is not available for mapping.")
        return

    locations = []
    for ip in df['source'].unique():
        geo_data = get_geolocation(ip)
        if geo_data:
            locations.append({
                'ip': ip,
                'latitude': geo_data.get('latitude', 0),
                'longitude': geo_data.get('longitude', 0),
                'country': geo_data.get('country_name', 'Unknown')
            })
    map_df = pd.DataFrame(locations)
    fig = px.scatter_geo(
        map_df,
        lat='latitude',
        lon='longitude',
        hover_name='country',
        title="Geographical Traffic Map"
    )
    st.plotly_chart(fig)

def monitor_traffic(df):
    """Check for traffic patterns and generate alerts."""
    alerts = []
    if len(df) > 0:
        # High packet count from a single IP
        top_sources = df['source'].value_counts()
        for ip, count in top_sources.items():
            if count > 100:
                alerts.append(f"High traffic from IP: {ip} ({count} packets)")

        # Excessive UDP traffic
        protocol_counts = df['protocol'].value_counts(normalize=True)
        if 'UDP' in protocol_counts and protocol_counts['UDP'] > 0.5:
            alerts.append("Excessive UDP traffic detected (>50%)")
    return alerts

def create_visualizations(df):
    """Create all dashboard visualizations."""
    if len(df) > 0:
        # Protocol distribution
        protocol_counts = df['protocol'].value_counts()
        fig_protocol = px.pie(
            values=protocol_counts.values,
            names=protocol_counts.index,
            title="Protocol Distribution"
        )
        st.plotly_chart(fig_protocol, use_container_width=True)

        # Packets timeline
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_grouped = df.groupby(df['timestamp'].dt.floor('S')).size()
        fig_timeline = px.line(
            x=df_grouped.index,
            y=df_grouped.values,
            title="Packets Over Time"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Top source IP addresses
        top_sources = df['source'].value_counts().head(10)
        fig_sources = px.bar(
            x=top_sources.index,
            y=top_sources.values,
            title="Top Source IP Addresses"
        )
        st.plotly_chart(fig_sources, use_container_width=True)

def main():
    """Main function to run the dashboard."""
    st.set_page_config(page_title="Network Traffic Analysis", layout="wide")
    st.title("Real-Time Network Traffic Analysis")

    # Initialize packet processor in session state
    if 'processor' not in st.session_state:
        st.session_state.processor = PacketProcessor()
        st.session_state.start_time = time.time()

        # Start the packet sniffer in a separate thread
        threading.Thread(
            target=packet_sniffer,
            args=(st.session_state.processor,),
            daemon=True
        ).start()

    # Get current data
    df = st.session_state.processor.get_dataframe()

    # Train anomaly detection model
    model = train_anomaly_detector(df)
    df = detect_anomalies(model, df)

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Packets", len(df))
    with col2:
        duration = time.time() - st.session_state.start_time
        st.metric("Capture Duration", f"{duration:.0f}s")

    # Create visualizations
    create_visualizations(df)

    # Display geographical map
    st.subheader("Geographical Map")
    create_map(df)

    # Monitor traffic and display alerts
    st.subheader("Alerts")
    alerts = monitor_traffic(df)
    for alert in alerts:
        st.error(alert)

    # Display recent packets
    st.subheader("Recent Packets")
    if len(df) > 0:
        st.dataframe(
            df.tail(10)[['timestamp', 'source', 'destination', 'protocol', 'size', 'anomaly']],
            use_container_width=True
        )

    # Add refresh button
    if st.button("Refresh Data"):
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()
