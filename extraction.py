import pyshark
import pandas as pd
import numpy as np

def load_pcap(file_path):
    return pyshark.FileCapture(file_path, display_filter="ip")


data = []

def process_packets(pcap):
    flows = {}

    for packet in pcap:
        try:
          
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                protocol = packet.highest_layer  
                timestamp = float(packet.sniff_timestamp)

                
                forward_key = (src_ip, dst_ip, protocol)
                backward_key = (dst_ip, src_ip, protocol)

                if forward_key not in flows:
                    flows[forward_key] = {
                        'timestamps': [],
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'protocol': protocol,
                        'direction': 'forward'
                    }

                if backward_key not in flows:
                    flows[backward_key] = {
                        'timestamps': [],
                        'src_ip': dst_ip,
                        'dst_ip': src_ip,
                        'protocol': protocol,
                        'direction': 'backward'
                    }

                if forward_key in flows:
                    flows[forward_key]['timestamps'].append(timestamp)
                elif backward_key in flows:
                    flows[backward_key]['timestamps'].append(timestamp)

        except AttributeError:
            
            continue

    for flow_key, flow_data in flows.items():
        timestamps = sorted(flow_data['timestamps'])
 
        inter_arrival_times = np.diff(timestamps)
        
        if flow_data['direction'] == 'forward':
            forward_iat = inter_arrival_times
            backward_iat = []
        else:
            forward_iat = []
            backward_iat = inter_arrival_times
       
        flow_duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        active_periods = []
        idle_periods = []

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff > 1:  
                idle_periods.append(time_diff)
            else:
                active_periods.append(time_diff)

      
        flow_iat = inter_arrival_times if len(inter_arrival_times) > 0 else [0]

        # Statistics calculation
        flow_stats = {
            'source_ip': flow_data['src_ip'],
            'destination_ip': flow_data['dst_ip'],
            'protocol': flow_data['protocol'],
            'flow_duration': flow_duration,
            'mean_forward_iat': np.mean(forward_iat) if len(forward_iat) > 0 else 0,
            'min_forward_iat': np.min(forward_iat) if len(forward_iat) > 0 else 0,
            'max_forward_iat': np.max(forward_iat) if len(forward_iat) > 0 else 0,
            'std_forward_iat': np.std(forward_iat) if len(forward_iat) > 0 else 0,

            'mean_backward_iat': np.mean(backward_iat) if len(backward_iat) > 0 else 0,
            'min_backward_iat': np.min(backward_iat) if len(backward_iat) > 0 else 0,
            'max_backward_iat': np.max(backward_iat) if len(backward_iat) > 0 else 0,
            'std_backward_iat': np.std(backward_iat) if len(backward_iat) > 0 else 0,

            'mean_flow_iat': np.mean(flow_iat) if len(flow_iat) > 0 else 0,
            'min_flow_iat': np.min(flow_iat) if len(flow_iat) > 0 else 0,
            'max_flow_iat': np.max(flow_iat) if len(flow_iat) > 0 else 0,
            'std_flow_iat': np.std(flow_iat) if len(flow_iat) > 0 else 0,

            'mean_active_time': np.mean(active_periods) if len(active_periods) > 0 else 0,
            'min_active_time': np.min(active_periods) if len(active_periods) > 0 else 0,
            'max_active_time': np.max(active_periods) if len(active_periods) > 0 else 0,
            'std_active_time': np.std(active_periods) if len(active_periods) > 0 else 0,

            'mean_idle_time': np.mean(idle_periods) if len(idle_periods) > 0 else 0,
            'min_idle_time': np.min(idle_periods) if len(idle_periods) > 0 else 0,
            'max_idle_time': np.max(idle_periods) if len(idle_periods) > 0 else 0,
            'std_idle_time': np.std(idle_periods) if len(idle_periods) > 0 else 0,

        }

        data.append(flow_stats)

pcap_file = "mergedoutput_138_09oct.pcap"  
packets = load_pcap(pcap_file)
process_packets(packets)


df = pd.DataFrame(data)
df.to_csv("Inference.csv", index=False)
print("Extraction complete. Data saved to Inference.csv.")
