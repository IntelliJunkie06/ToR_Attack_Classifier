# Importing necessary libraries for initial data inspection
import pandas as pd

# Loading the CSV dataset uploaded by the user
file_path = 'test.csv'
df = pd.read_csv(file_path)

# Displaying the first few rows and column information for analysis
# df_head = df.head()
# df_info = df.info()

# df_head, df_info
# Adding labels based on heuristic rules for DDoS, ToR, and Normal traffic
def label_traffic(row):
    # Define basic thresholds for DDoS based on empirical flow characteristics
    if row['protocol'] in ['ICMP', 'TCP'] and row['mean_forward_iat'] < 50 and row['flow_duration'] < 10000:
        return 'DDoS Traffic'
    elif row['protocol'] in ['DATA', 'TCP'] and (100 < row['mean_forward_iat'] < 300) and (100 < row['mean_backward_iat'] < 300):
        return 'ToR Traffic'
    else:
        return 'Normal Traffic'

# Applying the labeling function to the DataFrame
df['Traffic_Label'] = df.apply(label_traffic, axis=1)
df.to_csv('Inference ToR Data.csv')

# # Displaying a sample of the labeled data
# df[['source_ip', 'destination_ip', 'protocol', 'flow_duration', 'mean_forward_iat', 'mean_backward_iat', 'Traffic_Label']].head(10)
