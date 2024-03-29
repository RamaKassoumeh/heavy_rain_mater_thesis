import matplotlib.pyplot as plt
import pandas as pd

# Example data (replace with your own)
dates = ['2024-03-01', '2024-03-03', '2024-03-06']
values = [10, 20, 30]

# Convert dates to datetime objects
dates = pd.to_datetime(dates)

# Create a DataFrame with your data
data = pd.DataFrame({'Date': dates, 'Value': values})

# Generate a continuous date range
date_range = pd.date_range(start=min(data['Date']), end=max(data['Date']), freq='5T')

# Reindex your DataFrame to include missing dates
data = data.set_index('Date').reindex(date_range).reset_index()

# Plot the data
plt.plot(data['index'], data['Value'], marker='o')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series with Gaps')
plt.grid(True)
plt.savefig("test.png")
