# %load_ext cudf.pandas
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

# Define the number of rows
num_rows = 1000000

# Define the possible values
states = ["NY", "NJ", "CA", "TX"]
violations = ["Double Parking", "Expired Meter", "No Parking", "Fire Hydrant", "Bus Stop"]
vehicle_types = ["SUBN", "SDN"]

start_date = "2022-01-01"
end_date = "2022-12-31"
# Create a date range
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random data
data = {
    "Registration State": np.random.choice(states, size=num_rows),
    "Violation Description": np.random.choice(violations, size=num_rows),
    "Vehicle Body Type": np.random.choice(vehicle_types, size=num_rows),
    "Issue Date": np.random.choice(dates, size=num_rows),
    "Ticket Number": np.random.randint(1000000000, 9999999999, size=num_rows)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Adding issue weekday based on the "Issue Date"
weekday_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

df["issue_weekday"] = df["Issue Date"].dt.weekday.map(weekday_names)

# Grouping by issue_weekday and counting the Summons Number
df.groupby(["Issue Date"])["Ticket Number"
].count().sort_values()