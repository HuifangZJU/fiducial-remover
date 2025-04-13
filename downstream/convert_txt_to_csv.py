import pandas as pd

# Read the .txt file (assuming it’s tab-delimited or space-delimited)
txt_file ='/media/huifang/data/fiducial/tiff_data/151672/spatial/tissue_positions_list.txt'
df = pd.read_csv(txt_file, delimiter=',')  # Change delimiter if needed (e.g., ',' or ' ')

# Save it as a .csv file
df.to_csv(txt_file[:-4]+'.csv', index=False)
