import pandas as pd

nodes_features_path = "data/20250207NSSI(1).xlsx"
node_features_df = pd.read_excel(nodes_features_path, engine='openpyxl')
node_features = node_features_df['SF14']
count_non_zero = 0
count_zero = 0
for _, value in enumerate(node_features):
    if value == "nan":
        continue
    if value > 0:
        count_non_zero += 1
        continue
    count_zero += 1

print(count_zero)
print(count_non_zero)
