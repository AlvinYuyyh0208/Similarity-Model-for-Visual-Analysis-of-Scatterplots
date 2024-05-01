import pandas as pd


df = pd.read_csv('/Users/avilncurry/Desktop/CsvToJs/Walmart.csv')

json_str = df.to_json(orient='records', lines=True)


json_lines = json_str.split('\n')
json_lines_with_commas = [line + ',' for line in json_lines if line.strip()]

final_json_str = '[\n' + '\n'.join(json_lines_with_commas).rstrip(',') + '\n]'

with open('../dv-proj/plot/data8.js', 'w') as json_file:
    json_file.write(final_json_str)
