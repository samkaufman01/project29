import csv

if __name__ == "__main__":

    with open("AllDescriptors.csv", "rb") as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # Skip the header row
        collected = []
        for row in reader:
            collected.append(row[0])
            print("\n".join(collected))
            
import pandas as pd

d = pd.read_csv("AllDescriptors.csv",header = 0, sep = ',')
print(list(d.columns.values))
print(d['MPID'].tolist() + ' ')
