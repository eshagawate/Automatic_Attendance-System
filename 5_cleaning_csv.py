import csv

with open('student.csv', newline='\n') as f:
    reader = csv.reader(f)
    data = list(reader)
print(data)