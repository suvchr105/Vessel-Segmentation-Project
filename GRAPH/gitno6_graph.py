import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('ti.csv')

# Extract the data from the DataFrame
epochs = data['Epoch']
accuracy = data['Accuracy']
precision = data['Precision']
recall = data['Recall']
f1 = data['F1']

# Plot the graph
plt.plot(epochs, accuracy, marker='o', label='Accuracy')
plt.plot(epochs, precision, marker='o', label='Precision')
plt.plot(epochs, recall, marker='o', label='Recall')
plt.plot(epochs, f1, marker='o', label='F1')

# Set the title and labels for the axes
plt.title('Evaluation Metrics vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Metrics')

# Add a legend
plt.legend()

# Show the plot
plt.show()
