import numpy as np
import pandas as pd

def find_s_algorithm(data):
    # Initialize the hypothesis with the first positive example
    positive_examples = data[data.iloc[:, -1] == "No"]

    if len(positive_examples) == 0:
        print("No positive examples found.")
        return None

    hypothesis = positive_examples.iloc[0, :-1].copy()

    # Iterate through the positive examples to find the most specific hypothesis
    for i in range(1, len(positive_examples)):
        for j in range(len(hypothesis)):
            if positive_examples.iloc[i, j] != hypothesis[j]:
                hypothesis[j] = '?'

    return hypothesis

# Example usage with a dataset
data = pd.read_csv("FindS_datasetByGaurav.csv")
hypothesis = find_s_algorithm(data)

if hypothesis is not None:
    print("The final hypothesis is:", hypothesis)
