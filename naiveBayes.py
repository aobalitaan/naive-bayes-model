import math
import numpy as np


THRESHOLD = 0.005    # Negligible difference threshold (0.5% error (new-old)/old)

def compute_frequencies(varList, dataset):
    freq = {}

    for var in varList:     # For each variable in varlist
        freq[var] = {}

        for record in dataset:      # For each answer in the variable
            if var in record:
                value = record[var]
                freq[var][value] = freq[var].get(value, 0) + 1 # Counts them

    return freq

def getMetrics(evaluation):
    TP = FP = TN = FN = 0

    for pred, true in evaluation:
        if pred == '1' and true == '1':
            TP += 1
        elif pred == '1' and true == '0':
            FP += 1
        elif pred == '0' and true == '0':
            TN += 1
        elif pred == '0' and true == '1':
            FN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': [TP, FP, TN, FN]
    }

def naiveBayes(varList, values, testing, target, k):
    evaluation = []

    for dataset in [testing['wtarget'], testing['wotarget']]:   # Simply, loops through test set
        for test in dataset:
            # Gets the unnormalized probability that test == 1
            temp1 = [
                (values['P(x|target)'][var].get(test[var], 0) + k) / (len(testing['wtarget']) + k * len(values['P(x|target)'][var])) for var in varList
            ]

            
            v_target = values['P(target)'] * np.prod(temp1)

            # Gets the unnormalized probability that test == 0
            temp2 = [
                (values['P(x|~target)'][var].get(test[var], 0) + k) / (len(testing['wotarget']) + k * len(values['P(x|~target)'][var])) for var in varList
            ]
            v_nottarget = values['P(~target)'] * np.prod(temp2)

            # Normalize Probabilities
            p_target = v_target / (v_target + v_nottarget)
            p_nottarget = v_nottarget / (v_target + v_nottarget)

            # Appends (label, real)
            label = '1' if p_target > p_nottarget else '0'
            evaluation.append((label, test[target]))

    # Gets metrics
    metrics = getMetrics(evaluation)
    return metrics

def backwardElimination(varList, values, testing, target, k, metric, eliminated_variables=None):
    if eliminated_variables is None:
        eliminated_variables = []

    current_metrics = naiveBayes(varList, values, testing, target, k)   # Baseline metrics

    metric_list = []

    for i in range(len(varList)):       # For each variable in variable list
        varList_copy = varList.copy()       # Removes one item from the current variable list
        varList_copy.pop(i)

        child_metrics = naiveBayes(varList_copy, values, testing, target, k)    # Metrics after removing the variable
        metric_list.append(child_metrics[metric] - current_metrics[metric])

    candidate = -1 if len(metric_list) == 0 else max(metric_list)       # Gets the best effect on the metric

    if candidate > 0 or (candidate / current_metrics[metric] > -THRESHOLD):     # If (positive effect) or (negligible difference)
        i_candidate = metric_list.index(candidate)      # Index of the variable to remove
        varList_copy = varList.copy()

        eliminated_variable = varList_copy[i_candidate]
        eliminated_variables.append((eliminated_variable, candidate))  # Track variable with accuracy increase
        print(f"[{candidate:.4f}] Removed: {eliminated_variable}")
        varList_copy.pop(i_candidate)

        result_metrics, result_vars, eliminated_variables = backwardElimination(
            varList_copy, values, testing, target, k, metric, eliminated_variables
        )     # Recursive call with the reduced variable set
        return result_metrics, result_vars, eliminated_variables      # Return resulting metrics and variable list

    return current_metrics, varList, eliminated_variables     # if no variable was eliminated anymore

def write_output(k, metrics, variables, eliminated_variables):
    with open('output.txt', 'w') as file:
        file.write(f"Best k-value: {k}\n")
        file.write("\nBest Metrics:\n")
        max_metric_length = max(len(metric) for metric in metrics)
        for metric, value in metrics.items():
            file.write(f"- {metric.ljust(max_metric_length)}: {value}\n")
        file.write("\nBest variables after elimination:\n")
        for variable in variables:
            file.write(f"- {variable}\n")
        file.write("\nEliminated variables with accuracy increase:\n")
        for variable, accuracy_increase in eliminated_variables:
            file.write(f"- {variable}: {accuracy_increase:.4f}\n")
        file.write("====================\n")

def main(varList, training, testing, target, metric):
    k_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]     # k-values to test
    
    print(len(training))
    print(len(testing))
    print(training)
    print(testing)
    
    best_k = None
    best_metrics = None
    best_varList = None
    best_eliminated = None
    best_metric = -math.inf

    for k in k_values:
        print(f"\n\n===== K = {k} =====")
        # Initializes training values (bag-of-words probabilities)
        values = {
            'P(target)': len(training['wtarget']) / (len(training['wtarget']) + len(training['wotarget'])),
            'P(~target)': len(training['wotarget']) / (len(training['wtarget']) + len(training['wotarget'])),
            'P(x|target)': compute_frequencies(varList, training['wtarget']),
            'P(x|~target)': compute_frequencies(varList, training['wotarget'])
        }

        # Perform backward elimination
        metrics, varList_eliminated, eliminated_variables = backwardElimination(varList, values, testing, target, k, metric)

        # Check for the best k-value based on the metric
        if metrics[metric] > best_metric:
            best_k = k
            best_metrics = metrics
            best_varList = varList_eliminated
            best_eliminated = eliminated_variables
            best_metric = metrics[metric]
    
    print(f"\n\nBest k-value: {best_k}")
    print(f"Best Metrics ({metric}):", best_metrics)
    print("Best Variable List after Elimination:", best_varList)
    print("Eliminated Variables with Accuracy Increase:", best_eliminated)
    write_output(best_k, best_metrics, best_varList, best_eliminated)