import os
import pandas as pd
from tqdm import tqdm

from .sentence_stressification import evaluate_stress_sentence_level
from .accuracy import AccuracyMetrics

class NoSentencesProcessedError(Exception):
    """Custom error to indicate that no sentences were processed."""
    pass

def open_dataset_for_evaluation():
    """
    Opens the dataset for evaluation by reading a CSV file that contains stressed sentences.

    The function checks whether the file exists and raises an error if it does not.
    The dataset is filtered to include only rows where the 'Done' column is True.

    Returns:
        pd.Series: A Pandas Series containing the stressed sentences from the dataset.
    """
    uk_wiki_stressed_path = os.path.join('..', 'data', 'plug', 'plug_from_excel.csv')
    if not os.path.exists(uk_wiki_stressed_path):
        raise FileNotFoundError(f"The file '{uk_wiki_stressed_path}' does not exist.")

    try:
        dataset = pd.read_csv(uk_wiki_stressed_path, sep=';')
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")

    dataset = dataset[dataset['Done'] == True]
    dataset_series = dataset['StressedSentence']
    return dataset_series

def evaluate_stressification(stressify_sentence_function, stress_mark: str = '+',
                             show_progress: bool = True):
    """
    Evaluates the accuracy of stressification on a dataset by comparing the
    stressified sentences to the correct sentences.

    Args:
    - dataset (pd.Series): A pandas series of correct sentences with stress marks.
    - stressify_sentence_function (function): A function that stressifies a sentence.
    - stress_mark (str): The symbol used for stress marks in the correct sentences (default is '+').
    - show_progress (bool): Whether to display a progress bar (default is True).

    Returns:
    - sentence_accuracy (float): The sentence accuracy of stressification.
    - word_accuracy (float): The accuracy of word stressification.
    - heteronym_accuracy (float): The accuracy of heteronym stressification.
    """
    dataset = open_dataset_for_evaluation()
    metrics = AccuracyMetrics()

    iterator = tqdm(dataset, desc="Evaluating stressification", disable=not show_progress)

    # Process each sentence
    for correct_sentence in iterator:
        sentence = correct_sentence.replace(stress_mark, '')  # Remove the stress mark for comparison
        stressified_sentence = stressify_sentence_function(sentence)

        # Evaluate word and sentence accuracy
        current_metrics = evaluate_stress_sentence_level(correct_sentence, stressified_sentence)

        # Accumulate the accuracy
        metrics.accumulate(current_metrics)

    # Calculate average accuracies
    total_sentences = len(dataset)
    if total_sentences == 0:
        raise NoSentencesProcessedError("Warning: All sentences were skipped.")

    # Get the average accuracies
    return metrics.average_accuracies(total_sentences)

