import os

import pandas as pd
from tqdm import tqdm

from lexical_stress_benchmark.benchmark.accuracy import DatasetAccuracy
from lexical_stress_benchmark.benchmark.sentence_stress_evaluator import evaluate_stress_sentence_level


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
    uk_stressed_dataset_path = os.path.join("..", "data", "lexical_stress_dataset.csv")
    if not os.path.exists(uk_stressed_dataset_path):
        raise FileNotFoundError(f"The file '{uk_stressed_dataset_path}' does not exist.")

    try:
        dataset = pd.read_csv(uk_stressed_dataset_path)
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")

    dataset_series = dataset["StressedSentence"]
    return dataset_series


def evaluate_stressification(
    stressify_sentence_function,
    show_progress: bool = True,
    raise_on_mismatch: bool = True,
    ignore_mismatch: bool = False,
):
    """
    Evaluates the accuracy of word stress prediction across a dataset by comparing
    the predicted sentences to the correct sentences.

    Args:
        stressify_sentence_function (function): A function that generates a sentence with '+' as the stress marks.
        show_progress (bool, optional): Whether to display a progress bar while evaluating (default is True).
        raise_on_mismatch (bool, optional): If True, raises an error when sentence lengths or structure mismatch.
        ignore_mismatch (bool, optional): If True, skips sentences with mismatched without affecting accuracy.

    Returns:
        dict: A dictionary containing the accuracy metrics.
    """
    dataset = open_dataset_for_evaluation()
    metrics = DatasetAccuracy()
    stress_mark = "+"

    iterator = tqdm(dataset, desc="Evaluating", disable=not show_progress)
    skipped = 0

    # Process each sentence
    for correct_sentence in iterator:
        sentence = correct_sentence.replace(stress_mark, "")  # Remove the stress mark for comparison
        predicted_sentence = stressify_sentence_function(sentence)

        # Evaluate word and sentence accuracy
        current_metrics = evaluate_stress_sentence_level(
            correct_sentence,
            predicted_sentence,
            raise_on_mismatch=raise_on_mismatch,
            ignore_mismatch=ignore_mismatch,
        )
        if ignore_mismatch and current_metrics is None:
            skipped += 1
            continue

        # Accumulate the accuracy
        metrics.update_with_sentence(current_metrics)

    # Calculate average accuracies
    total_sentences = len(dataset) - skipped
    if skipped:
        print(f"Warning: {skipped} sentences skipped due to a mismatch.")
    if total_sentences == 0:
        raise NoSentencesProcessedError("Warning: All sentences were skipped.")

    # Get the average accuracies
    return metrics.compute_averages(total_sentences)
