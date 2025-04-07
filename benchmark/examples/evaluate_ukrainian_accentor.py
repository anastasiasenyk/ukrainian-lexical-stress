from benchmark.benchmark import evaluate_stressification
import ukrainian_accentor as accentor # pip install ukrainian_word_stress


def shift_stress_marks(text: str, stress_mark='+'):
    """
    Examples:
    >>> shift_stress_marks("+Я спів+аю в+еселу п+існю в Укра+їні")
    'Я+ співа+ю ве+селу пі+сню в Украї+ні'
    >>> shift_stress_marks("Прив+іт")
    'Приві+т'
    >>> shift_stress_marks("Т+и біж+иш")
    'Ти+ біжи+ш'
    """
    text_list = list(text)

    i = 0
    while i <= len(text_list) - 2:
        if text_list[i] == stress_mark:
            text_list[i], text_list[i + 1] = text_list[i + 1], text_list[i]
            i += 1
        i += 1
    return ''.join(text_list)


def custom_stressify(text):
    result = accentor.process(text, mode='plus')
    return shift_stress_marks(result)


if __name__ == '__main__':
    sentence_accuracy, word_accuracy, heteronym_accuracy = evaluate_stressification(custom_stressify, stress_mark='+')

    print('Ukrainian Accentor results:')
    print(f"Sentence Accuracy: {sentence_accuracy:.2f}%")
    print(f"Word Accuracy: {word_accuracy:.2f}%")
    print(f"Heteronym Accuracy: {heteronym_accuracy:.2f}%")

    # 08.04.2025
    # Ukrainian Accentor results:
    # Sentence Accuracy: 0.16%
    # Word Accuracy: 0.80%
    # Heteronym Accuracy: 0.42%