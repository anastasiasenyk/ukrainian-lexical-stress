def shift_stress_marks_right(text: str, stress_mark="+"):
    """
    Examples:
    >>> shift_stress_marks_right("+У вишн+евому садк+у")
    'У+ вишне+вому садку+'
    >>> shift_stress_marks_right("Прив+іт")
    'Приві+т'
    >>> shift_stress_marks_right("Т+и біж+иш")
    'Ти+ біжи+ш'
    """
    text_list = list(text)

    i = 0
    while i <= len(text_list) - 2:
        if text_list[i] == stress_mark:
            text_list[i], text_list[i + 1] = text_list[i + 1], text_list[i]
            i += 1
        i += 1
    return "".join(text_list)


def shift_stress_marks_right_left(text: str, stress_mark="+"):
    """
    Examples:
    >>> shift_stress_marks_right_left('У+ вишне+вому садку+')
    '+У вишн+евому садк+у'
    >>> shift_stress_marks_right_left('Приві+т')
    'Прив+іт'
    >>> shift_stress_marks_right_left('Ти+ біжи+ш')
    'Т+и біж+иш'
    """
    text_list = list(text)

    i = 1
    while i <= len(text_list) - 1:
        if text_list[i] == stress_mark:
            text_list[i - 1], text_list[i] = text_list[i], text_list[i - 1]
            i += 1
        i += 1
    return "".join(text_list)
