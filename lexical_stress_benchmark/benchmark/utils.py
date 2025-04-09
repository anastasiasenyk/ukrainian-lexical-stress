def shift_stress_marks(text: str, stress_mark='+'):
    """
    Examples:
    >>> shift_stress_marks("+У вишн+евому садк+у")
    'У+ вишне+вому садку+'
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