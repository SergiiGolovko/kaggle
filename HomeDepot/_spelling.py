from _spelling_dict import spell_check_dict


def spell_check(s):

    if s.lower() in spell_check_dict:
        return spell_check_dict[s]
    else:
        return s