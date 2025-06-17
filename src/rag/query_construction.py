def construct_query(user_query, context=None, style='default'):
    if style == 'cot':
        return f"Step-by-step reasoning for: {user_query}"
    elif style == 'expand':
        return f"{user_query} and related LAMMPS commands"
    elif style == 'default':
        return user_query
    else:
        return user_query 