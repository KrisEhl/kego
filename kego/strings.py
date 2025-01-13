def break_lines(string, max_length=100):
    all_lines = ""
    current_line = ""
    for word in string.split(" "):
        if len(current_line + " " + word) < max_length:
            current_line += " " + word
        else:
            all_lines += "\n" + current_line
            current_line = word
    all_lines += "\n" + current_line
    all_lines = all_lines.strip("\n ")
    return all_lines
