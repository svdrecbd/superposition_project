import ast
import re

def validate_code(code_string: str) -> bool:
    """
    Cheaply validates a string of Python code using ast.parse.
    Returns True if syntax is valid, False otherwise.
    """
    try:
        # Extract code from markdown fences if present
        match = re.search(r"```(python)?\n(.*?)```", code_string, re.DOTALL)
        if match:
            code_string = match.group(2)
        ast.parse(code_string)
        return True
    except (SyntaxError, ValueError):
        return False

def validate_latex(s: str) -> bool:
    """
    Cheaply validates a string of LaTeX for balanced delimiters and align blocks.
    """
    if s.count("$") % 2 != 0: return False
    if s.count("{") != s.count("}"): return False
    if s.count("\\begin") != s.count("\\end"): return False
    # Check for & in align blocks
    if re.search(r"\\(begin|end)\s*{\s*align\*?\s*}", s):
        align_content = re.findall(r"\\begin{align\*?}(.*?)\\end{align\*?}", s, re.DOTALL)
        for block in align_content:
            if "&" not in block: return False
    return True