# Coding Instructions

## General Guidelines

- Write all code in Python unless otherwise specified.
- Generate complete, working code. If a full solution is not possible, provide detailed comments explaining missing parts.
- Use double quotes for all string literals.
- Use string templates (f-strings) or the `.join()` method for string concatenation, not the `+` operator.
- Include clear inline comments and section headers to describe each step of the code.
- Implement error checking and type validation for all user inputs and function arguments.
- For TypeScript code:
  - Use strict typing.
  - Define new types as necessary.
  - Do not use the `any` type.
  - Do not use the non-null assertion operator (`!`).
  - Do not cast to unknown (e.g., `as unknown as T`).

## Python-Specific Instructions

- Use type hints for all function arguments and return types.
- Raise informative exceptions for invalid input.
- Use context managers (`with` statements) for file and resource handling.
- Prefer list comprehensions and generator expressions for concise and efficient code.
- Use logging instead of print statements for status and error messages.
- Write modular code: break large functions into smaller, reusable ones.
- Include docstrings for all public functions and classes.

## Documentation

- All public functions, classes, and modules must have docstrings explaining their purpose, arguments, return values, and exceptions raised.
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

## Version Control

- Commit messages should be clear and descriptive.

---

For any questions or clarifications, please refer to this file or contact the repository maintainers.
