coverage run -m pytest  pytests/
coverage html
rm pytests/Coverage/coverage.svg
coverage-badge -o pytests/Coverage/coverage.svg