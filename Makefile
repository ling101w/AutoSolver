PYTHON ?= python
INPUT ?= instance/large_seed301-1.txt

.PHONY: run run-large run-problem score score-large score-problem

run:
	@$(PYTHON) -c "from pathlib import Path; from solver import solve; print(solve(Path('$(INPUT)').read_text()))"

run-large:
	@$(MAKE) run INPUT=instance/large_seed301-1.txt

run-problem:
	@$(MAKE) run INPUT=problem.txt

score:
	@$(PYTHON) score_solution.py "$(INPUT)"

score-large:
	@$(MAKE) score INPUT=instance/large_seed301-1.txt

score-problem:
	@$(MAKE) score INPUT=problem.txt
