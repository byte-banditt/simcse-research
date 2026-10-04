# SimCSE reproduction

DRAFT - NEEDS HUMAN VERIFICATION

- [paper_repro](paper_repro/)
- [Results index](RESULTS_INDEX.md)
- [Questions for human](QUESTIONS_FOR_HUMAN.md)
- [Interview explainers](explainers/)

Existing result files retain their original contents and historical commit hashes. Shared datasets and model caches remain outside these projects. No test-set tuning.

Run commands from this project root using `.venv/bin/python`. The environment currently links to the existing CVE virtual environment; library pins are in the study requirements lock. Historical checkpoints are resolved by the study project_paths.py helper without changing saved results. Historical run commits are retained under refs/remotes/history/.
