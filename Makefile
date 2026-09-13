PYTHON ?= python3

.PHONY: setup verify analysis paper-figures figure-check reproduce privacy

setup:
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) experiments/one_step_panel/verify.py

analysis:
	$(PYTHON) experiments/one_step_panel/run.py

paper-figures:
	$(PYTHON) paper_figures/render_count_figures.py
	$(PYTHON) paper_figures/render_approximate_figure.py
	$(PYTHON) paper_figures/render_authorization_figures.py
	$(PYTHON) paper_figures/render_topic_calibration.py

figure-check: paper-figures
	$(PYTHON) scripts/compare_figures.py

privacy:
	$(PYTHON) scripts/check_public_release.py

reproduce: analysis verify figure-check privacy
