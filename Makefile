.PHONY: selftest test analysis replication figures checksums
selftest:
	python src/prometrics_network_analysis.py --selftest
test:
	pytest -q
analysis:
	python src/prometrics_network_analysis.py --out results/integrated
replication:
	python src/prometrics_network_analysis.py --data data/bibfusion_output/scopus --out results/scopus_replication
figures:
	python src/make_figures.py --out figures
checksums:
	sha256sum -c data/SHA256SUMS
