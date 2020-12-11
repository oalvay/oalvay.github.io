.PHONY: backup
backup:
	git add *.md
	git commit -m "backup blogs"
	git push origin source
