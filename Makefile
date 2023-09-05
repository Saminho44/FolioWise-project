reinstall_package:
	@pip uninstall -y FolioWise || :
	@pip install -e .


run_api:
	uvicorn FolioWise.api.fast:app --reload
