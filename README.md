# IdentifyMe


IdentifyMe is a Python project that uses DeepFace for face and gender detection in images. The app features a modern UI and allows users to select a picture, then identify if a person is present and their gender.

## Project Structure

- `models/` - Data models (e.g., User)
- `viewmodels/` - Logic and data manipulation for views
- `views/` - User interface and display logic
- `tests/` - Unit tests for the project
- `IdentifyMe.py` - Main entry point (if needed)

## Getting Started

1. Clone the repository:
	```sh
	git clone https://github.com/bholsinger09/IdentifyMe.git
	cd IdentifyMe
	```
2. (Optional) Create and activate a virtual environment:
	```sh
	python3 -m venv venv
	source venv/bin/activate
	```

3. Install dependencies:
	```sh
	pip install deepface ttkbootstrap pillow
	# For Apple Silicon, you may need:
	# pip install tensorflow-macos
	# For Intel/Windows/Linux:
	# pip install tensorflow
	```
4. Run the app:
	```sh
	python views/user_view.py
	```
5. Run tests:
	```sh
	python -m unittest discover tests
	```


## Features
- Select an image and display it in the app
- Identify if a person is present in the image
- Detect gender using DeepFace

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)