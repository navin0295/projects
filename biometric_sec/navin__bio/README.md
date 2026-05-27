# Fingerprint Biometric Cryptosystem

## What this project does
- Enrolls fingerprint users into a local encrypted database.
- Verifies fingerprints against enrolled templates.
- Supports bulk enrollment, evaluation, training, a Tkinter GUI, and a self-contained demo.

## Requirements
- Python 3.10+ recommended.
- `pip` for installing dependencies.
- A webcam is only needed for camera-based GUI features.

## Install
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare the dataset
- Place the SOCOFing dataset inside `data/SOCOFing/Real`.
- The bulk enrollment and evaluation commands expect `.BMP` images in that folder.
- If you want to use a different location, update `DATA_DIR` in `config.py`.

Example layout:

```text
navin__bio/
├─ data/
│  └─ SOCOFing/
│     └─ Real/
│        ├─ 1__M_Left_index_finger.BMP
│        ├─ 1__F_Right_thumb.BMP
│        └─ ...
```

## Run the CLI
Show all commands:

```bash
python main.py --help
```

### Enroll a user
```bash
python main.py enroll --user alice --images path/to/img1.bmp path/to/img2.bmp path/to/img3.bmp
```

### Verify a user
```bash
python main.py verify --user alice --image path/to/query.bmp
```

### List users
```bash
python main.py list-users
```

### View one user
```bash
python main.py user-info --user alice
```

### Delete a user
```bash
python main.py delete --user alice
```

### Bulk enroll from SOCOFing
```bash
python main.py bulk-enroll --dir ./data/SOCOFing/Real --subjects 10 --images-per 3
```

### Evaluate the system
```bash
python main.py evaluate --subjects 5 --genuine 20 --impostor 20
```

### Train the CNN
```bash
python main.py train --dir ./data/SOCOFing/Real --epochs 10
```

### Export / import the database
```bash
python main.py export --out backup.json
python main.py import-db --file backup.json
```

## Run the GUI
Launch the desktop interface:

```bash
python gui.py
```

The GUI includes enrollment, verification, user management, and training tabs.

## Run the demo
The demo works without SOCOFing data:

```bash
python demo.py
```


## Troubleshooting
- If a command says the dataset is missing, confirm `data/SOCOFing/Real` exists.
- If `tensorflow` or `opencv` fails to install, upgrade `pip` first and retry.
- If the GUI does not open, make sure your environment supports desktop apps.