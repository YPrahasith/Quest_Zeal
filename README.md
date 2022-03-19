# Quest Zeal
An AI Based Question Answering System!

## Running Application

> **Step #1** - Clone sources (this repo)

```bash
$ git clone https://github.com/YPrahasith/Quest_Zeal.git
$ cd Quest_Zeal
```

<br />

> **Step #2** - Create a virtual environment

```bash
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ virtualenv env
$ .\env\Scripts\activate
```

<br />

> **Step #3** - Install dependencies

```bash
$ pip install -r requirements.txt
```

<br />

> **Step #4** - Set Up Environment

```bash
$ # Set the FLASK_APP environment variable
$ (Unix/Mac) export FLASK_APP=run.py
$ (Windows) set FLASK_APP=run.py
$ (Powershell) $env:FLASK_APP = ".\run.py"
```

<br />

> **Step #5** - Create Tables (SQLite persistance)

```bash
$ # Create tables
$ flask shell
$ >>> from app import db
$ >>> db.create_all()
```

<br />

<br />

> **Step #7** - Start the project

```bash
$ flask run
$
$ # Access the app in browser: http://127.0.0.1:5000/
```

<br />

## Code-base structure

The project has a super simple structure, represented as bellow:

```bash
< PROJECT ROOT >
   |
   |-- app/
   |    |-- static/
   |    |    |-- <css, JS, images>    # CSS files, Javascripts files
   |    |
   |    |-- templates/
   |    |    |
   |    |    |-- index.html           # Index File
   |    |    |-- login.html           # Login Page
   |    |    |-- register.html        # Registration Page
   |    |    
   |    |
   |   config.py                      # Provides APP Configuration 
   |   forms.py                       # Defines Forms (login, register) 
   |   models.py                      # Defines app models 
   |   views.py                       # Application Routes 
   |
   |-- requirements.txt
   |-- run.py
   |
   |-- ************************************************************************
```

<br />
