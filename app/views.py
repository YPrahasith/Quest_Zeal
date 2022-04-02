# Python modules
import random
import os

# Flask modules
from flask import render_template, request, url_for, redirect, send_from_directory
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
# App modules
from app        import app, lm, db, bc
from app.models import Users
from app.forms  import LoginForm, RegisterForm
from app.QA_Gen_Model import QA_Gen_Model
from app.Objective_QA_Gen_Model import MCQ_Generator
# model = pickle.load(open('app/model.pkl','rb'))
UPLOAD_FOLDER = 'Uploaded Material'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# Logout user
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# Register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    
    # declare the Registration Form
    form = RegisterForm(request.form)

    msg     = None
    success = False

    if request.method == 'GET': 

        return render_template( 'register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        # filter User out of database through username
        user = Users.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = Users.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'
        
        else:         

            pw_hash = bc.generate_password_hash(password)

            user = Users(username, email, pw_hash)

            user.save()

            msg     = 'User created, please <a href="' + url_for('login') + '">login</a>'     
            success = True

    else:
        msg = 'Input error'     

    return render_template( 'register.html', form=form, msg=msg, success=success )

# Authenticate user
@app.route('/login', methods=['GET', 'POST'])
def login():
    
    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = Users.query.filter_by(user=username).first()

        if user:
            
            if bc.check_password_hash(user.password, password):
                login_user(user)
                if(username =="admin"):
                    return redirect(url_for('tutorDashboard'))
                else :
                    return redirect(url_for('studentDashboard'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unknown user"

    return render_template( 'login.html', form=form, msg=msg )

# App main route + generic routing
@app.route('/')
def index():

    try:

        return render_template( 'index.html' )
    
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
#Render Student Dashboard
@app.route('/studentDashboard')
@login_required
def studentDashboard():
    if current_user.user == "admin":
        return render_template('page-404.html'), 404
    else :
        return render_template( 'studentDashboard.html',name = current_user.user , email= current_user.email)

#Render Tutor Dashboard
@app.route('/tutorDashboard', methods=['GET', 'POST'])
@login_required
def tutorDashboard():
    name = current_user.user
    if name=="admin" :
        if request.method == 'POST':
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename)))
               
            return redirect(url_for('tutorDashboard'))

        return render_template( 'tutorDashboard.html',name = name , email= current_user.email)
    else :
        return render_template('page-404.html'), 404


@app.route('/Question_Generation')
@login_required
def Question_Generation():
    name = current_user.user
    if name!="admin" :
        try:
            with open('Uploaded Material/dbms.txt', 'r') as f:
                content = f.read()
        except:
            return render_template('page-404.html'), 404
        
        que, ans = QA_Gen_Model.generate_test(content)
        size = len(que)
        return render_template('Question_Generation.html', content = content, question = que, answer = ans, size = size )

#Objective Question Generation
@app.route('/Objective_QA_Generation')
@login_required 
def Objective_QA_Generation():
    name = current_user.user
    if name!="admin" :
        try:
            with open('Uploaded Material/dbms.txt', 'r') as f:
                content = f.read()
        except:
            return render_template('page-404.html'), 404
        
        Objective_Questions = MCQ_Generator.generate_mcq_questions(content, 10)
        i = 1
        for questions in Objective_Questions:
            print(i)
            print(questions.questionText)
            questions.distractors.append(questions.answerText)
            print(questions.answerText)
            random.shuffle(questions.distractors)
            print(questions.distractors)
            i+=1
            
        return render_template('Objective_Questions.html', Objective_Questions = Objective_Questions)

# Return sitemap
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')