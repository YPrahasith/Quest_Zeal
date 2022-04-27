
from enum import unique
from app         import db
from flask_login import UserMixin

class Users(db.Model, UserMixin):

    __tablename__ = 'Users'

    id       = db.Column(db.Integer,     primary_key=True)
    user     = db.Column(db.String(64),  unique = True)
    email    = db.Column(db.String(120), unique = True)
    password = db.Column(db.String(500))

    def __init__(self, user, email, password):
        self.user       = user
        self.password   = password
        self.email      = email

    def __repr__(self):
        return str(self.id) + ' - ' + str(self.user)

    def save(self):

        # inject self into db session    
        db.session.add ( self )

        # commit change and save the object
        db.session.commit( )

        return self 

class Tests(db.Model):
    
    __tablename__ = 'Tests'
    
    id = db.Column(db.Integer, primary_key=True)
    course_name = db.Column(db.String(20), unique=False, nullable=False)
    test_name = db.Column(db.String(20), unique=False, nullable=False)
    file_name = db.Column(db.String(20), unique=True, nullable=False)
    
    def __init__(self,course_name,test_name,file_name):
        self.course_name = course_name
        self.test_name =  test_name
        self.file_name = file_name
    
    def __repr__(self):
        return f"Name : {self.course_name}, File name: {self.file_name}"
    
    def save(self):
        db.session.add(self)
        db.session.commit()
        
        return self

class Students(db.Model):
    
    __tablename__ = 'Students'
    
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, unique=False, nullable = False)
    course_name = db.Column(db.String(20), unique=False, nullable=False)
    test_name = db.Column(db.String(20), unique=False, nullable=False)
    type = db.Column(db.String(20),unique=False,nullable=False)
    score = db.Column(db.String(10),unique=False, nullable=False)
    
    def __init__(self,course_id,course_name,test_name,type,score):
        self.course_id = course_id
        self.course_name = course_name
        self.test_name =  test_name
        self.type = type
        self.score = score
    
    def __repr__(self):
        return f"id : {self.id}, Score: {self.score}"
    
    def save(self):
        db.session.add(self)
        db.session.commit()
        
        return self