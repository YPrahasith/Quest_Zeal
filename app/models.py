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
class Courses(db.Model):
    
    __tablename__ = 'Courses'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(20), unique=False, nullable=False)
    last_name = db.Column(db.String(20), unique=False, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    
    def __init__(self,first_name, last_name, age):
        self.first_name = first_name
        self.last_name =last_name
        self.age = age
    
    def __repr__(self):
        return f"Name : {self.first_name}, Age: {self.age}"
    
    def save(self):
        db.session.add(self)
        db.session.commit()
        
        return self

