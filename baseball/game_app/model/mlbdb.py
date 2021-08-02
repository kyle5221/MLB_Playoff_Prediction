from flask import Flask 
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.mlb_db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False   

db = SQLAlchemy(app)

class Games(db.Model):
    __tablename__ = 'games'

    game_id = db.Column(db.Integer(), nullable=False, primary_key=True)
    season = db.Column(db.Integer(), nullable=False)
    team1 = db.Column(db.String(), nullable=True)
    team2 = db.Column(db.String(), nullable=True)
    score1 = db.Column(db.Integer(), nullable=True)
    score2 = db.Column(db.Integer(), nullable=True)
    result = db.Column(db.String(), nullable=True)

    game = db.relationship('Details', backref = 'games')

    
    def __repr__(self):
        return f"Games {self.game_id}"


class Details(db.Model):
    __tablename__ = 'details'

    id = db.Column(db.Integer(), nullable=False, primary_key=True)
    elo1_pre = db.Column(db.Float(), nullable=False)
    elo2_pre = db.Column(db.Float(), nullable=True)
    elo_prob1 = db.Column(db.Float(), nullable=True)
    elo_prob2 = db.Column(db.Float(), nullable=True)
    elo1_post = db.Column(db.Float(), nullable=True)
    elo2_post = db.Column(db.Float(), nullable=True)
    rating1_pre = db.Column(db.Float(), nullable=True)
    rating2_pre = db.Column(db.Float(), nullable=True)
    pitcher1 = db.Column(db.String(), nullable=True)
    pitcher2 = db.Column(db.String(), nullable=True)
    rating_prob1 = db.Column(db.Float(), nullable=True)
    rating_prob2 = db.Column(db.Float(), nullable=True)
    rating1_post = db.Column(db.Float(), nullable=True)
    rating2_post = db.Column(db.Float(), nullable=True)
    game_id = db.Column(db.Integer(), db.ForeignKey('games.game_id'))
    


    def __repr__(self):
        return f"Details {self.id}"

db.create_all()