# =============================================================
#   AgroMind AI – Smart Agriculture Crop Prediction App
#   Single File Flask Application
#   Install: pip install flask flask-sqlalchemy scikit-learn numpy werkzeug
#   Run:     python agromind_app.py
#   Admin:   Email: admin@jacob.ai | Password: Admin@1234
# =============================================================

from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import pickle, os

# ─────────────────────────────────────────────
#   APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'agromind_secret_2024_jacob_ai'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agromind.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ADMIN_EMAIL    = 'admin@jacob.ai'
ADMIN_PASSWORD = 'Admin@1234'

# ─────────────────────────────────────────────
#   DATABASE MODELS
# ─────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    region        = db.Column(db.String(100))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    nitrogen       = db.Column(db.Float)
    phosphorus     = db.Column(db.Float)
    potassium      = db.Column(db.Float)
    temperature    = db.Column(db.Float)
    humidity       = db.Column(db.Float)
    ph             = db.Column(db.Float)
    rainfall       = db.Column(db.Float)
    predicted_crop = db.Column(db.String(100))
    confidence     = db.Column(db.Float)
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────────
#   ML MODEL
# ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

TRAINING_DATA = [
    [90,42,43,20.8,82,6.5,202,'rice'],[85,58,41,21.7,80,7.0,226,'rice'],[60,55,44,23.0,82,6.8,230,'rice'],
    [77,48,20,22.6,65,6.2,82,'maize'],[65,37,25,27.4,63,6.5,73,'maize'],[85,58,22,26.0,62,7.2,80,'maize'],
    [40,67,19,17.4,16,7.3,80,'chickpea'],[37,65,22,18.0,18,7.8,71,'chickpea'],[50,70,20,16.5,17,7.5,68,'chickpea'],
    [20,67,20,18.8,22,5.7,105,'kidneybeans'],[40,60,20,19.2,18,5.9,115,'kidneybeans'],
    [20,67,20,27.7,48,5.7,149,'pigeonpeas'],[25,62,18,27.0,50,6.0,145,'pigeonpeas'],
    [20,27,14,28.2,52,6.9,51,'mothbeans'],[22,30,16,30.0,55,7.0,48,'mothbeans'],
    [20,47,20,28.5,85,6.4,48,'mungbean'],[18,44,22,29.0,83,6.6,50,'mungbean'],
    [40,67,19,29.9,64,7.0,67,'blackgram'],[45,65,20,30.5,62,7.2,65,'blackgram'],
    [18,68,19,24.5,64,6.9,45,'lentil'],[20,65,22,25.0,62,7.1,42,'lentil'],
    [18,18,20,21.8,90,6.4,107,'pomegranate'],[20,20,18,22.0,88,6.6,110,'pomegranate'],
    [100,82,50,27.4,81,5.9,105,'banana'],[95,80,48,26.5,82,6.0,100,'banana'],
    [20,27,30,30.8,50,5.8,94,'mango'],[22,25,28,32.0,52,6.0,90,'mango'],
    [23,132,200,23.8,81,6.0,69,'grapes'],[25,130,195,24.0,80,6.2,72,'grapes'],
    [99,59,50,25.6,85,6.5,51,'watermelon'],[100,55,48,26.0,83,6.8,55,'watermelon'],
    [100,17,50,28.7,92,6.4,24,'muskmelon'],[98,20,52,29.0,90,6.6,22,'muskmelon'],
    [20,134,199,22.6,92,5.9,113,'apple'],[22,130,195,23.0,91,6.1,110,'apple'],
    [20,16,10,22.8,92,7.0,110,'orange'],[18,15,12,23.0,90,7.2,115,'orange'],
    [49,59,50,33.0,92,7.0,142,'papaya'],[50,60,48,34.0,90,7.2,145,'papaya'],
    [22,16,30,26.9,94,5.9,176,'coconut'],[20,18,32,27.0,92,6.0,180,'coconut'],
    [118,46,20,23.9,79,6.9,80,'cotton'],[115,44,22,24.0,80,7.0,82,'cotton'],
    [78,46,44,24.9,79,6.7,174,'jute'],[80,44,42,25.0,80,6.9,170,'jute'],
    [101,28,29,25.5,58,6.8,158,'coffee'],[100,30,30,26.0,60,7.0,160,'coffee'],
    [103,40,36,21.9,82,6.5,55,'wheat'],[100,42,38,22.0,80,6.8,58,'wheat'],
    [20,72,42,25.3,65,6.5,48,'sunflower'],[22,70,40,26.0,63,6.8,50,'sunflower'],
]

CROP_META = {
    'rice':        {'emoji':'🌾','season':'Kharif','profit':'High','tip':'Needs flooded fields & high humidity'},
    'maize':       {'emoji':'🌽','season':'Kharif/Rabi','profit':'Medium','tip':'Versatile, good for semi-dry areas'},
    'chickpea':    {'emoji':'🫘','season':'Rabi','profit':'High','tip':'Excellent for nitrogen fixation'},
    'kidneybeans': {'emoji':'🫘','season':'Kharif','profit':'Medium','tip':'Rich in protein, high export value'},
    'pigeonpeas':  {'emoji':'🌿','season':'Kharif','profit':'Medium','tip':'Very drought tolerant'},
    'mothbeans':   {'emoji':'🌿','season':'Kharif','profit':'Low','tip':'Extremely drought resistant'},
    'mungbean':    {'emoji':'🌱','season':'Kharif','profit':'Medium','tip':'Short duration crop'},
    'blackgram':   {'emoji':'🫘','season':'Kharif/Rabi','profit':'Medium','tip':'High demand in food industry'},
    'lentil':      {'emoji':'🫘','season':'Rabi','profit':'High','tip':'High nutrition, good market price'},
    'pomegranate': {'emoji':'🍎','season':'Perennial','profit':'Very High','tip':'High export demand, drought tolerant'},
    'banana':      {'emoji':'🍌','season':'Perennial','profit':'High','tip':'Year-round income source'},
    'mango':       {'emoji':'🥭','season':'Summer','profit':'Very High','tip':'King of fruits, high export value'},
    'grapes':      {'emoji':'🍇','season':'Rabi','profit':'Very High','tip':'Premium wine and table grapes'},
    'watermelon':  {'emoji':'🍉','season':'Summer','profit':'High','tip':'Fast growing, high yield'},
    'muskmelon':   {'emoji':'🍈','season':'Summer','profit':'Medium','tip':'Good for warm dry regions'},
    'apple':       {'emoji':'🍎','season':'Winter','profit':'Very High','tip':'High altitude, premium market'},
    'orange':      {'emoji':'🍊','season':'Winter','profit':'High','tip':'Juice industry demand rising'},
    'papaya':      {'emoji':'🍈','season':'Perennial','profit':'High','tip':'Pharmaceutical use, high demand'},
    'coconut':     {'emoji':'🥥','season':'Perennial','profit':'High','tip':'Multi-use crop, coastal regions'},
    'cotton':      {'emoji':'🌸','season':'Kharif','profit':'High','tip':'Cash crop for textile industry'},
    'jute':        {'emoji':'🌿','season':'Kharif','profit':'Medium','tip':'Eco-friendly fiber crop'},
    'coffee':      {'emoji':'☕','season':'Perennial','profit':'Very High','tip':'Premium export price'},
    'wheat':       {'emoji':'🌾','season':'Rabi','profit':'High','tip':'Staple food, government MSP support'},
    'sunflower':   {'emoji':'🌻','season':'Kharif/Rabi','profit':'Medium','tip':'Oilseed, good for dry areas'},
}

MODEL_PATH = 'agromind_model.pkl'

def train_model():
    X = np.array([r[:7] for r in TRAINING_DATA])
    y = np.array([r[7]  for r in TRAINING_DATA])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X, y_enc)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((clf, le), f)
    return clf, le

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return train_model()

def predict_crop(N, P, K, temp, hum, ph, rain):
    clf, le = load_model()
    feat   = np.array([[N, P, K, temp, hum, ph, rain]])
    pred   = clf.predict(feat)[0]
    proba  = clf.predict_proba(feat)[0]
    conf   = round(max(proba) * 100, 1)
    crop   = le.inverse_transform([pred])[0]
    top3i  = np.argsort(proba)[-3:][::-1]
    top3   = [(le.inverse_transform([i])[0], round(proba[i]*100,1)) for i in top3i]
    meta   = CROP_META.get(crop, {'emoji':'🌱','season':'Variable','profit':'Medium','tip':'Monitor closely'})
    return {'crop': crop, 'confidence': conf, 'top3': top3, **meta}

# Pre-train on startup
if not os.path.exists(MODEL_PATH):
    train_model()

# ─────────────────────────────────────────────
#   SHARED CSS  (injected into every page)
# ─────────────────────────────────────────────
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --g:#2d6a4f;--gl:#40916c;--gp:#d8f3dc;--acc:#f4a261;--dk:#1b2a1e;
  --tx:#2d3a2e;--mt:#6b8f71;--wh:#fff;--bg:#f0f7f1;--bd:#c9e4ca;
  --dan:#e63946;--rad:14px;--sh:0 4px 24px rgba(45,106,79,.10);
  --fd:'Playfair Display',serif;--fb:'DM Sans',sans-serif;
}
body{font-family:var(--fb);color:var(--tx);background:var(--bg);min-height:100vh}
a{color:var(--gl);text-decoration:none}
a:hover{text-decoration:underline}

/* ── ALERTS ── */
.alert{padding:12px 16px;border-radius:9px;margin-bottom:16px;font-size:.9rem;font-weight:500}
.alert-danger{background:#fde8ea;color:#c62828;border:1px solid #f5c6cb}
.alert-warning{background:#fff8e1;color:#b57a00;border:1px solid #ffe082}
.alert-success{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}

/* ── NAVBAR ── */
.navbar{display:flex;justify-content:space-between;align-items:center;padding:16px 48px;
  background:rgba(255,255,255,.97);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100;
  border-bottom:1px solid var(--bd)}
.brand{font-family:var(--fd);font-size:1.35rem;color:var(--g);font-weight:900;letter-spacing:-.5px}
.nav-links{display:flex;gap:20px;align-items:center}
.nav-links a{color:var(--tx);font-weight:500;font-size:.95rem}
.btn-nav{background:var(--g);color:#fff!important;padding:9px 22px;border-radius:30px;font-weight:700;transition:.2s}
.btn-nav:hover{background:var(--gl);text-decoration:none!important}
.btn-logout{background:#fff0f0;color:var(--dan);padding:8px 18px;border-radius:20px;
  font-weight:600;font-size:.85rem;border:1px solid #ffc9c9}
.btn-logout:hover{background:var(--dan);color:#fff;text-decoration:none}

/* ── HOME ── */
.hero{display:flex;align-items:center;justify-content:space-between;padding:80px 48px 60px;
  min-height:88vh;position:relative;overflow:hidden}
.hero-bg{position:absolute;inset:0;
  background:radial-gradient(ellipse at 68% 40%,#b7e4c7 0%,var(--bg) 60%);z-index:0}
.hero-content{position:relative;z-index:1;max-width:560px}
.badge{display:inline-block;background:var(--gp);color:var(--g);padding:6px 16px;
  border-radius:20px;font-size:.82rem;font-weight:700;margin-bottom:18px;border:1px solid var(--bd)}
.hero-content h1{font-family:var(--fd);font-size:3.1rem;line-height:1.15;color:var(--dk);margin-bottom:18px}
.hl{color:var(--gl)}
.hero-sub{font-size:1.05rem;color:var(--mt);line-height:1.75;margin-bottom:32px}
.hero-btns{display:flex;gap:16px;flex-wrap:wrap}
.btn-primary{background:var(--g);color:#fff;padding:14px 30px;border-radius:30px;font-weight:700;
  font-size:1rem;box-shadow:0 4px 16px rgba(45,106,79,.25);transition:.2s}
.btn-primary:hover{background:var(--gl);text-decoration:none;transform:translateY(-2px)}
.btn-outline{border:2px solid var(--g);color:var(--g);padding:12px 28px;border-radius:30px;font-weight:600;transition:.2s}
.btn-outline:hover{background:var(--gp);text-decoration:none}
.hero-cards{position:relative;z-index:1;display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:320px;margin-left:40px}
.hcard{background:#fff;border-radius:var(--rad);padding:18px 20px;box-shadow:var(--sh);
  border:1px solid var(--bd);font-size:.9rem;color:var(--mt);line-height:1.8;
  animation:flt 3.5s ease-in-out infinite}
.hcard b{color:var(--dk);font-size:1.2rem;display:block}
.hcard.res{grid-column:span 2;background:var(--g);color:#fff;border-color:transparent;animation-delay:1s}
.hcard.res b{color:#fff;font-size:1.3rem}
.hcard.res small{opacity:.8}
@keyframes flt{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.features{padding:72px 48px;background:#fff;text-align:center}
.features h2{font-family:var(--fd);font-size:2.1rem;color:var(--dk);margin-bottom:40px}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:22px;max-width:920px;margin:0 auto}
.feat-card{padding:28px 22px;border:1px solid var(--bd);border-radius:var(--rad);background:var(--bg);text-align:left;transition:.2s}
.feat-card:hover{box-shadow:var(--sh);transform:translateY(-3px)}
.feat-icon{font-size:2rem;margin-bottom:10px}
.feat-card h3{font-size:.97rem;font-weight:700;margin-bottom:7px;color:var(--dk)}
.feat-card p{font-size:.87rem;color:var(--mt);line-height:1.6}
.footer{background:var(--dk);color:rgba(255,255,255,.55);text-align:center;padding:22px;font-size:.82rem}

/* ── AUTH ── */
.auth-body{background:var(--bg)}
.auth-wrap{display:flex;min-height:100vh}
.auth-left{flex:0 0 400px;background:linear-gradient(155deg,var(--g) 0%,var(--dk) 100%);
  padding:56px 44px;display:flex;flex-direction:column;justify-content:center;color:#fff}
.auth-left .brand{color:#fff;margin-bottom:44px;display:block;font-size:1.4rem}
.auth-left h2{font-family:var(--fd);font-size:2.1rem;line-height:1.3;margin-bottom:14px}
.auth-left p{color:rgba(255,255,255,.72);line-height:1.75;font-size:.97rem}
.auth-illo{font-size:2.4rem;letter-spacing:8px;margin-top:36px}
.auth-right{flex:1;display:flex;align-items:center;justify-content:center;padding:40px}
.auth-card{background:#fff;border-radius:20px;padding:44px;max-width:430px;width:100%;
  box-shadow:0 8px 44px rgba(45,106,79,.12)}
.auth-card h3{font-family:var(--fd);font-size:1.75rem;color:var(--dk);margin-bottom:26px}
.form-group{margin-bottom:17px}
.form-group label{display:block;font-size:.85rem;font-weight:600;color:var(--tx);margin-bottom:6px}
.form-group label span{color:var(--mt);font-weight:400;font-size:.78rem;margin-left:4px}
.form-group input,.form-group select{width:100%;padding:12px 15px;border:1.5px solid var(--bd);
  border-radius:10px;font-size:.94rem;font-family:var(--fb);transition:.2s;background:#fafcfa}
.form-group input:focus,.form-group select:focus{outline:none;border-color:var(--gl);background:#fff}
.btn-submit{width:100%;padding:13px;background:var(--g);color:#fff;border:none;border-radius:11px;
  font-size:.98rem;font-weight:700;cursor:pointer;transition:.2s;font-family:var(--fb)}
.btn-submit:hover{background:var(--gl);transform:translateY(-1px)}
.auth-switch{text-align:center;margin-top:18px;font-size:.88rem;color:var(--mt)}
.admin-hint{text-align:center;margin-top:9px;font-size:.78rem;color:var(--mt)}
.admin-hint code{background:var(--bg);padding:2px 7px;border-radius:4px;font-size:.8rem}

/* ── DASHBOARD ── */
.dash-nav{display:flex;justify-content:space-between;align-items:center;
  padding:15px 38px;background:#fff;border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:100}
.dash-nav .user-info{font-size:.88rem;color:var(--mt)}
.dash-nav .right{display:flex;align-items:center;gap:18px}
.dash-wrap{max-width:1180px;margin:0 auto;padding:36px 28px}
.dash-head{margin-bottom:28px}
.dash-head h1{font-family:var(--fd);font-size:1.95rem;color:var(--dk)}
.dash-head p{color:var(--mt);margin-top:6px;font-size:.95rem}
.dash-grid{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-bottom:26px}
.panel{background:#fff;border-radius:18px;padding:28px;box-shadow:var(--sh);border:1px solid var(--bd)}
.panel h2{font-family:var(--fd);font-size:1.32rem;margin-bottom:22px;color:var(--dk)}
.irow{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.btn-predict{width:100%;padding:13px;background:var(--g);color:#fff;border:none;border-radius:11px;
  font-size:1rem;font-weight:700;cursor:pointer;transition:.2s;font-family:var(--fb);margin-top:10px;letter-spacing:.4px}
.btn-predict:hover{background:var(--gl);transform:translateY(-2px)}
.result-ph{text-align:center;padding:44px 20px;color:var(--mt)}
.ph-ico{font-size:3.5rem;opacity:.4;margin-bottom:14px}
.result-ph h3{font-family:var(--fd);font-size:1.3rem;color:var(--tx);margin-bottom:8px}
.result-ph p{font-size:.88rem;line-height:1.6}
.result-box{text-align:center;animation:sli .4s ease-out}
@keyframes sli{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.r-emoji{font-size:3.8rem;margin-bottom:6px}
.r-label{font-size:.8rem;color:var(--mt);font-weight:600;text-transform:uppercase;letter-spacing:1px}
.r-crop{font-family:var(--fd);font-size:2.3rem;color:var(--g);font-weight:900;
  text-transform:capitalize;margin-bottom:14px}
.conf-bar{margin-bottom:18px}
.conf-bar .cb-label{font-size:.8rem;color:var(--mt);font-weight:600}
.cb-wrap{background:var(--bg);border-radius:10px;height:9px;margin:7px 0;overflow:hidden}
.cb-fill{height:100%;background:linear-gradient(90deg,var(--gl),#e9c46a);border-radius:10px}
.conf-bar b{font-size:1.05rem;color:var(--g)}
.crop-meta{text-align:left;background:var(--bg);border-radius:11px;padding:14px 16px;margin-bottom:18px}
.meta-row{font-size:.87rem;padding:3px 0;color:var(--tx)}
.top3 h4{font-size:.82rem;color:var(--mt);margin-bottom:9px;font-weight:700;text-align:left;text-transform:uppercase;letter-spacing:.5px}
.t3-item{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:.83rem}
.t3-item span{width:90px;text-align:right;text-transform:capitalize;color:var(--tx)}
.t3-bar{flex:1;background:var(--bg);border-radius:5px;height:7px;overflow:hidden}
.t3-bar div{height:100%;background:var(--gl);border-radius:5px}
.t3-item b{width:38px;text-align:right;color:var(--g)}
.hist-panel{background:#fff;border-radius:18px;padding:28px;box-shadow:var(--sh);border:1px solid var(--bd)}
.hist-panel h2{font-family:var(--fd);font-size:1.32rem;margin-bottom:18px;color:var(--dk)}
.tbl{width:100%;border-collapse:collapse;font-size:.84rem}
.tbl th{background:var(--bg);padding:11px 13px;text-align:left;font-weight:700;
  color:var(--mt);border-bottom:2px solid var(--bd)}
.tbl td{padding:10px 13px;border-bottom:1px solid var(--bd)}
.tbl tr:hover td{background:#f6fbf7}
.crop-pill{background:var(--gp);color:var(--g);padding:3px 10px;border-radius:20px;
  font-size:.78rem;text-transform:capitalize;font-weight:600}

/* ── ADMIN ── */
.admin-body{background:#eef0f5}
.admin-nav{display:flex;justify-content:space-between;align-items:center;
  padding:15px 38px;background:var(--dk)}
.admin-nav .brand{color:#fff}
.abadge{background:var(--acc);color:var(--dk);font-size:.68rem;font-weight:800;
  padding:3px 8px;border-radius:4px;letter-spacing:1px;font-family:var(--fb)}
.admin-nav .right{display:flex;align-items:center;gap:18px;color:rgba(255,255,255,.65);font-size:.86rem}
.admin-wrap{max-width:1320px;margin:0 auto;padding:36px 28px}
.admin-wrap>h1{font-family:var(--fd);font-size:1.9rem;color:var(--dk)}
.admin-sub{color:var(--mt);margin-top:5px;margin-bottom:30px;font-size:.93rem}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:36px}
.stat{background:#fff;border-radius:var(--rad);padding:24px;text-align:center;box-shadow:var(--sh);border:1px solid var(--bd)}
.stat .ico{font-size:1.8rem;margin-bottom:8px}
.stat .num{font-family:var(--fd);font-size:1.9rem;font-weight:900;color:var(--g)}
.stat .lbl{font-size:.8rem;color:var(--mt);margin-top:3px}
.asec{background:#fff;border-radius:18px;padding:28px;margin-bottom:24px;box-shadow:var(--sh);border:1px solid var(--bd)}
.asec h2{font-family:var(--fd);font-size:1.3rem;color:var(--dk);margin-bottom:18px}
.tscroll{overflow-x:auto}
.empty{color:var(--mt);font-style:italic;text-align:center;padding:24px}

@media(max-width:900px){
  .hero{flex-direction:column;padding:40px 22px}
  .hero-cards{margin:28px 0 0;width:100%;max-width:100%}
  .dash-grid{grid-template-columns:1fr}
  .auth-left{display:none}
  .stats{grid-template-columns:1fr 1fr}
  .navbar,.dash-nav,.admin-nav{padding:14px 18px}
  .features{padding:44px 18px}
  .hero-content h1{font-size:2.1rem}
  .irow{grid-template-columns:1fr}
}
</style>
"""

# ─────────────────────────────────────────────
#   HTML TEMPLATES
# ─────────────────────────────────────────────

HOME_HTML = CSS + """
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgroMind AI – Smart Crop Prediction</title></head>
<body>
<nav class="navbar">
  <div class="brand">🌾 AgroMind AI</div>
  <div class="nav-links">
    <a href="/login">Login</a>
    <a href="/signup" class="btn-nav">Get Started Free</a>
  </div>
</nav>

<section class="hero">
  <div class="hero-bg"></div>
  <div class="hero-content">
    <span class="badge">🤖 AI-Powered Agriculture</span>
    <h1>Predict the <span class="hl">Perfect Crop</span><br>for Your Region</h1>
    <p class="hero-sub">
      Enter your soil nutrients, rainfall, temperature & humidity —
      our Random Forest AI instantly recommends the most profitable crop for your farm.
    </p>
    <div class="hero-btns">
      <a href="/signup" class="btn-primary">Start Predicting Free →</a>
      <a href="/login" class="btn-outline">I Have an Account</a>
    </div>
  </div>
  <div class="hero-cards">
    <div class="hcard">🌡️ Temperature<b>24 °C</b></div>
    <div class="hcard" style="animation-delay:.6s">💧 Rainfall<b>180 mm</b></div>
    <div class="hcard" style="animation-delay:1.2s">🌱 Soil pH<b>6.5</b></div>
    <div class="hcard res">✅ Best Crop<b>Rice 🌾</b><small>94% AI Confidence</small></div>
  </div>
</section>

<section class="features">
  <h2>Why AgroMind AI?</h2>
  <div class="feat-grid">
    <div class="feat-card"><div class="feat-icon">🧬</div><h3>Multi-Factor Analysis</h3>
      <p>Analyses N, P, K nutrients, temperature, humidity, pH and annual rainfall together for precise results.</p></div>
    <div class="feat-card"><div class="feat-icon">🤖</div><h3>Random Forest Model</h3>
      <p>Trained on 24 crop varieties. Provides confidence scores and top-3 alternative recommendations.</p></div>
    <div class="feat-card"><div class="feat-icon">💰</div><h3>Profit Optimization</h3>
      <p>Each recommendation includes profitability rating, best season, and expert farming tips.</p></div>
    <div class="feat-card"><div class="feat-icon">📊</div><h3>Full Admin Visibility</h3>
      <p>Admin dashboard shows all users and every prediction made — complete data transparency.</p></div>
  </div>
</section>
<footer class="footer">© 2024 AgroMind AI · Jacob.ai Smart Agriculture Challenge · Admin: admin@jacob.ai</footer>
</body></html>
"""

LOGIN_HTML = CSS + """
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Login – AgroMind AI</title></head>
<body class="auth-body">
<div class="auth-wrap">
  <div class="auth-left">
    <a href="/" class="brand">🌾 AgroMind AI</a>
    <h2>Welcome Back,<br>Farmer 👋</h2>
    <p>Login to access your AI-powered crop predictions and personal dashboard.</p>
    <div class="auth-illo">🌱🌿🌾🌽🍅</div>
  </div>
  <div class="auth-right">
    <div class="auth-card">
      <h3>Sign In</h3>
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ cat }}">{{ msg }}</div>
        {% endfor %}
      {% endwith %}
      <form method="POST">
        <div class="form-group">
          <label>Email Address</label>
          <input type="email" name="email" placeholder="you@example.com" required>
        </div>
        <div class="form-group">
          <label>Password</label>
          <input type="password" name="password" placeholder="••••••••" required>
        </div>
        <button type="submit" class="btn-submit">Login →</button>
      </form>
      <p class="auth-switch">New user? <a href="/signup">Create an account</a></p>
      <p class="admin-hint">Admin login: <code>admin@jacob.ai</code> / <code>Admin@1234</code></p>
    </div>
  </div>
</div>
</body></html>
"""

SIGNUP_HTML = CSS + """
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign Up – AgroMind AI</title></head>
<body class="auth-body">
<div class="auth-wrap">
  <div class="auth-left">
    <a href="/" class="brand">🌾 AgroMind AI</a>
    <h2>Join AgroMind,<br>Grow Smarter 🌱</h2>
    <p>Create your free account and receive AI-powered crop recommendations tailored to your farm region.</p>
    <div class="auth-illo">🌻☀️🌧️🌍🧑‍🌾</div>
  </div>
  <div class="auth-right">
    <div class="auth-card">
      <h3>Create Account</h3>
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ cat }}">{{ msg }}</div>
        {% endfor %}
      {% endwith %}
      <form method="POST">
        <div class="form-group">
          <label>Full Name</label>
          <input type="text" name="name" placeholder="Your full name" required>
        </div>
        <div class="form-group">
          <label>Email Address</label>
          <input type="email" name="email" placeholder="you@example.com" required>
        </div>
        <div class="form-group">
          <label>Region / District</label>
          <input type="text" name="region" placeholder="e.g. Punjab, Tamil Nadu, Telangana…" required>
        </div>
        <div class="form-group">
          <label>Password <span>(min 6 characters)</span></label>
          <input type="password" name="password" placeholder="••••••••" required minlength="6">
        </div>
        <button type="submit" class="btn-submit">Create Account →</button>
      </form>
      <p class="auth-switch">Already registered? <a href="/login">Login</a></p>
    </div>
  </div>
</div>
</body></html>
"""

DASHBOARD_HTML = CSS + """
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard – AgroMind AI</title></head>
<body>
<nav class="dash-nav">
  <div class="brand">🌾 AgroMind AI</div>
  <div class="right">
    <span class="user-info">👤 {{ user.name }} &nbsp;·&nbsp; 📍 {{ user.region }}</span>
    <a href="/logout" class="btn-logout">Logout</a>
  </div>
</nav>

<div class="dash-wrap">
  <div class="dash-head">
    <h1>Hello, {{ user.name }} 👋</h1>
    <p>Enter your field data below — our AI will recommend the best and most profitable crop for your region.</p>
  </div>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for cat, msg in messages %}<div class="alert alert-{{ cat }}">{{ msg }}</div>{% endfor %}
  {% endwith %}

  <div class="dash-grid">
    <!-- INPUT FORM -->
    <div class="panel">
      <h2>🧪 Enter Field Data</h2>
      <form method="POST">
        <div class="irow">
          <div class="form-group">
            <label>Nitrogen (N) <span>kg/ha</span></label>
            <input type="number" name="nitrogen" placeholder="0 – 140" min="0" max="140" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Phosphorus (P) <span>kg/ha</span></label>
            <input type="number" name="phosphorus" placeholder="5 – 145" min="0" max="145" step="0.1" required>
          </div>
        </div>
        <div class="irow">
          <div class="form-group">
            <label>Potassium (K) <span>kg/ha</span></label>
            <input type="number" name="potassium" placeholder="5 – 205" min="0" max="205" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Temperature <span>°C</span></label>
            <input type="number" name="temperature" placeholder="8 – 44" min="0" max="60" step="0.1" required>
          </div>
        </div>
        <div class="irow">
          <div class="form-group">
            <label>Humidity <span>%</span></label>
            <input type="number" name="humidity" placeholder="10 – 100" min="0" max="100" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Soil pH <span>0 – 14</span></label>
            <input type="number" name="ph" placeholder="3.5 – 9.9" min="0" max="14" step="0.01" required>
          </div>
        </div>
        <div class="form-group">
          <label>Annual Rainfall <span>mm</span></label>
          <input type="number" name="rainfall" placeholder="20 – 300" min="0" max="500" step="0.1" required>
        </div>
        <button type="submit" class="btn-predict">🤖 Predict Best Crop</button>
      </form>
    </div>

    <!-- RESULT PANEL -->
    <div class="panel">
      {% if result %}
      <h2>🌱 AI Recommendation</h2>
      <div class="result-box">
        <div class="r-emoji">{{ result.emoji }}</div>
        <div class="r-label">Recommended Crop</div>
        <div class="r-crop">{{ result.crop }}</div>
        <div class="conf-bar">
          <span class="cb-label">AI Confidence</span>
          <div class="cb-wrap"><div class="cb-fill" style="width:{{ result.confidence }}%"></div></div>
          <b>{{ result.confidence }}%</b>
        </div>
        <div class="crop-meta">
          <div class="meta-row">📅 Best Season: <b>{{ result.season }}</b></div>
          <div class="meta-row">💰 Profitability: <b>{{ result.profit }}</b></div>
          <div class="meta-row">💡 Tip: <i>{{ result.tip }}</i></div>
        </div>
        <div class="top3">
          <h4>Top 3 Predictions</h4>
          {% for crop, pct in result.top3 %}
          <div class="t3-item">
            <span>{{ crop }}</span>
            <div class="t3-bar"><div style="width:{{ pct }}%"></div></div>
            <b>{{ pct }}%</b>
          </div>
          {% endfor %}
        </div>
      </div>
      {% else %}
      <h2>🌱 AI Recommendation</h2>
      <div class="result-ph">
        <div class="ph-ico">🌾</div>
        <h3>Your Prediction Appears Here</h3>
        <p>Fill in your soil & climate data on the left,<br>then click <b>Predict Best Crop</b>.</p>
      </div>
      {% endif %}
    </div>
  </div>

  <!-- HISTORY -->
  {% if history %}
  <div class="hist-panel">
    <h2>📋 Your Recent Predictions</h2>
    <div class="tscroll">
    <table class="tbl">
      <thead>
        <tr><th>Date</th><th>N</th><th>P</th><th>K</th><th>Temp °C</th>
        <th>Humidity %</th><th>pH</th><th>Rainfall mm</th><th>Predicted Crop</th><th>Confidence</th></tr>
      </thead>
      <tbody>
        {% for p in history %}
        <tr>
          <td>{{ p.timestamp.strftime('%d %b %Y') }}</td>
          <td>{{ p.nitrogen }}</td><td>{{ p.phosphorus }}</td><td>{{ p.potassium }}</td>
          <td>{{ p.temperature }}</td><td>{{ p.humidity }}</td><td>{{ p.ph }}</td>
          <td>{{ p.rainfall }}</td>
          <td><span class="crop-pill">{{ p.predicted_crop }}</span></td>
          <td>{{ p.confidence }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </div>
  {% endif %}
</div>
</body></html>
"""

ADMIN_HTML = CSS + """
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin Panel – AgroMind AI</title></head>
<body class="admin-body">
<nav class="admin-nav">
  <div class="brand">🌾 AgroMind AI &nbsp;<span class="abadge">ADMIN</span></div>
  <div class="right">
    <span>🔐 admin@jacob.ai</span>
    <a href="/logout" class="btn-logout">Logout</a>
  </div>
</nav>

<div class="admin-wrap">
  <h1>Admin Dashboard</h1>
  <p class="admin-sub">Full visibility into all registered users and their AI crop predictions.</p>

  <div class="stats">
    <div class="stat"><div class="ico">👥</div><div class="num">{{ total_users }}</div><div class="lbl">Registered Users</div></div>
    <div class="stat"><div class="ico">🤖</div><div class="num">{{ total_predictions }}</div><div class="lbl">Total Predictions</div></div>
    <div class="stat"><div class="ico">🌾</div><div class="num">24</div><div class="lbl">Supported Crops</div></div>
    <div class="stat"><div class="ico">✅</div><div class="num" style="font-size:1.2rem">Active</div><div class="lbl">AI Model Status</div></div>
  </div>

  <!-- USERS -->
  <div class="asec">
    <h2>👥 All Registered Users</h2>
    {% if users %}
    <table class="tbl">
      <thead><tr><th>#</th><th>Name</th><th>Email</th><th>Region</th><th>Registered On</th><th>Predictions</th></tr></thead>
      <tbody>
        {% for u in users %}
        <tr>
          <td>{{ u.id }}</td><td>{{ u.name }}</td><td>{{ u.email }}</td>
          <td>{{ u.region }}</td>
          <td>{{ u.created_at.strftime('%d %b %Y') if u.created_at else 'N/A' }}</td>
          <td>{{ u.predictions|length }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}<p class="empty">No users registered yet.</p>{% endif %}
  </div>

  <!-- ALL PREDICTIONS -->
  <div class="asec">
    <h2>🌱 All Crop Predictions</h2>
    {% if predictions %}
    <div class="tscroll">
    <table class="tbl">
      <thead>
        <tr><th>#</th><th>User</th><th>Region</th><th>N</th><th>P</th><th>K</th>
        <th>Temp °C</th><th>Humidity %</th><th>pH</th><th>Rainfall mm</th>
        <th>Predicted Crop</th><th>Confidence</th><th>Date & Time</th></tr>
      </thead>
      <tbody>
        {% for p in predictions %}
        <tr>
          <td>{{ p.id }}</td>
          <td>{{ p.user.name }}</td>
          <td>{{ p.user.region }}</td>
          <td>{{ p.nitrogen }}</td><td>{{ p.phosphorus }}</td><td>{{ p.potassium }}</td>
          <td>{{ p.temperature }}</td><td>{{ p.humidity }}</td><td>{{ p.ph }}</td>
          <td>{{ p.rainfall }}</td>
          <td><span class="crop-pill">{{ p.predicted_crop }}</span></td>
          <td>{{ p.confidence }}%</td>
          <td>{{ p.timestamp.strftime('%d %b %Y %H:%M') }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
    {% else %}<p class="empty">No predictions made yet.</p>{% endif %}
  </div>
</div>
</body></html>
"""

# ─────────────────────────────────────────────
#   ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pw    = request.form['password']
        if email == ADMIN_EMAIL and pw == ADMIN_PASSWORD:
            session.clear()
            session['admin'] = True
            return redirect(url_for('admin'))
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(pw):
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        flash('Invalid email or password. Please try again.', 'danger')
    return render_template_string(LOGIN_HTML)

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        name   = request.form['name']
        email  = request.form['email']
        pw     = request.form['password']
        region = request.form['region']
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))
        u = User(name=name, email=email, region=region, created_at=datetime.utcnow())
        u.set_password(pw)
        db.session.add(u)
        db.session.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template_string(SIGNUP_HTML)

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user   = User.query.get(session['user_id'])
    result = None
    if request.method == 'POST':
        try:
            N    = float(request.form['nitrogen'])
            P    = float(request.form['phosphorus'])
            K    = float(request.form['potassium'])
            temp = float(request.form['temperature'])
            hum  = float(request.form['humidity'])
            ph   = float(request.form['ph'])
            rain = float(request.form['rainfall'])
            result = predict_crop(N, P, K, temp, hum, ph, rain)
            pred = Prediction(
                user_id=user.id, nitrogen=N, phosphorus=P, potassium=K,
                temperature=temp, humidity=hum, ph=ph, rainfall=rain,
                predicted_crop=result['crop'], confidence=result['confidence'],
                timestamp=datetime.utcnow()
            )
            db.session.add(pred)
            db.session.commit()
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    history = Prediction.query.filter_by(user_id=user.id)\
                .order_by(Prediction.timestamp.desc()).limit(8).all()
    return render_template_string(DASHBOARD_HTML, user=user, result=result, history=history)

@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect(url_for('login'))
    users       = User.query.all()
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template_string(ADMIN_HTML,
        users=users, predictions=predictions,
        total_users=len(users), total_predictions=len(predictions))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# ─────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("\n" + "="*55)
    print("  🌾 AgroMind AI – Smart Agriculture App")
    print("="*55)
    print("  URL    : http://127.0.0.1:5000")
    print("  Admin  : admin@jacob.ai  |  Admin@1234")
    print("="*55 + "\n")
    app.run(debug=True)