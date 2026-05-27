import os
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import List
import time

from flask import Flask, request, jsonify, g
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime,
    Table, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
import jwt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
APP_SECRET = os.environ.get("RBAC_APP_SECRET", "change_this_strong_secret")
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_MINUTES = 60

DB_FILE = "rbac_full.db"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "rbac_audit.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
audit = logging.getLogger("rbac_audit")

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False, future=True)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)


# ---------------------------------------------------------
# TABLES
# ---------------------------------------------------------
user_roles = Table(
    "user_roles", Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True)
)

role_permissions = Table(
    "role_permissions", Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("permissions.id"), primary_key=True)
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    full_name = Column(String(120), nullable=True)
    email = Column(String(120), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    roles = relationship("Role", secondary=user_roles, back_populates="users")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_permissions(self):
        perms = set()
        for r in self.roles:
            for p in r.permissions:
                perms.add(p.name)
        return sorted(perms)


class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True, nullable=False)
    description = Column(Text)
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):
    __tablename__ = "permissions"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), unique=True, nullable=False)
    description = Column(Text)
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


# ---------------------------------------------------------
# DB UTILITIES
# ---------------------------------------------------------
def init_db():
    Base.metadata.create_all(bind=engine)


def create_user(session, username, password, full="", email=""):
    if session.query(User).filter_by(username=username).first():
        raise ValueError("User already exists.")
    u = User(username=username, full_name=full, email=email)
    u.set_password(password)
    session.add(u)
    session.commit()
    return u


def create_role(session, name, desc=""):
    r = session.query(Role).filter_by(name=name).first()
    if not r:
        r = Role(name=name, description=desc)
        session.add(r)
        session.commit()
    return r


def create_permission(session, name, desc=""):
    p = session.query(Permission).filter_by(name=name).first()
    if not p:
        p = Permission(name=name, description=desc)
        session.add(p)
        session.commit()
    return p


def assign_role_to_user(session, username, role_name):
    u = session.query(User).filter_by(username=username).first()
    r = session.query(Role).filter_by(name=role_name).first()
    if not u or not r:
        raise ValueError("Invalid user or role")
    if r not in u.roles:
        u.roles.append(r)
        session.commit()


def add_permission_to_role(session, role_name, perm_name):
    r = session.query(Role).filter_by(name=role_name).first()
    p = session.query(Permission).filter_by(name=perm_name).first()
    if r and p and p not in r.permissions:
        r.permissions.append(p)
        session.commit()


# ---------------------------------------------------------
# JWT
# ---------------------------------------------------------
def generate_token(user):
    payload = {
        "sub": user.username,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES),
        "roles": [r.name for r in user.roles],
        "perms": user.get_permissions()
    }
    token = jwt.PyJWT().encode(payload, APP_SECRET, algorithm=JWT_ALGORITHM)
    return token


def decode_token(token):
    return jwt.PyJWT().decode(token, APP_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------
# PERMISSION DECORATOR
# ---------------------------------------------------------
def require_permission(perm):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"error": "missing auth"}), 401

            token = auth.split()[1]

            try:
                data = decode_token(token)
            except:
                duration = (time.time() - start) * 1000
                return jsonify({"error": "invalid token", "permission_time_ms": duration}), 401

            if perm not in data.get("perms", []):
                duration = (time.time() - start) * 1000
                return jsonify({"error": "forbidden", "permission_time_ms": duration}), 403

            g.permission_time = (time.time() - start) * 1000
            g.user = data["sub"]

            return f(*args, **kwargs)

        return wrapper
    return decorator


# ---------------------------------------------------------
# ROLE DECORATOR
# ---------------------------------------------------------
def require_role(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"error": "missing auth"}), 401

            token = auth.split()[1]

            try:
                data = decode_token(token)
            except:
                duration = (time.time() - start) * 1000
                return jsonify({"error": "invalid token", "role_time_ms": duration}), 401

            if role not in data.get("roles", []):
                duration = (time.time() - start) * 1000
                return jsonify({"error": "forbidden", "role_time_ms": duration}), 403

            g.role_time = (time.time() - start) * 1000
            g.user = data["sub"]

            return f(*args, **kwargs)

        return wrapper
    return decorator


# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)

@app.before_request
def open_db():
    g.db = Session()

@app.teardown_request
def close_db(e):
    db = getattr(g, "db", None)
    if db:
        db.close()
        Session.remove()


# ---------------------------------------------------------
# ROUTES WITH PERFORMANCE
# ---------------------------------------------------------
@app.route("/login", methods=["POST"])
def login():
    start = time.time()

    d = request.json or {}
    u = g.db.query(User).filter_by(username=d.get("username")).first()

    if not u or not u.check_password(d.get("password", "")):
        duration = (time.time() - start) * 1000
        return jsonify({"error": "invalid credentials", "login_time_ms": duration}), 401

    token = generate_token(u)

    duration = (time.time() - start) * 1000
    return jsonify({"token": token, "login_time_ms": duration})


# ---------------------------------------------------------
# PROTECTED USER ROUTES
# ---------------------------------------------------------
@app.route("/items", methods=["GET"])
@require_permission("view_item")
def items():
    start = time.time()

    data = [
        {"id": 1, "name": "DocA"},
        {"id": 2, "name": "ReportB"}
    ]

    api_time = (time.time() - start) * 1000

    return jsonify({
        "items": data,
        "api_time_ms": api_time,
        "permission_check_ms": g.permission_time
    })


@app.route("/items", methods=["POST"])
@require_permission("create_item")
def items_create():
    start = time.time()

    d = request.json or {}
    api_time = (time.time() - start) * 1000

    return jsonify({
        "message": "item created",
        "name": d.get("name", ""),
        "api_time_ms": api_time,
        "permission_check_ms": g.permission_time
    })


# ---------------------------------------------------------
# ADMIN ROUTES (NEW)
# ---------------------------------------------------------
@app.route("/admin/create_user", methods=["POST"])
@require_role("Admin")
def admin_create_user():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    full = data.get("full_name", "")
    email = data.get("email", "")

    start = time.time()

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400

    try:
        u = create_user(g.db, username, password, full, email)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    api_time = (time.time() - start) * 1000
    return jsonify({"message": "user created", "api_time_ms": api_time})


@app.route("/admin/create_role", methods=["POST"])
@require_role("Admin")
def admin_create_role():
    data = request.json or {}
    name = data.get("name")
    desc = data.get("description", "")

    start = time.time()

    if not name:
        return jsonify({"error": "role name required"}), 400

    r = create_role(g.db, name, desc)

    api_time = (time.time() - start) * 1000
    return jsonify({"message": "role created", "role": name, "api_time_ms": api_time})


@app.route("/admin/create_permission", methods=["POST"])
@require_role("Admin")
def admin_create_permission():
    data = request.json or {}
    name = data.get("name")
    desc = data.get("description", "")

    start = time.time()

    if not name:
        return jsonify({"error": "permission name required"}), 400

    p = create_permission(g.db, name, desc)

    api_time = (time.time() - start) * 1000
    return jsonify({"message": "permission created", "permission": name, "api_time_ms": api_time})


@app.route("/admin/assign_role", methods=["POST"])
@require_role("Admin")
def admin_assign_role():
    data = request.json or {}
    username = data.get("username")
    role = data.get("role")

    start = time.time()

    if not username or not role:
        return jsonify({"error": "username and role required"}), 400

    try:
        assign_role_to_user(g.db, username, role)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    api_time = (time.time() - start) * 1000
    return jsonify({"message": f"role {role} assigned", "api_time_ms": api_time})


@app.route("/admin/add_permission", methods=["POST"])
@require_role("Admin")
def admin_add_permission():
    data = request.json or {}
    role = data.get("role")
    perm = data.get("permission")

    start = time.time()

    if not role or not perm:
        return jsonify({"error": "role and permission required"}), 400

    add_permission_to_role(g.db, role, perm)

    api_time = (time.time() - start) * 1000
    return jsonify({"message": f"permission {perm} added to role {role}", "api_time_ms": api_time})


# ---------------------------------------------------------
# ADMIN VIEW ROUTES
# ---------------------------------------------------------
@app.route("/admin/users", methods=["GET"])
@require_role("Admin")
def list_users():
    start = time.time()

    users = g.db.query(User).all()
    out = [{"username": u.username, "roles": [r.name for r in u.roles]} for u in users]

    api_time = (time.time() - start) * 1000

    return jsonify({
        "users": out,
        "api_time_ms": api_time,
        "role_check_ms": g.role_time
    })


@app.route("/admin/roles", methods=["GET"])
@require_role("Admin")
def list_roles():
    start = time.time()

    roles = g.db.query(Role).all()
    out = [{"role": r.name, "perms": [p.name for p in r.permissions]} for r in roles]

    api_time = (time.time() - start) * 1000

    return jsonify({
        "roles": out,
        "api_time_ms": api_time,
        "role_check_ms": g.role_time
    })


# ---------------------------------------------------------
# INITIAL DATA
# ---------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        init_db()

        s = Session()
        admin = create_role(s, "Admin")
        user_role = create_role(s, "User")
        editor = create_role(s, "Editor")

        p_view = create_permission(s, "view_item")
        p_create = create_permission(s, "create_item")

        add_permission_to_role(s, "Admin", "view_item")
        add_permission_to_role(s, "Admin", "create_item")
        add_permission_to_role(s, "User", "view_item")

        alice = create_user(s, "alice", "AliceStrongPass1!", "Alice Admin")
        assign_role_to_user(s, "alice", "Admin")

        bob = create_user(s, "bob", "BobPass123!")
        assign_role_to_user(s, "bob", "User")

        s.close()

        print("DB created.")

    print("Running at http://127.0.0.1:5000")
    app.run(debug=True)
