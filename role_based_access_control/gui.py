import streamlit as st
import requests
import jwt

API = "http://127.0.0.1:5000"
APP_SECRET = "change_this_strong_secret"
JWT_ALGORITHM = "HS256"

st.title("RBAC Demo - Streamlit GUI")

# Store token
if "token" not in st.session_state:
    st.session_state.token = None


# ------------------------------------------------------
# Helper: decode token and extract roles
# ------------------------------------------------------
def get_user_roles(token):
    try:
        decoded = jwt.decode(token, APP_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded.get("roles", [])
    except Exception:
        return []


# ------------------------------------------------------
# If not logged in → Show login page
# ------------------------------------------------------
if st.session_state.token is None:

    st.subheader("Login")

    user = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pwd")

    if st.button("Login", key="login_btn"):
        r = requests.post(API + "/login", json={"username": user, "password": pwd})
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data["token"]

            st.success("Login successful")

            if "login_time_ms" in data:
                st.info(f"Login Time: {data['login_time_ms']:.3f} ms")

            st.rerun()
        else:
            st.error("Invalid credentials")

# ------------------------------------------------------
# If logged in → Show dashboard
# ------------------------------------------------------
else:

    token = st.session_state.token
    roles = get_user_roles(token)
    is_admin = "Admin" in roles

    st.success(f"Logged in  |  Roles: {roles}")

    # Logout button
    if st.button("Logout", key="logout_btn"):
        st.session_state.token = None
        st.rerun()

    auth_header = {"Authorization": "Bearer " + token}

    # ------------------------------------------------------
    # VIEW ITEMS
    # ------------------------------------------------------
    st.header("View Items")

    if st.button("Load Items", key="load_items_btn"):
        r = requests.get(API + "/items", headers=auth_header)
        data = r.json()

        st.json(data)

        if "api_time_ms" in data:
            st.info(f"API Time: {data['api_time_ms']:.3f} ms")
        if "permission_check_ms" in data:
            st.info(f"Permission Check: {data['permission_check_ms']:.3f} ms")


    # ------------------------------------------------------
    # CREATE ITEM
    # ------------------------------------------------------
    st.header("Create Item")

    item_name = st.text_input("Item name", key="create_item_name")

    if st.button("Create Item", key="create_item_btn"):
        r = requests.post(
            API + "/items",
            json={"name": item_name},
            headers=auth_header
        )
        data = r.json()

        st.json(data)

        if "api_time_ms" in data:
            st.info(f"API Time: {data['api_time_ms']:.3f} ms")
        if "permission_check_ms" in data:
            st.info(f"Permission Check: {data['permission_check_ms']:.3f} ms")


    # ------------------------------------------------------
    # ADMIN PANEL - ONLY FOR ADMIN USERS
    # ------------------------------------------------------
    if is_admin:

        st.header("Admin Panel   (Admin Only)")

        # -------------------- LIST USERS --------------------
        with st.expander("List Users"):
            if st.button("Load Users", key="load_users_btn"):
                r = requests.get(API + "/admin/users", headers=auth_header)
                st.json(r.json())

        # -------------------- LIST ROLES --------------------
        with st.expander("List Roles"):
            if st.button("Load Roles", key="load_roles_btn"):
                r = requests.get(API + "/admin/roles", headers=auth_header)
                st.json(r.json())

        # -------------------- CREATE USER --------------------
        with st.expander("Create User"):
            new_user = st.text_input("Username", key="new_user_name")
            new_pass = st.text_input("Password", type="password", key="new_user_pass")

            if st.button("Create User", key="create_user_btn"):
                r = requests.post(
                    API + "/admin/create_user",
                    json={"username": new_user, "password": new_pass},
                    headers=auth_header
                )
                st.json(r.json())

        # -------------------- CREATE ROLE --------------------
        with st.expander("Create Role"):
            new_role = st.text_input("Role Name", key="new_role_name")

            if st.button("Create Role", key="create_role_btn"):
                r = requests.post(
                    API + "/admin/create_role",
                    json={"name": new_role},
                    headers=auth_header
                )
                st.json(r.json())

        # -------------------- CREATE PERMISSION --------------------
        with st.expander("Create Permission"):
            new_perm = st.text_input("Permission Name", key="new_perm_name")

            if st.button("Create Permission", key="create_perm_btn"):
                r = requests.post(
                    API + "/admin/create_permission",
                    json={"name": new_perm},
                    headers=auth_header
                )
                st.json(r.json())

        # -------------------- ASSIGN ROLE TO USER --------------------
        with st.expander("Assign Role to User"):
            target_user = st.text_input("Target Username", key="assign_role_user")
            target_role = st.text_input("Role Name", key="assign_role_role")

            if st.button("Assign Role", key="assign_role_btn"):
                r = requests.post(
                    API + "/admin/assign_role",
                    json={"username": target_user, "role": target_role},
                    headers=auth_header
                )
                st.json(r.json())

        # -------------------- ADD PERMISSION TO ROLE --------------------
        with st.expander("Add Permission to Role"):
            role_name = st.text_input("Role Name", key="add_perm_role")
            perm_name = st.text_input("Permission Name", key="add_perm_name")

            if st.button("Add Permission", key="add_perm_btn"):
                r = requests.post(
                    API + "/admin/add_permission",
                    json={"role": role_name, "permission": perm_name},
                    headers=auth_header
                )
                st.json(r.json())

    else:
        st.warning("Admin features hidden. You do not have admin privileges.")