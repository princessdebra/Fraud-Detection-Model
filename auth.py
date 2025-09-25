# auth.py — Beautiful glassmorphism UI with working signup switch
import re
import base64
from pathlib import Path
import streamlit as st
from database import FraudDetectionDB
import secrets
from datetime import datetime, timedelta

# --- password rules you already have will be reused ---
RESET_TOKEN_TTL_MIN = 60  # token valid for 60 minutes
# ----------------------------- Page / Theme -----------------------------
def set_page_config():
    st.set_page_config(
        page_title="Minet Fraud Detection",
        page_icon=" ",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

# ----------------------------- Assets -----------------------------
def _inline_logo_html() -> str:
    # Try common locations; if none found, return markup with text fallback
    for p in [
        Path("assets/min1.png"), Path("assets/min1.jpg"), Path("assets/min1.jpeg"),
        Path.cwd() / "assets" / "min1.png", Path.cwd() / "assets" / "min1.jpg", Path.cwd() / "assets" / "min1.jpeg",
    ]:
        if p.exists():
            ext = p.suffix.lower().lstrip(".")
            mime = "png" if ext == "png" else "jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f'<img class="minet-logo" src="data:image/{mime};base64,{b64}" alt="Minet Logo" />'
    # text fallback (now styled)
    return '<div class="minet-text-logo">MINET</div>'
def render_auth_branding(title="Minet Fraud Detection"):
    st.markdown(
        f"""
        <div class="auth-brand">
            {_inline_logo_html()}
            <div class="minet-text-logo" style="letter-spacing:.08rem; font-size:1.15rem;">
                {title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def _route_from_query_param():
    """Handle ?reset=... and ?auth=... using ONLY st.query_params."""
    qp = dict(st.query_params)

    # 1) Reset link route
    if "reset" in qp and qp["reset"]:
        token = qp["reset"]
        # clean URL, then render reset screen
        st.query_params.clear()
        # store token in state and rerun so we render the correct screen
        st.session_state._pending_reset_token = token
        st.rerun()

    # 2) Auth tab route
    tab = qp.get("auth")
    if tab in {"login", "signup", "forgot"}:
        st.session_state.auth_tab = tab
        st.query_params.clear()
        st.rerun()




# ----------------------------- CSS -----------------------------
def apply_styles():
    is_dark = (st.get_option("theme.base") or "").lower() == "dark"
    st.markdown(f"""
<script>
document.documentElement.classList.toggle('st-dark', {str(is_dark).lower()});
</script>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
/* ------------------ Base Reset ------------------ */
header, footer, #MainMenu { display: none !important; }

/* Kill the Streamlit white top-pill decoration */
div[data-testid="stDecoration"],
section[data-testid="stDecoration"],
header[data-testid="stDecoration"] {
    display: none !important;
}

/* ------------------ Beautiful Background ------------------ */
.stApp {
    background:
        radial-gradient(900px 500px at 50% -200px, #ffffff 0%, #f5f7fb 40%, #eef1f7 100%),
        radial-gradient(800px 500px at 100% 120%, #ffdfe1 0%, transparent 50%),
        radial-gradient(700px 400px at 0% 120%, #ffe8e9 0%, transparent 50%);
    min-height: 100vh;
    display: flex;
    align-items: center;
}
.st-dark .stApp {
    background:
        radial-gradient(900px 500px at 50% -200px, #10131a 0%, #0d1117 40%, #0b0f14 100%),
        radial-gradient(800px 500px at 100% 120%, rgba(215,25,32,.20) 0%, transparent 55%),
        radial-gradient(700px 400px at 0% 120%, rgba(215,25,32,.12) 0%, transparent 60%);
}

/* ------------------ Logo ------------------ */
.minet-logo {
    display: block;
    margin: 0 auto 8px;
    width: 132px;
    height: auto;
    filter: drop-shadow(0 10px 22px #d719200d);
}

/* ------------------ Typography ------------------ */
.auth-title {
    text-align: center;
    font-size: 1.95rem;
    font-weight: 800;
    color: #0f1216;
    letter-spacing: .2px;
    margin-bottom: 0.5rem;
}
.auth-sub {
    text-align: center;
    color: #7b8595;
    margin-bottom: 2rem;
}
.st-dark .auth-title { color: #e8eef6; }
.st-dark .auth-sub { color: #a8b2c2; }

/* ------------------ Glass Card ------------------ */
.glass-container {
    max-width: 440px;
    margin: 0 auto;
    padding: 20px;
}
.glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(25px) saturate(180%);
    -webkit-backdrop-filter: blur(25px) saturate(180%);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    padding: 32px 28px;
    box-shadow:
        0 22px 48px rgba(15,18,22,0.14),
        inset 0 1px 0 rgba(255,255,255,0.75);
    position: relative;
}
.glass-card::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 24px;
    background:
        radial-gradient(120px 60px at 90% 0%, #ffd6d8aa 0%, transparent 65%),
        radial-gradient(180px 80px at 0% 100%, #ffe6e8aa 0%, transparent 65%);
    pointer-events: none;
    z-index: -1;
}
.st-dark .glass-card {
    background: rgba(16,19,26,0.38);
    border-color: rgba(255,255,255,0.10);
    box-shadow:
        0 26px 60px rgba(0,0,0,.55),
        inset 0 1px 0 rgba(255,255,255,0.07);
}
.st-dark .glass-card::before {
    background:
        radial-gradient(130px 60px at 92% 0%, rgba(215,25,32,.30) 0%, transparent 70%),
        radial-gradient(200px 90px at -2% 102%, rgba(215,25,32,.18) 0%, transparent 72%);
}

/* ------------------ Input Fields ------------------ */
.stTextInput>div>div>input {
    border-radius: 14px !important;
    border: 1.6px solid #e6e8ef !important;
    padding: 0.78rem 0.95rem !important;
    background: #ffffff !important;
    color: #0f1216 !important;
    font-size: 0.95rem;
}
.stTextInput>div>div>input:focus {
    border-color: #D71920 !important;
    box-shadow: 0 0 0 4px #d7192033 !important;
}
.st-dark .stTextInput>div>div>input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e8eef6 !important;
}

/* ------------------ Buttons ------------------ */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid transparent;
    background: #D71920;
    color: #fff;
    font-weight: 800;
    padding: 0.85rem 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 14px 26px #d719201f, 0 2px 0 #b30f15 inset;
    margin-top: 1rem;
}
.stButton>button:hover {
    transform: translateY(-1px);
    background: #B9151B;
    box-shadow: 0 18px 30px #d719202a;
}

/* Special styling for link-style buttons */
.stButton > button.link-btn {
    background: none !important;
    border: none !important;
    color: #D71920 !important;
    font-weight: 600 !important;
    text-decoration: none;
    box-shadow: none !important;
    margin-top: 0.2rem !important;
    padding: 0 !important;
    width: auto !important;
}
.stButton > button.link-btn:hover {
    text-decoration: underline !important;
    color: #B9151B !important;
}

/* ------------------ Auth Footer ------------------ */
.auth-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
}
.st-dark .auth-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.auth-link {
    font-size: 0.9rem;
    font-weight: 500;
    color: #7b8595;
    cursor: pointer;
    transition: color 0.2s ease;
}
.auth-link:hover { color: #B9151B; }
.st-dark .auth-link { color: #a8b2c2; }
.st-dark .auth-link:hover { color: #D71920; }
.create-account-text {
    font-size: 0.9rem;
    color: #7b8595;
    text-align: right;
}
.st-dark .create-account-text { color: #a8b2c2; }

/* ------------------ Validation Messages ------------------ */
.msg-ok { color: #16a34a; font-weight: 600; margin: 0.5rem 0; }
.msg-warn { color: #d97706; font-weight: 600; margin: 0.5rem 0; }
.msg-err { color: #dc2626; font-weight: 600; margin: 0.5rem 0; }

.req-box {
    background: #fcfdff;
    border-left: 4px solid #D71920;
    padding: 0.6rem 0.75rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    line-height: 1.35;
}
.st-dark .req-box {
    background: rgba(255,255,255,0.06);
    color: #e8eef6;
}
.req-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)
    st.markdown("""
    <style>
      .auth-link { text-decoration: none; }
      .auth-link:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
<style>
/* --- Auth brand header (safe, not a <header> tag) --- */
.auth-brand{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:8px; margin-bottom:10px;
}

/* Image logo already has .minet-logo; keep it */
.minet-logo { display:block; margin:0 auto 6px; width:148px; height:auto;
  filter: drop-shadow(0 8px 18px rgba(215,25,32,.18));
}

/* Text fallback if the image is missing */
.minet-text-logo{
  font-weight:900; letter-spacing:.22rem; font-size:1.35rem;
  color:#D71920; text-align:center; line-height:1; margin-top:4px;
  text-transform:uppercase;
}

/* Optional small “subtitle” under the logo */
.minet-brand-sub{
  font-size:.92rem; color:#7b8595; text-align:center; margin-top:2px;
}
.st-dark .minet-brand-sub{ color:#a8b2c2; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Validation helpers -----------------------------
EMAIL_RE = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
def is_valid_email(email: str) -> bool:
    return bool(re.match(EMAIL_RE, email or ""))

def password_requirements(password: str):
    return {
        "length": len(password) >= 8,
        "uppercase": bool(re.search(r"[A-Z]", password or "")),
        "lowercase": bool(re.search(r"[a-z]", password or "")),
        "digit": bool(re.search(r"[0-9]", password or "")),
        "special": bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password or "")),
    }

def password_strength(password: str) -> str:
    req = password_requirements(password)
    score = sum(req.values())
    if score == 5: return "strong"
    if score >= 3: return "medium"
    return "weak"

# ----------------------------- UI pieces -----------------------------
def _password_requirements_html(req: dict) -> str:
    labels = {
        'length': 'At least 8 characters',
        'uppercase': 'Uppercase letter (A-Z)',
        'lowercase': 'Lowercase letter (a-z)',
        'digit': 'A number (0-9)',
        'special': 'Special char (!@#$%…)'
    }
    lines = []
    for k, label in labels.items():
        ok = req.get(k, False)
        mark = "✓" if ok else "○"
        lines.append(f'<div class="req-item">{mark} {label}</div>')
    return f'<div class="req-box">{"".join(lines)}</div>'

def _init_state():
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "login"

# ----------------------------- Screens -----------------------------
def login_screen(db: FraudDetectionDB):
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    render_auth_branding("MINET")  # shows image if present; otherwise a styled “MINET”
    st.markdown('<div class="auth-title">Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Sign in to continue</div>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email address", placeholder="you@company.com", key="li_email")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pw")

        if email and not is_valid_email(email):
            st.markdown('<div class="msg-err">Please enter a valid email</div>', unsafe_allow_html=True)

        submit = st.form_submit_button("Sign In")
        if submit:
            if not email or not password:
                st.error("Please fill in all fields.")
            elif not is_valid_email(email):
                st.error("Email address looks invalid.")
            else:
                user = db.verify_user(email, password)
                if user:
                    st.session_state["user"] = user
                    db.audit(user['email'], "user_login", {"ok": True})
                    st.success(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    db.audit(email, "user_login", {"ok": False})
                    st.error("Invalid email or password.")

    # --- Clean footer: true links (no hidden buttons) ---
    st.markdown("""
        <div class="auth-footer">
            <a class="auth-link" href="?auth=forgot">Forgot password?</a>
            <div class="create-account-text">
                Don't have an account?
                <a class="auth-link" href="?auth=signup">Create a new account</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# close glass-card & wrap
def signup_screen(db: FraudDetectionDB):
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown(_inline_logo_html(), unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create your account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Join the Minet fraud detection workspace</div>', unsafe_allow_html=True)

    name = st.text_input("Full name", key="su_name", placeholder="Jane Doe")
    email = st.text_input("Email address", key="su_email", placeholder="you@company.com")
    pw = st.text_input("Password", type="password", key="su_pw", placeholder="Use a strong password")

    req = password_requirements(pw)
    strength = password_strength(pw)
    if pw:
        if strength == "strong":
            st.markdown('<div class="msg-ok">✓ Password is strong</div>', unsafe_allow_html=True)
        elif strength == "medium":
            st.markdown('<div class="msg-warn">⚠ Could be stronger</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-err">✗ Password is weak</div>', unsafe_allow_html=True)
    
    st.markdown(_password_requirements_html(req), unsafe_allow_html=True)

    cpw = st.text_input("Confirm password", type="password", key="su_cpw", placeholder="Re-enter password")

    if cpw:
        if pw == cpw:
            st.markdown('<div class="msg-ok">Passwords match</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-err">Passwords do not match</div>', unsafe_allow_html=True)

    can_submit = (
        bool(name.strip()) and
        is_valid_email(email) and
        (password_strength(pw) == "strong") and
        (pw == cpw)
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Create account", disabled=not can_submit, use_container_width=True):
            created = db.create_user(email=email, name=name, password=pw, role="analyst")
            if created:
                db.audit(email, "user_signup", {"ok": True})
                st.success("Account created! Please sign in.")
                st.session_state.auth_tab = "login"
                st.rerun()
            else:
                db.audit(email, "user_signup", {"ok": False})
                st.error("An account with this email already exists.")
    with col2:
        if st.button("Back to sign in", key="back_to_login", use_container_width=True, type="secondary"):
            st.session_state.auth_tab = "login"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
def forgot_password_screen(db: FraudDetectionDB):
    st.markdown('<div class="glass-container"><div class="glass-card">', unsafe_allow_html=True)
    render_auth_branding("MINET")  # shows image if present; otherwise a styled “MINET”
    st.markdown('<div class="auth-title">Reset your password</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Enter the email you use to sign in</div>', unsafe_allow_html=True)

    # simple resend cooldown to avoid spam
    if "reset_cooldown_until" not in st.session_state:
        st.session_state.reset_cooldown_until = datetime.min

    email = st.text_input("Email address", placeholder="you@company.com", key="fp_email")

    disabled = datetime.utcnow() < st.session_state.reset_cooldown_until
    btn = st.button("Send reset link", type="primary", use_container_width=True, disabled=disabled)

    if btn:
        # Always show a generic success to avoid account enumeration
        ok = is_valid_email(email)
        try:
            if ok:
                # 1) Generate token + expiry
                token = secrets.token_urlsafe(32)
                expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MIN)

                # 2) Persist the reset request
                # Expected DB helpers (implement if missing):
                #   db.create_password_reset(email=email, token=token, expires_at=expires_at)
                #     - should silently no-op if email doesn't exist
                db.create_password_reset(email=email, token=token, expires_at=expires_at)
                
                # 3) Build reset URL (adjust to your app’s base URL)
                base = st.get_option("browser.serverAddress") or "http://localhost:8501"
                # Use the new query param API
                reset_url = f"{base}?reset={token}"

                # 4) Send the email (stub here – integrate SMTP/provider)
                #send_reset_email(to=email, link=reset_url)
                # For dev, show the link so you can click it immediately:
                SHOW_DEV_RESET_LINK = st.secrets.get("SHOW_DEV_RESET_LINK", "false").lower() == "true"
                if SHOW_DEV_RESET_LINK:
                    st.info("Development helper: click the link below to continue:")
                    st.code(reset_url, language="")
            
            st.success("If that email exists, a reset link has been sent.")
            # 30s cooldown
            st.session_state.reset_cooldown_until = datetime.utcnow() + timedelta(seconds=30)
        except Exception as e:
            st.error("We couldn’t create a reset link right now. Please try again.")
            st.caption(str(e))

    # Link back to login
    if st.button("Back to sign in", key="back_to_login_from_forgot", use_container_width=True):
        st.session_state.auth_tab = "login"
        st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
def reset_password_screen(db: FraudDetectionDB, token: str):
        st.markdown('<div class="glass-container"><div class="glass-card">', unsafe_allow_html=True)
        render_auth_branding("MINET")  # shows image if present; otherwise a styled “MINET”
        st.markdown('<div class="auth-title">Choose a new password</div>', unsafe_allow_html=True)

    # Expected DB helpers:
    #   db.lookup_password_reset(token) -> dict | None with {email, expires_at, used}
    #   db.consume_password_reset(token) -> marks as used
    #   db.update_user_password(email, new_password)
        rec = db.lookup_password_reset(token)

        if not rec:
            st.error("This reset link is invalid.")
        elif datetime.utcnow() > rec["expires_at"]:
            st.error("This reset link has expired. Please request a new one.")
        elif rec.get("used"):
         st.error("This reset link has already been used.")
        else:
            email = rec["email"]
            pw = st.text_input("New password", type="password", key="rp_pw")
            cpw = st.text_input("Confirm password", type="password", key="rp_cpw")

        # live feedback
            req = password_requirements(pw)
            st.markdown(_password_requirements_html(req), unsafe_allow_html=True)

            strong = password_strength(pw) == "strong"
            matches = pw and pw == cpw

            submit = st.button("Update password", type="primary", use_container_width=True, disabled=not(strong and matches))
            if submit:
                try:
                    db.update_user_password(email=email, new_password=pw)
                    db.consume_password_reset(token)
                    st.success("Your password has been updated. Please sign in.")
                    st.session_state.auth_tab = "login"
                    st.rerun()
                except Exception as e:
                    st.error("We couldn’t update your password. Please try again.")
                    st.caption(str(e))

        if st.button("Back to sign in", key="back_to_login_from_reset", use_container_width=True):
            st.session_state.auth_tab = "login"
            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)




# ----------------------------- Public API -----------------------------
def require_login():
    _route_from_query_param()
    apply_styles()
    #set_page_config()
    _init_state()
    # Add session state to track if styles are applied
    # if 'styles_applied' not in st.session_state:
    #     st.session_state.styles_applied = True
    #     # Force rerun to apply styles
    #     st.rerun()
    
    

    
    db = FraudDetectionDB()

    # Seed admin as you do now...
    if "seeded_admin" not in st.session_state:
        db.create_user("admin@minet.local", "Admin", "ChangeMe!123", role="admin")
        st.session_state.seeded_admin = True

    # If already logged in
    if st.session_state.get("user"):
        return st.session_state["user"]

    # --- NEW: handle query param routes with modern API ---
    

    # Normal auth flow
    if st.session_state.auth_tab == "login":
        login_screen(db)
    elif st.session_state.auth_tab == "forgot":
        forgot_password_screen(db)
    else:
        signup_screen(db)

    st.stop()



def logout():
    st.session_state.pop("user", None)
    st.rerun()
