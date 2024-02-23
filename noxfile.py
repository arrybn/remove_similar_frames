import nox

@nox.session(python="3.11", reuse_venv=True)
def create_session(session):
    session.install("-r", "requirements.txt")