import nox

@nox.session(python="3.11", reuse_venv=True)
def create_env(session):
    session.install("-r", "requirements.txt")

@nox.session(python="3.11", reuse_venv=True)
def remove_similar_frames(session):
    session.install("-r", "requirements.txt")
    session.run("python", "-m", "remove_similar_frames", *session.posargs)

@nox.session(python="3.11", tags=["style"], reuse_venv=True)
def black(session):
    session.install("black")
    session.run("black", "similar_frames_remover")

@nox.session(python="3.11", tags=["style"], reuse_venv=True)
def isort(session):
    session.install("isort")
    session.run("isort", "similar_frames_remover")

@nox.session(python="3.11", tags=["style"], reuse_venv=True)
def flake8(session):
    session.install("flake8")
    session.run("flake8", "similar_frames_remover")