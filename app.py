from flask import Flask
import pickle


class Models:
    network_traffics="./TrainedModel/"
    permission="./TrainedModel/"

app=Flask(__name__)


@app.route("/")
def home():
    return "Welcome"

@app.route("/api/get/permissions")
def permission():
    return "permissions"

@app.route("/api/get/network_traffics")
def traffics():
    return "getting traffics"


