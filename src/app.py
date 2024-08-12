# src/app.py
from flask import Flask, request, render_template
from rag_pipeline import rag

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        response = rag.query(query)
        return render_template("index.html", query=query, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
