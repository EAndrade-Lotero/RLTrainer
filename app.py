from flask import Flask, jsonify, render_template, request

from rl.trainer import trainer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(trainer.state_snapshot())


@app.route("/api/control/start", methods=["POST"])
def api_start():
    trainer.start()
    return jsonify({"ok": True})


@app.route("/api/control/pause", methods=["POST"])
def api_pause():
    trainer.pause()
    return jsonify({"ok": True})


@app.route("/api/control/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    start_position = data.get("start_position")
    side_to_move = data.get("side_to_move")
    trainer.reset(start_position=start_position, side_to_move=side_to_move)
    return jsonify({"ok": True})


@app.route("/api/control/speed", methods=["POST"])
def api_speed():
    data = request.get_json(silent=True) or {}
    trainer.set_speed(data.get("value", 8))
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
