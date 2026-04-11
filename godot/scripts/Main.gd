extends Node

const WS_URL := "ws://localhost:9876"

@onready var status_label: Label = $UI/VBox/StatusLabel
@onready var log_label: RichTextLabel = $UI/VBox/LogScroll/LogLabel
@onready var do_nothing_button: Button = $UI/VBox/DoNothingButton

var ws := WebSocketPeer.new()
var connected := false


func _ready() -> void:
	status_label.text = "Connecting to %s..." % WS_URL
	do_nothing_button.disabled = true
	do_nothing_button.pressed.connect(_on_do_nothing_pressed)
	var err := ws.connect_to_url(WS_URL)
	if err != OK:
		_log("Erreur connect_to_url: %d" % err)


func _process(_delta: float) -> void:
	ws.poll()
	var state := ws.get_ready_state()
	match state:
		WebSocketPeer.STATE_OPEN:
			if not connected:
				connected = true
				status_label.text = "Connected"
				do_nothing_button.disabled = false
			while ws.get_available_packet_count() > 0:
				_handle_packet(ws.get_packet())
		WebSocketPeer.STATE_CLOSED:
			if connected:
				connected = false
				status_label.text = "Disconnected (code %d)" % ws.get_close_code()
				do_nothing_button.disabled = true
			set_process(false)


func _handle_packet(pkt: PackedByteArray) -> void:
	var text := pkt.get_string_from_utf8()
	var data = JSON.parse_string(text)
	if data == null:
		_log("[color=red]JSON invalide:[/color] " + text)
		return
	var msg_type: String = data.get("type", "?")
	_log("[b]← %s[/b]" % msg_type)
	match msg_type:
		"init":
			_log("  seed=%s  size=%s  model_loaded=%s" % [data.seed, data.size, data.model_loaded])
			_log("  catalog: %d bâtiments" % data.buildings_catalog.size())
		"state":
			var st = data.state
			_log("  turn=%s  denarii=%s  pop=%s  sat=%.2f" % [st.turn, st.denarii, st.total_population, st.global_satisfaction])
			_log("  done=%s  victory=%s" % [data.done, data.victory])
		"error":
			_log("  [color=red]%s[/color]" % data.message)
		"ack":
			_log("  action=%s  ok=%s" % [data.action, data.ok])


func _on_do_nothing_pressed() -> void:
	if not connected:
		return
	var msg := {"type": "action", "action": {"type": "do_nothing"}}
	ws.send_text(JSON.stringify(msg))
	_log("[b]→ action do_nothing[/b]")


func _log(line: String) -> void:
	log_label.append_text(line + "\n")
