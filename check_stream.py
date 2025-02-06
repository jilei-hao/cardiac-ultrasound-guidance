import socket

def check_stream(host, port):
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            print(f"✅ Successfully connected to {host}:{port}")
            data = sock.recv(1024)
            if data:
                print("✅ Stream data is being received.")
            else:
                print("❌ No data received from the stream.")
    except Exception as e:
        print(f"❌ Error: {e}")

check_stream("127.0.0.1", 8080)