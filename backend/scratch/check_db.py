import socket

def check_port(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect((host, port))
        print(f"Port {port} on {host} is open!")
        s.close()
        return True
    except Exception as e:
        print(f"Port {port} on {host} is closed: {e}")
        return False

if __name__ == "__main__":
    check_port("127.0.0.1", 5432)
    check_port("localhost", 5432)
