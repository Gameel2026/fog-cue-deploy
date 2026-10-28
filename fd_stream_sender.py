import sys
import time
import socket
import struct
import threading
import numpy as np

PORT = 50007
PKT_SAMPLES = 4


def main():
    replay, minutes, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    target = sys.argv[4] if len(sys.argv) > 4 else None
    with open(replay, "rb") as f:
        assert f.read(4) == b"FOGR"
        n, _ = struct.unpack("<2i", f.read(8))
        raw = np.frombuffer(f.read(8 * 3 * n), np.float64).reshape(n, 3)
    fs = float(sys.argv[sys.argv.index("--fs") + 1]) if "--fs" in sys.argv else 100.0
    if target:
        s = socket.create_connection((target, PORT))
    else:
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("0.0.0.0", PORT)); srv.listen(1)
        print(f"waiting for the phone on port {PORT} ...", flush=True)
        s, _ = srv.accept(); srv.close()
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    total = int(minutes * 60 * fs) // PKT_SAMPLES
    sent_at = {}
    replies = []

    def reader():
        buf = b""
        while True:
            try:
                d = s.recv(4096)
            except OSError:
                break
            if not d:
                break
            buf += d
            while len(buf) >= 13:
                k, on, p = struct.unpack("<IBd", buf[:13]); buf = buf[13:]
                replies.append((k, on, p, time.perf_counter()))

    th = threading.Thread(target=reader, daemon=True); th.start()
    t0 = time.perf_counter(); period = PKT_SAMPLES / fs
    for seq in range(total):
        tgt = t0 + (seq + 1) * period
        while True:
            dt = tgt - time.perf_counter()
            if dt <= 0:
                break
            time.sleep(min(dt, 0.002))
        i = (seq * PKT_SAMPLES) % (n - PKT_SAMPLES)
        s.sendall(struct.pack("<I", seq) + raw[i:i + PKT_SAMPLES].astype("<f8").tobytes())
        sent_at[seq] = time.perf_counter()
        if seq % 1500 == 0:
            print(f"sent {seq}/{total} packets, {len(replies)} decisions back", flush=True)
    time.sleep(2.0)
    s.shutdown(socket.SHUT_WR); th.join(timeout=5); s.close()
    win = int(sys.argv[sys.argv.index("--win") + 1]) if "--win" in sys.argv else 50
    with open(out, "w") as f:
        f.write("window,cue_on,p,reply_s,last_packet_sent_s,round_trip_ms\n")
        for k, on, p, tr in replies:
            last_seq = ((k + 1) * win - 1) // PKT_SAMPLES
            ts = sent_at.get(last_seq, np.nan)
            f.write(f"{k},{on},{p:.12f},{tr - t0:.6f},{ts - t0:.6f},{1e3 * (tr - ts):.3f}\n")
    print(f"done: {total} packets sent, {len(replies)} decisions received -> {out}", flush=True)


if __name__ == "__main__":
    main()
