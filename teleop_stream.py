"""Local-only joint stream shared by the hardware and Isaac processes."""
import json
import math
import socket
import time

HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, 0.0]


def franka_target(q, wrist_roll_offset_deg):
    """Map leader angles to Franka angles without modifying the input or stale state."""
    if q is None:
        return None
    result = list(q)
    result[6] += math.radians(wrist_roll_offset_deg)
    return result


def decode_packet(payload, now, timeout):
    """Reject malformed or old packets (both processes use this host's monotonic clock)."""
    try:
        packet = json.loads(payload)
        q = packet['q']
        stamp = float(packet['time'])
        if packet['version'] != 1 or len(q) != 7:
            return None
        q = [float(value) for value in q]
        if not all(math.isfinite(value) for value in q) or not math.isfinite(stamp):
            return None
        if not 0 <= now - stamp <= timeout:
            return None
        return stamp, q
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


class Publisher:
    def __init__(self, port=5557, hz=60):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.address = ('127.0.0.1', port)
        self.period = 1 / hz
        self.previous = -math.inf

    def send(self, q):
        now = time.monotonic()
        if now - self.previous < self.period:
            return
        payload = json.dumps({'version': 1, 'time': now, 'q': list(map(float, q))},
                             allow_nan=False).encode()
        try:
            self.socket.sendto(payload, self.address)
        except BlockingIOError:
            return  # Never stall the motor loop to wait for the simulator.
        self.previous = now

    def close(self):
        self.socket.close()


class Receiver:
    def __init__(self, port=5557, timeout=.25):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.setblocking(False)
        self.timeout = timeout
        self.stamp = -math.inf
        self.q = None

    def poll(self):
        now = time.monotonic()
        for _ in range(256):
            try:
                payload = self.socket.recv(4096)
            except BlockingIOError:
                break
            result = decode_packet(payload, now, self.timeout)
            if result is not None and result[0] > self.stamp:
                self.stamp, self.q = result
        return self.q if now - self.stamp <= self.timeout else None

    def close(self):
        self.socket.close()


def advance_target(previous, desired, lower, upper, dt, speed):
    """Apply joint position and per-step velocity limits; freeze on a stale stream."""
    if desired is None:
        return list(previous)
    result = []
    for old, goal, lo, hi in zip(previous, desired, lower, upper):
        goal = min(max(goal, lo), hi)
        result.append(min(max(old + min(max(goal - old, -speed * dt), speed * dt), lo), hi))
    return result
