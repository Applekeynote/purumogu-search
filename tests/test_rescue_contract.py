"""Protocol 2 integration checks against the real loopback Rust server (stdlib only)."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'rust/target/release' / ('rescue_server.exe' if os.name == 'nt' else 'rescue_server')


def packet(seq=0, step=25, full=True):
    return dict(protocol_version='2.0', run_id='wire-rescue', seq_no=seq, sim_step=step,
                cell_m=.05, request_map=full, scans=[dict(
                    node_id='N00', deployment_anchor=[0., .025, 0.],
                    odometry_delta=[0., 0., 0.], pose_sigma_m=.001,
                    rays=[dict(direction=[0., 0., 1.], distance_m=.4, hit=True)],
                    bearing=[0., 0., 1.], bearing_sigma_deg=10., acoustic_snr_db=5.,
                    voice_band_ratio=.6, modulation=.6, temperature_delta_c=.2)])


class RescueWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); cls.port = probe.getsockname()[1]
        cls.process = subprocess.Popen([str(BIN), '--port', str(cls.port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError('Rescue server failed to start')
            try:
                with socket.create_connection(('127.0.0.1', cls.port), timeout=.2):
                    return
            except OSError:
                time.sleep(.05)
        cls.process.terminate(); cls.process.wait(timeout=5)
        raise TimeoutError('Rescue server did not listen')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate(); cls.process.wait(timeout=10)

    def connect(self):
        sock = socket.create_connection(('127.0.0.1', self.port), timeout=5)
        self.addCleanup(sock.close)
        stream = sock.makefile('rwb'); self.addCleanup(stream.close)
        return sock, stream

    def send(self, stream, value):
        stream.write(json.dumps(value, allow_nan=False).encode() + b'\n'); stream.flush()

    def receive(self, stream):
        line = stream.readline(8_388_609)
        self.assertTrue(line); self.assertLessEqual(len(line), 8_388_608)
        return json.loads(line)

    def test_fragmented_frame_and_identical_duplicate(self):
        sock, stream = self.connect(); data = json.dumps(packet()).encode() + b'\n'
        for index in range(0, len(data), 7):
            sock.sendall(data[index:index+7])
        first = self.receive(stream)
        self.assertEqual(first['seq_no'], 0); self.assertTrue(first['cells'])
        self.send(stream, packet()); self.assertEqual(self.receive(stream), first)
        self.assertEqual(first['cues'], [])  # One robot cannot confirm a person.

    def test_compact_then_full_preserves_accumulated_map(self):
        _, stream = self.connect(); self.send(stream, packet(full=False))
        compact = self.receive(stream); self.assertEqual(compact['cells'], [])
        self.send(stream, packet(seq=1, step=50)); full = self.receive(stream)
        self.assertTrue(full['cells']); self.assertGreater(full['observed_cells'], 0)

    def test_environment_and_fine_surface_are_observed_data_with_independent_units(self):
        _,stream=self.connect();p=packet();p['request_surface']=True
        p['scans'][0].update(gas_level=.8,water_depth_m=.025)
        second=copy.deepcopy(p['scans'][0]);second['node_id']='N01';second['deployment_anchor'][0]=.03;p['scans'].append(second)
        self.send(stream,p);first=self.receive(stream)
        self.assertEqual({h['kind'] for h in first['hazards']},{'gas','water'})
        self.assertEqual(first['gas_observed_nodes'],2);self.assertEqual(first['water_observed_nodes'],2)
        p['seq_no']=1;p['sim_step']=50;self.send(stream,p);second=self.receive(stream)
        self.assertEqual(second['surface_resolution_m'],.01);self.assertTrue(second['surface_points'])
        self.assertTrue(all(len(row)==6 for row in second['surface_points']))
        for field,value in [('gas_level',1.1),('water_depth_m',-.01)]:
            sock,bad_stream=self.connect();invalid=packet();invalid['scans'][0][field]=value;self.send(bad_stream,invalid)
            self.assertTrue(self.receive(bad_stream)['safe_stop']);bad_stream.close();sock.close()

    def test_truth_version_and_sequence_faults_stop_only_the_session(self):
        invalid = []
        p = packet(); p['world_position'] = [0, 0, 0]; invalid.append(p)
        p = packet(); p['scans'][0]['is_human'] = True; invalid.append(p)
        p = packet(); p['scans'][0]['rays'][0]['ground_truth'] = True; invalid.append(p)
        p = packet(); p['protocol_version'] = '1.0'; invalid.append(p)
        invalid.append(packet(seq=1))
        for value in invalid:
            with self.subTest(value=value):
                sock, stream = self.connect(); self.send(stream, value)
                error = self.receive(stream)
                self.assertEqual(error['type'], 'error'); self.assertTrue(error['safe_stop'])
                self.assertEqual(stream.readline(), b''); stream.close(); sock.close()
        _, stream = self.connect(); self.send(stream, packet())
        self.assertEqual(self.receive(stream)['protocol_version'], '2.0')

    def test_changed_duplicate_and_gap_are_rejected(self):
        for change in ('payload', 'gap', 'clock'):
            with self.subTest(change=change):
                sock, stream = self.connect(); self.send(stream, packet()); self.receive(stream)
                p = copy.deepcopy(packet())
                if change == 'payload': p['scans'][0]['temperature_delta_c'] = 1.
                if change == 'gap': p['seq_no'] = 2; p['sim_step'] = 50
                if change == 'clock': p['seq_no'] = 1; p['sim_step'] = 526
                self.send(stream, p); self.assertTrue(self.receive(stream)['safe_stop'])
                stream.close(); sock.close()

    def test_oversized_frame_does_not_kill_listener(self):
        sock, stream = self.connect()
        try: sock.sendall(b' ' * 4_194_305 + b'\n')
        except OSError: pass
        try: self.assertEqual(stream.readline(), b'')
        except (ConnectionResetError, ConnectionAbortedError): pass
        stream.close(); sock.close()
        _, stream = self.connect(); self.send(stream, packet())
        self.assertEqual(self.receive(stream)['seq_no'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
