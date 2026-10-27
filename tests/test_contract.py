"""Real TCP fault tests and a measured replay regression, using only the stdlib."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "rust/target/release" / ("ghost_flow_server.exe" if os.name == "nt" else "ghost_flow_server")

def packet(kind="hello", seq=0, **kwargs):
    return dict(protocol_version="1.0", type=kind, run_id="contract-test", seq_no=seq,
                sim_step=1, timestamp_ms=20, policy="fixed", policy_seed=123, **kwargs)

class WireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0)); cls.port=probe.getsockname()[1]
        cls.process=subprocess.Popen([str(BIN), "serve", "--port", str(cls.port), "--model", str(ROOT / "models/m-v1/model.json"), "--logs", cls.temp.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            try:
                with socket.create_connection(("127.0.0.1",cls.port),timeout=.2):break
            except OSError:time.sleep(.05)
        else:
            cls.process.terminate();_,error=cls.process.communicate(timeout=5);cls.temp.cleanup()
            raise RuntimeError("Server failed to start: "+error.decode(errors='replace'))

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.communicate(timeout=10);cls.temp.cleanup()

    def connect(self):
        s=socket.create_connection(("127.0.0.1",self.port),timeout=5)
        self.addCleanup(s.close)
        f=s.makefile("rwb");self.addCleanup(f.close)
        return s,f

    def send(self,f,p):f.write((json.dumps(p,allow_nan=False)+"\n").encode());f.flush()
    def receive(self,f):return json.loads(f.readline())

    def test_handshake_actuator_and_duplicate_observation(self):
        _,f=self.connect();self.send(f,packet());self.assertEqual(self.receive(f)["type"],"ready");action=self.receive(f)
        self.send(f,packet("action_ack",1,action_id=action["action_id"],accepted=True));self.assertEqual(self.receive(f)["type"],"ack")
        obs=packet("observation_batch",2,action_id=action["action_id"],entry_id=action["entry_id"],formation_mode=action["formation_mode"],node_count=5,batch_end=True,events=[dict(node_id="N0",t_ms=0,node_time_ms=0,kind="NODE_SPAWN",quality=1),dict(node_id="N0",t_ms=20,node_time_ms=20,kind="IMPACT",value=12,quality=1)])
        self.send(f,obs);first=[self.receive(f),self.receive(f)]
        self.send(f,obs);duplicate=[self.receive(f),self.receive(f)]
        self.assertEqual(first,duplicate);self.assertEqual(first[0]["batch"],0);self.assertEqual(first[1]["action_id"],"a-001")

    def test_truth_unknown_enum_version_and_future_are_rejected(self):
        invalid=[packet(world_position=[0,0,0]),packet(),packet("heartbeat"),packet()]
        invalid[1]["protocol_version"]="9.0"
        invalid[2]["events"]=[dict(node_id="N0",t_ms=21_000,node_time_ms=0,kind="IMPACT",quality=1)]
        invalid[3]["events"]=[dict(node_id="N0",t_ms=0,node_time_ms=0,kind="PERFECT_ROOM",quality=1)]
        for p in invalid:
            with self.subTest(p=p):
                _,f=self.connect();self.send(f,p);error=self.receive(f);self.assertEqual(error["type"],"error");self.assertTrue(error["safe_stop"])

    def test_oversized_frame_closes_without_killing_listener(self):
        s,f=self.connect()
        try:s.sendall(b" "*(1_048_576+1)+b"\n")
        except OSError:pass
        try:self.assertEqual(f.readline(),b"")
        except (ConnectionResetError,ConnectionAbortedError):pass
        _,next_client=self.connect();self.send(next_client,packet());self.assertEqual(self.receive(next_client)["type"],"ready")

    def test_measured_replay(self):
        replay=subprocess.run([str(BIN),"replay","--model",str(ROOT/"models/m-v1/model.json"),"--events",str(ROOT/"datasets/sample.jsonl")],capture_output=True,text=True,check=True)
        got=[json.loads(line) for line in replay.stdout.splitlines()]
        expected=json.loads((ROOT/"datasets/sample-expected.json").read_text())
        self.assertEqual(len(got),3)
        for actual,want in zip(got,expected):
            self.assertEqual(actual["abstain"],want["abstain"])
            self.assertEqual(actual["batch"],want["batch"])
            for a,b in zip(actual["beliefs"],want["beliefs"]):self.assertAlmostEqual(a["probability"],b["probability"],places=12)

if __name__=="__main__":unittest.main(verbosity=2)
