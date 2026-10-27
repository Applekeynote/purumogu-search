using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace Purumogu {
    [Serializable] public sealed class SensorEvent {
        public string node_id, kind, peer_id = "";
        public long t_ms, node_time_ms;
        public double value, quality = 1;
    }
    [Serializable] public sealed class ObservationPacket {
        public string protocol_version="1.0", type="observation_batch", run_id, action_id="", formation_mode="", policy="";
        public long seq_no, sim_step, timestamp_ms;
        public long policy_seed;
        public List<SensorEvent> events=new List<SensorEvent>();
        public bool batch_end, accepted;
        public int entry_id,node_count,remaining_nodes;
    }
    public sealed class ActionCommand {
        public string action_id, formation_mode, policy;
        public int entry_id, node_count, delay_steps;
    }
    public static class Wire {
        public const int MaxLine=1048576, MaxEvents=2000;
        public static string Serialize(ObservationPacket p) => JsonConvert.SerializeObject(p);
    }
}
