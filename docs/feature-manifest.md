# Observation features

All features are built by `rust/crates/graph`. The model receives neither node coordinates nor a geometry label. IDs name graph vertices; their numeric spelling never enters the feature vector. The first 16 channels have 16 explicit presence-mask channels.

| Feature | Sensor source | Limitation |
|---|---|---|
| impact count / observed node | quantized pseudo-IMU threshold events | Not an exact contact count |
| impact mean | saturated body-frame acceleration magnitude | No wall identity or collision normal |
| still fraction | low acceleration variation and angular rate | Constant-speed motion can look still |
| radio count / node | received packets | Missing packets have multiple causes |
| mean and SD of RSSI-like value | noisy received signal | Not a perfect range measurement |
| largest component ratio | union of received radio edges in the batch | Temporal union can merge non-simultaneous connections |
| components / node | same graph | Not an estimate of the number of physical rooms |
| timeouts / node | last received peer timestamp | Does not identify the cause of loss |
| recontacts / node | reception after a timeout | May be radio noise |
| first impact delay | receiver timestamps | Also depends on the hopper |
| couplings / node | observed local coupling state transitions | Action context, not structural truth |
| mean quality | configured sensor confidence | A simulation assumption |
| missing ratio | dropped-packet observation-health events | Instrumentation loss proxy, not a peer-distance oracle |
| elapsed time | receiver clock | Nine seconds per probe |
| prior deployments / 20 | known actuator history | Coarse history only; latent friction remains shared across batches |

Entry and formation condition the learned likelihood table. Training, normalization, codebook learning and temperature selection never use test labels. Raw acceleration is generated at 50 Hz; only threshold events leave the simulation.

`OTHER` uses a uniform code distribution and a support-distance check. It is a conservative rejection aid, not a calibrated open-world detector. The UI says so.
