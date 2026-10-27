using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace Purumogu {
    /// This type does not reference UnityEngine. Its worker may only enqueue data.
    public sealed class UnityRustBridge:IDisposable {
        public readonly ConcurrentQueue<JObject> Incoming=new ConcurrentQueue<JObject>();
        readonly ConcurrentQueue<string> outgoing=new ConcurrentQueue<string>();
        readonly CancellationTokenSource cancel=new CancellationTokenSource();
        TcpClient client;Task worker;
        public volatile bool Connected;
        public string Failure {get;private set;}
        int queued;
        public int MaxFrameBytes=Wire.MaxLine;
        public void Start(int port){worker=Task.Run(()=>Run(port));}
        public void Send(ObservationPacket p)=>SendJson(Wire.Serialize(p));
        public void SendJson(string line){if(Encoding.UTF8.GetByteCount(line)>MaxFrameBytes)throw new InvalidDataException("wire line too large");if(Interlocked.Increment(ref queued)>128){Interlocked.Decrement(ref queued);throw new IOException("outgoing queue full");}outgoing.Enqueue(line);}
        async Task Run(int port){
            try{
                for(int retry=0;retry<4;retry++){try{client=new TcpClient();await client.ConnectAsync("127.0.0.1",port);break;}catch when(retry<3){client?.Dispose();await Task.Delay(250*(1<<retry),cancel.Token);}}
                client.NoDelay=true;Connected=true;var stream=client.GetStream();
                var reader=Task.Run(async()=>{var buffer=new byte[8192];var line=new MemoryStream();while(!cancel.IsCancellationRequested){int n=await stream.ReadAsync(buffer,0,buffer.Length,cancel.Token);if(n==0)throw new EndOfStreamException("Rust disconnected");for(int i=0;i<n;i++){if(buffer[i]==10){var obj=JObject.Parse(Encoding.UTF8.GetString(line.ToArray()));line.SetLength(0);if(Incoming.Count>=128)throw new IOException("incoming queue full");Incoming.Enqueue(obj);}else{if(line.Length>=MaxFrameBytes)throw new InvalidDataException("incoming line too large");line.WriteByte(buffer[i]);}}}},cancel.Token);
                while(!cancel.IsCancellationRequested){if(reader.IsCompleted){await reader;break;}if(outgoing.TryDequeue(out string text)){Interlocked.Decrement(ref queued);byte[] bytes=Encoding.UTF8.GetBytes(text+"\n");await stream.WriteAsync(bytes,0,bytes.Length,cancel.Token);}else await Task.Delay(2,cancel.Token);}
            }catch(Exception e){if(!cancel.IsCancellationRequested)Failure=e.Message;}finally{Connected=false;client?.Dispose();}
        }
        public void Dispose(){cancel.Cancel();client?.Dispose();}
    }
}
