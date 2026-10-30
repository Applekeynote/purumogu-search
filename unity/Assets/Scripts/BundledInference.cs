using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;

namespace Purumogu.Rescue
{
    /// Owns exactly one packaged process. The Windows job also closes it after a forced player exit.
    public sealed class BundledInference : IDisposable
    {
        Process process;
        IntPtr job;
        public int Port { get; private set; }
        public int ProcessId => process?.Id ?? 0;
        public bool Alive => process != null && !process.HasExited;

        public void Start(string folder)
        {
            if (process != null) throw new InvalidOperationException("Inference already started");
            string executable=Path.Combine(folder,"rescue_server.exe"),model=Path.Combine(folder,"model.json");
            if(!File.Exists(executable)||!File.Exists(model))
                throw new FileNotFoundException("推論ファイルがありません。配布ZIPをすべて展開してください。",folder);
            var probe=new TcpListener(IPAddress.Loopback,0);
            try {probe.Start();Port=((IPEndPoint)probe.LocalEndpoint).Port;} finally {probe.Stop();}
            job=CreateJobObject(IntPtr.Zero,null);
            if(job==IntPtr.Zero)throw new System.ComponentModel.Win32Exception();
            var limits=new ExtendedLimits();limits.Basic.LimitFlags=0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if(!SetInformationJobObject(job,9,ref limits,(uint)Marshal.SizeOf(typeof(ExtendedLimits))))
            {Dispose();throw new System.ComponentModel.Win32Exception();}
            try
            {
                process=Process.Start(new ProcessStartInfo(executable,"--port "+Port+" --model \""+model+"\"")
                {UseShellExecute=false,CreateNoWindow=true,WorkingDirectory=folder,WindowStyle=ProcessWindowStyle.Hidden});
                if(process==null||!AssignProcessToJobObject(job,process.Handle))throw new InvalidOperationException("推論プロセスの管理を開始できませんでした。");
            }
            catch {Dispose();throw;}
        }
        public void Dispose()
        {
            if(job!=IntPtr.Zero){CloseHandle(job);job=IntPtr.Zero;}
            if(process!=null)
            {
                try {if(!process.HasExited){process.Kill();process.WaitForExit(3000);}}catch(InvalidOperationException){}
                finally {process.Dispose();process=null;}
            }
        }
        [StructLayout(LayoutKind.Sequential)] struct BasicLimits
        {public long ProcessTime,JobTime;public uint LimitFlags;public UIntPtr MinWorking,MaxWorking;public uint ActiveProcesses;public UIntPtr Affinity;public uint Priority,Scheduling;}
        [StructLayout(LayoutKind.Sequential)] struct IoCounters
        {public ulong ReadOperations,WriteOperations,OtherOperations,ReadBytes,WriteBytes,OtherBytes;}
        [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits
        {public BasicLimits Basic;public IoCounters Io;public UIntPtr ProcessMemory,JobMemory,PeakProcessMemory,PeakJobMemory;}
        [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attributes,string name);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr handle,int info,ref ExtendedLimits value,uint length);
        [DllImport("kernel32.dll",SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr handle,IntPtr processHandle);
        [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    }
}
