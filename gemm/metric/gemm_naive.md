M=1024, N=1024, K=1024

==PROF== Connected to process 828 (/tmp/tmplm3nr6as/fc91237c-6a5d-4956-8db7-fef5e52fd7a3/cuda_exec.out)

==PROF== Profiling "sgemm" - 0: 0%....50%....100% - 8 passes

==PROF== Disconnected from process 828

[828] cuda_exec.out@127.0.0.1

  sgemm(int, int, int, float, float, const float *, const float *, float *) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 7.5
  
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          5.00
    SM Frequency            cycle/usecond        584.99
    Elapsed Cycles                  cycle    56,344,784
    Memory Throughput                   %         49.23
    DRAM Throughput                     %          0.47
    Duration                      msecond         96.32
    L1/TEX Cache Throughput             %         98.45
    L2 Cache Throughput                 %          0.40
    SM Active Cycles                cycle 55,313,213.62
    Compute (SM) Throughput             %          5.97
    ----------------------- ------------- -------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance 
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate    
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.                 

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                 1,024
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  1,024
    Registers Per Thread             register/thread              49
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread       1,048,576
    Waves Per SM                                               25.60
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            1
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            1
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        97.95
    Achieved Active Warps Per SM           warp        31.35
    ------------------------------- ----------- ------------

    INF   This kernel's theoretical occupancy is not impacted by any block limit.  
