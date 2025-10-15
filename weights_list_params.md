```
Total number of valid dims combinations: 18

================================================================================
STEP-BY-STEP PARAMETER CALCULATION
================================================================================

1. MULTIPOSTERIORS
----------------------------------------
  dims=[0, 1, 0, 0, 0]: shape=[9, 4], mean=36, local_cap=36, total=72
  dims=[1, 1, 0, 0, 0]: shape=[9, 9, 4], mean=324, local_cap=324, total=648
  dims=[0, 0, 1, 0, 0]: shape=[8, 4], mean=32, local_cap=32, total=64
  dims=[1, 0, 1, 0, 0]: shape=[9, 8, 4], mean=288, local_cap=288, total=576
  dims=[0, 1, 1, 0, 0]: shape=[9, 8, 4], mean=288, local_cap=288, total=576
  dims=[1, 1, 1, 0, 0]: shape=[9, 9, 8, 4], mean=2592, local_cap=2592, total=5184
  dims=[1, 0, 0, 1, 0]: shape=[9, 30, 4], mean=1080, local_cap=1080, total=2160
  dims=[1, 1, 0, 1, 0]: shape=[9, 9, 30, 4], mean=9720, local_cap=9720, total=19440
  dims=[1, 0, 1, 1, 0]: shape=[9, 8, 30, 4], mean=8640, local_cap=8640, total=17280
  dims=[1, 1, 1, 1, 0]: shape=[9, 9, 8, 30, 4], mean=77760, local_cap=77760, total=155520
  dims=[1, 0, 0, 0, 1]: shape=[9, 30, 4], mean=1080, local_cap=1080, total=2160
  dims=[1, 1, 0, 0, 1]: shape=[9, 9, 30, 4], mean=9720, local_cap=9720, total=19440
  dims=[1, 0, 1, 0, 1]: shape=[9, 8, 30, 4], mean=8640, local_cap=8640, total=17280
  dims=[1, 1, 1, 0, 1]: shape=[9, 9, 8, 30, 4], mean=77760, local_cap=77760, total=155520
  dims=[1, 0, 0, 1, 1]: shape=[9, 30, 30, 4], mean=32400, local_cap=32400, total=64800
  dims=[1, 1, 0, 1, 1]: shape=[9, 9, 30, 30, 4], mean=291600, local_cap=291600, total=583200
  dims=[1, 0, 1, 1, 1]: shape=[9, 8, 30, 30, 4], mean=259200, local_cap=259200, total=518400
  dims=[1, 1, 1, 1, 1]: shape=[9, 9, 8, 30, 30, 4], mean=2332800, local_cap=2332800, total=4665600
  Subtotal params: 6227920
  Subtotal weights: 36

2. DECODE_WEIGHTS
----------------------------------------
  dims=[0, 1, 0, 0, 0]: weight=4x16=64, bias=16, total=80
  dims=[1, 1, 0, 0, 0]: weight=4x16=64, bias=16, total=80
  dims=[0, 0, 1, 0, 0]: weight=4x8=32, bias=8, total=40
  dims=[1, 0, 1, 0, 0]: weight=4x8=32, bias=8, total=40
  dims=[0, 1, 1, 0, 0]: weight=4x8=32, bias=8, total=40
  dims=[1, 1, 1, 0, 0]: weight=4x8=32, bias=8, total=40
  dims=[1, 0, 0, 1, 0]: weight=4x16=64, bias=16, total=80
  dims=[1, 1, 0, 1, 0]: weight=4x16=64, bias=16, total=80
  dims=[1, 0, 1, 1, 0]: weight=4x8=32, bias=8, total=40
  dims=[1, 1, 1, 1, 0]: weight=4x8=32, bias=8, total=40
  dims=[1, 0, 0, 0, 1]: weight=4x16=64, bias=16, total=80
  dims=[1, 1, 0, 0, 1]: weight=4x16=64, bias=16, total=80
  dims=[1, 0, 1, 0, 1]: weight=4x8=32, bias=8, total=40
  dims=[1, 1, 1, 0, 1]: weight=4x8=32, bias=8, total=40
  dims=[1, 0, 0, 1, 1]: weight=4x16=64, bias=16, total=80
  dims=[1, 1, 0, 1, 1]: weight=4x16=64, bias=16, total=80
  dims=[1, 0, 1, 1, 1]: weight=4x8=32, bias=8, total=40
  dims=[1, 1, 1, 1, 1]: weight=4x8=32, bias=8, total=40
  Subtotal params: 1040
  Subtotal weights: 36

3. TARGET_CAPACITIES
----------------------------------------
  dims=[0, 1, 0, 0, 0]: shape=[4], params=4
  dims=[1, 1, 0, 0, 0]: shape=[4], params=4
  dims=[0, 0, 1, 0, 0]: shape=[4], params=4
  dims=[1, 0, 1, 0, 0]: shape=[4], params=4
  dims=[0, 1, 1, 0, 0]: shape=[4], params=4
  dims=[1, 1, 1, 0, 0]: shape=[4], params=4
  dims=[1, 0, 0, 1, 0]: shape=[4], params=4
  dims=[1, 1, 0, 1, 0]: shape=[4], params=4
  dims=[1, 0, 1, 1, 0]: shape=[4], params=4
  dims=[1, 1, 1, 1, 0]: shape=[4], params=4
  dims=[1, 0, 0, 0, 1]: shape=[4], params=4
  dims=[1, 1, 0, 0, 1]: shape=[4], params=4
  dims=[1, 0, 1, 0, 1]: shape=[4], params=4
  dims=[1, 1, 1, 0, 1]: shape=[4], params=4
  dims=[1, 0, 0, 1, 1]: shape=[4], params=4
  dims=[1, 1, 0, 1, 1]: shape=[4], params=4
  dims=[1, 0, 1, 1, 1]: shape=[4], params=4
  dims=[1, 1, 1, 1, 1]: shape=[4], params=4
  Subtotal params: 72
  Subtotal weights: 18

================================================================================
LAYER 1
================================================================================

4.1. SHARE_UP_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

5.1. SHARE_DOWN_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[0, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[0, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  Subtotal params: 3680
  Subtotal weights: 72

6.1. SOFTMAX_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[0, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[1, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[0, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 0, 0, 1, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 1, 0]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 1, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 1, 0]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 0, 1]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 0, 1]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 1, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 1, 0, 1, 1]: output_scaling=14, L1=16x2+2, L2=14x16+16, total=274
  dims=[1, 0, 1, 1, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 1, 1, 1, 1]: output_scaling=30, L1=8x2+2, L2=30x8+8, total=266
  Subtotal params: 2100
  Subtotal weights: 72

7.1. CUMMAX_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

8.1. SHIFT_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

9.1. DIRECTION_SHARE_WEIGHTS (Layer 1)
----------------------------------------
  Each dims has 8x8=64 linear maps, each 16x16 + bias
  Subtotal params: 185344
  Subtotal weights: 2304

10.1. NONLINEAR_WEIGHTS (Layer 1)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

================================================================================
LAYER 2
================================================================================

4.2. SHARE_UP_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

5.2. SHARE_DOWN_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[0, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[0, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  Subtotal params: 3680
  Subtotal weights: 72

6.2. SOFTMAX_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[0, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[1, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[0, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 0, 0, 1, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 1, 0]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 1, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 1, 0]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 0, 1]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 0, 1]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 1, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 1, 0, 1, 1]: output_scaling=14, L1=16x2+2, L2=14x16+16, total=274
  dims=[1, 0, 1, 1, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 1, 1, 1, 1]: output_scaling=30, L1=8x2+2, L2=30x8+8, total=266
  Subtotal params: 2100
  Subtotal weights: 72

7.2. CUMMAX_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

8.2. SHIFT_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

9.2. DIRECTION_SHARE_WEIGHTS (Layer 2)
----------------------------------------
  Each dims has 8x8=64 linear maps, each 16x16 + bias
  Subtotal params: 185344
  Subtotal weights: 2304

10.2. NONLINEAR_WEIGHTS (Layer 2)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

================================================================================
LAYER 3
================================================================================

4.3. SHARE_UP_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

5.3. SHARE_DOWN_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[0, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[0, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  Subtotal params: 3680
  Subtotal weights: 72

6.3. SOFTMAX_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[0, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[1, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[0, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 0, 0, 1, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 1, 0]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 1, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 1, 0]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 0, 1]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 0, 1]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 1, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 1, 0, 1, 1]: output_scaling=14, L1=16x2+2, L2=14x16+16, total=274
  dims=[1, 0, 1, 1, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 1, 1, 1, 1]: output_scaling=30, L1=8x2+2, L2=30x8+8, total=266
  Subtotal params: 2100
  Subtotal weights: 72

7.3. CUMMAX_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

8.3. SHIFT_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

9.3. DIRECTION_SHARE_WEIGHTS (Layer 3)
----------------------------------------
  Each dims has 8x8=64 linear maps, each 16x16 + bias
  Subtotal params: 185344
  Subtotal weights: 2304

10.3. NONLINEAR_WEIGHTS (Layer 3)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

================================================================================
LAYER 4
================================================================================

4.4. SHARE_UP_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

5.4. SHARE_DOWN_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[0, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[0, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 0]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 0]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 0, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 0, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 0, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 1, 0, 1, 1]: L1=16x8+8, L2=8x16+16, total=280
  dims=[1, 0, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  dims=[1, 1, 1, 1, 1]: L1=8x8+8, L2=8x8+8, total=144
  Subtotal params: 3680
  Subtotal weights: 72

6.4. SOFTMAX_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[0, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[1, 0, 1, 0, 0]: output_scaling=2, L1=8x2+2, L2=2x8+8, total=42
  dims=[0, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 0, 0, 1, 0]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 1, 0]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 1, 0]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 1, 0]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 0, 1]: output_scaling=2, L1=16x2+2, L2=2x16+16, total=82
  dims=[1, 1, 0, 0, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 0, 1, 0, 1]: output_scaling=6, L1=8x2+2, L2=6x8+8, total=74
  dims=[1, 1, 1, 0, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 0, 0, 1, 1]: output_scaling=6, L1=16x2+2, L2=6x16+16, total=146
  dims=[1, 1, 0, 1, 1]: output_scaling=14, L1=16x2+2, L2=14x16+16, total=274
  dims=[1, 0, 1, 1, 1]: output_scaling=14, L1=8x2+2, L2=14x8+8, total=138
  dims=[1, 1, 1, 1, 1]: output_scaling=30, L1=8x2+2, L2=30x8+8, total=266
  Subtotal params: 2100
  Subtotal weights: 72

7.4. CUMMAX_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

8.4. SHIFT_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[0, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[0, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 0]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 0]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 0, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 0, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 0, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 1, 0, 1, 1]: L1=16x4+4, L2=4x16+16, total=148
  dims=[1, 0, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  dims=[1, 1, 1, 1, 1]: L1=8x4+4, L2=4x8+8, total=76
  Subtotal params: 1944
  Subtotal weights: 72

9.4. DIRECTION_SHARE_WEIGHTS (Layer 4)
----------------------------------------
  Each dims has 8x8=64 linear maps, each 16x16 + bias
  Subtotal params: 185344
  Subtotal weights: 2304

10.4. NONLINEAR_WEIGHTS (Layer 4)
----------------------------------------
  dims=[0, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[0, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[0, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 0]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 0]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 0, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 0, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 0, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 1, 0, 1, 1]: L1=16x16+16, L2=16x16+16, total=544
  dims=[1, 0, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  dims=[1, 1, 1, 1, 1]: L1=8x16+16, L2=16x8+8, total=280
  Subtotal params: 7152
  Subtotal weights: 72

11. HEAD_WEIGHTS
----------------------------------------
  dims=[1, 1, 0, 1, 1]: weight=16x2=32, bias=2, total=34

12. MASK_WEIGHTS
----------------------------------------
  dims=[1, 0, 0, 1, 0]: weight=16x2=32, bias=2, total=34

================================================================================
FINAL SUMMARY
================================================================================
Total number of parameters: 7,066,364
Total number of elements in weights_list: 11,038
Valid dims combinations: 18

================================================================================
VERIFICATION BY LAYER TYPE
================================================================================
Initial weights (multiposteriors + decode + target_capacities): 6,229,032
share_up_weights (4 layers): 28,608
share_down_weights (4 layers): 14,720
softmax_weights (4 layers): 8,400
cummax_weights (4 layers): 7,776
shift_weights (4 layers): 7,776
direction_share_weights (4 layers): 741,376
nonlinear_weights (4 layers): 28,608
Final weights (head + mask): 68

Verification total: 7,066,364
```