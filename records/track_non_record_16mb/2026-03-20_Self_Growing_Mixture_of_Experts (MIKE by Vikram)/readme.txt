For 3060 logs: 3060 results (1.5215 bpb) based on 66M tokens in 42 mins . H100 at full batch atleast enables 15-30x tokens within competition time.

Results on a single RTX 3060 laptop (6GB VRAM, 65536 competition batch size): Loss
val_loss:2.5689 val_bpb:1.5215 (int8+zlib roundtrip) 3 experts active, depth=6, curve still improving at step 1000. Also tried frozen/ unfrozen backbone.


For H100 logs: Added data logs for 4000 steps run with 1 x H100 cluster (personal runpod credits). This run was my first runpod test and was quite suboptimal, with experts being spawned too early, backbone only tests after this are ongoing and reveal that a much later spawning (~1000steps isntead of 150) should do much better as the embedding clarity is much better. Also token count is is about half what will be achieved by 8xh100 in 10mins And expert top k is set to 1 and expert count is only 3, however we still achieve 1.2639 , which is ~0.035 from naive baseline 1.2244. However , final model size is 20mB here, which needs a little tuning (or quantization), but baseline seems close with this architecture and more tests, we still have a lot of levers to pull here.
