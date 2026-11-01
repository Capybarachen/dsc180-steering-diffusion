MNIST CFG: 10,000 samples per w (1,000 per digit), 16 DDIM steps. Label accuracy and confidence from the lab's frozen classifier; FID on its 128 features against the 60,000 MNIST training digits; FID_class per digit, averaged. Real test digits come from other writers, so they are a reference point, not a floor.

| model | w | images | network calls per image | label accuracy | confidence | FID | FID_class | pixel spread |
|---|---|---|---|---|---|---|---|---|
| ours, 30,000 updates | 0 | 10000 | 16 | 10.6% | 0.105 | 58.30 | – | 0.2444 |
| ours, 30,000 updates | 1 | 10000 | 16 | 98.3% | 0.981 | 7.34 | 20.89 | 0.1930 |
| ours, 30,000 updates | 2 | 10000 | 32 | 100.0% | 1.000 | 60.60 | 134.66 | 0.1778 |
| ours, 30,000 updates | 4 | 10000 | 32 | 100.0% | 1.000 | 133.70 | 259.71 | 0.1725 |
| course generator, 96,000 updates | 0 | 10000 | 16 | 10.5% | 0.106 | 28.72 | – | 0.2461 |
| course generator, 96,000 updates | 1 | 10000 | 16 | 98.6% | 0.984 | 5.91 | 16.06 | 0.1943 |
| course generator, 96,000 updates | 2 | 10000 | 32 | 100.0% | 1.000 | 48.19 | 111.31 | 0.1803 |
| course generator, 96,000 updates | 4 | 10000 | 32 | 100.0% | 1.000 | 100.37 | 201.71 | 0.1762 |
| real test digits | – | 10000 | – | 99.3% | 0.991 | 5.58 | 17.42 | 0.2124 |
