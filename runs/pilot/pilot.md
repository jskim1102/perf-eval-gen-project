# Strength pilot

- N: 50
- Seed: 0
- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Selection: maximum PSNR among strengths meeting PSNR and SSIM targets
- FID note: FID is excluded from the EVAL500 measurement axis and transferred to a separate dataset pending definition.

| Strength | FID (historical) | PSNR | SSIM | Result |
| ---: | ---: | ---: | ---: | :---: |
| 0.15 | 16.367665 | 32.763442 | 0.929036 | PASS |
| 0.20 | 19.329095 | 31.666764 | 0.923174 | PASS |
| 0.25 | 21.991344 | 31.199200 | 0.920524 | PASS |
| 0.30 | 26.098682 | 30.336582 | 0.915037 | PASS |
| 0.35 | 28.422060 | 29.905268 | 0.912265 | PASS |
| 0.40 | 33.692233 | 29.076304 | 0.906201 | PASS |

Selected: `0.15`
