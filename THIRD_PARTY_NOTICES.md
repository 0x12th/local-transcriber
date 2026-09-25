# Third-party notices

This file records licenses and provenance for third-party portions used by the
optional GigaAM backend and its supported model artifact. These notices apply
to the corresponding third-party material; they do not relicense the
local-transcriber project as a whole.

## Giga Pisar source code

The modules in `local_transcriber/gigaam/` are adapted from
`server/giga_core.py` in Giga Pisar at commit
[`8e5e0bb77b5e63ff27e74a909ee7c7e24afe0fc0`](https://github.com/moznoazachem/giga-pisar/tree/8e5e0bb77b5e63ff27e74a909ee7c7e24afe0fc0).
The pinned source is licensed under the following terms:

```text
MIT License

Copyright (c) 2026 moznoazachem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GigaAM model artifact

The supported `v3_e2e_rnnt` ONNX int8 bundle is a converted and quantized form
of the GigaAM v3 model created by the GigaChat Team. The upstream model and the
release bundle are licensed separately from the Giga Pisar source code.

Primary model source:
[`salute-developers/GigaAM`](https://github.com/salute-developers/GigaAM/tree/559d88d6b72541412743929f633a6ae7c9950b85).
The official repository identifies the `v3_e2e_rnnt` weights as
[`ai-sage/GigaAM-v3`, revision `e2e_rnnt`](https://huggingface.co/ai-sage/GigaAM-v3/tree/7655ad717f8122257385bb4b2f373db3697e8680).

Supported converted bundle:
[`gigaam-v3-onnx-int8.tar.gz`](https://github.com/moznoazachem/giga-pisar-cli/releases/download/v1.0/gigaam-v3-onnx-int8.tar.gz)
from Pisar CLI v1.0, GitHub asset ID `525206315`, SHA-256
`e5a75ab56ab6d3f3a70ab17dd1ce858fe8180597963839dc806014447483224c`.
The download URL and tag are not treated as immutable; the checksum identifies
the supported artifact.

The model is licensed under the following terms:

```text
MIT License

Copyright (c) 2024 GigaChat Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

No code from the `giga-pisar-cli` repository is copied into this project. Its
release is referenced only as the provenance of the pinned model artifact.
