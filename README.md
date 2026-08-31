# Huawei Competition Workspace

## 目录结构

```text
huawei_competition/
├── README.md
├── docs/
│   ├── 项目调研.md
│   ├── NVFP4_HiF4_算法流程总结.md
│   └── NVFP4_HiF4_相关技术调研.md
├── src/
│   ├── baseline.py
│   └── solution-0818.py
├── reference/
│   ├── 2026年华为算法大赛-初赛任务书-0819-V2.docx
│   ├── 2026+Huawei+Algorithm+Competition+-+Preliminary+Round+Task+Document-0819-V2.docx
│   └── 本地调试参考-0818.zip
├── output/
│   └── pdf/
│       └── 项目调研.pdf
└── tmp/
    └── local_debug_extracted/
```

## 说明

- `docs/`：项目说明、算法流程和技术调研。
- `src/`：算法原型与官方提交模板。
- `reference/`：任务书和官方本地调试资料。
- `output/pdf/`：最终导出的 PDF 文档。
- `tmp/`：解压或渲染产生的临时文件，不作为提交内容。

正式比赛提交时，应以 `src/solution-0818.py` 为基础整理为符合六接口要求的 `solution.py`，并单独打包。
