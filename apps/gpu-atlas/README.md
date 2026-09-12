# GPU Atlas 子应用

图谱入口和 AMD 筹备页由教程的 VitePress 站点提供，NVIDIA 展馆维护为一个独立的 [index.html](./nvidia/index.html)。页面内嵌脚本、样式、字体和图片，保留原有三维模型与交互，可直接用浏览器离线打开。站点入口链接直接打开同站的完整页面，没有 iframe，也不参与教程正文的章节编号。

## 源码与目录

NVIDIA 图谱的模块化源码、素材生成脚本与测试在私有仓库维护。本仓库只分发经过 JavaScript 混淆的单文件，图片来源与第三方许可说明保存在 `index.html` 的 `atlas-source-notices` JSON 数据块中。

| 目录 | 用途 |
| --- | --- |
| `apps/gpu-atlas/nvidia/index.html` | NVIDIA 展馆的完整单文件页面，直接维护并提交 |
| `docs/atlas/` | 图谱总入口及 AMD 筹备页 |
| `docs/public/atlas/nvidia/` | 构建脚本生成的 NVIDIA 静态文件，不提交 |

未来 AMD 应用可放到 `apps/gpu-atlas/amd/`，按自己的硬件结构、数据与模型维护。现阶段 AMD 入口提供筹备说明和已有教程链接，不复用 NVIDIA 数据冒充 AMD 内容。

## 离线打开与更新

直接用浏览器打开 `apps/gpu-atlas/nvidia/index.html` 即可使用 NVIDIA 展馆，无须安装应用依赖或运行开发服务器。离线使用时，返回教程与参考资料等外部链接仍需要可访问的站点。

功能开发与混淆构建在私有源码仓库进行，维护者通过 `build:protected` 生成 `dist-protected/index.html` 后更新本仓库的单文件。保持脚本、样式和素材全部内嵌，并保留 `atlas-source-notices` 中的来源与第三方许可说明。公开版本不包含 source map；不要把模块化源码或混淆配置复制回本仓库。提交前分别检查直接打开文件和教程站点内的显示与交互。

混淆提高阅读和复用成本，浏览器运行时仍可以提取脚本与资源。用户需要看到的页面文字、官方资料链接和第三方许可保留可读。

## 教程站点集成

在仓库根目录安装教程站点依赖并启动预览：

```bash
npm install
npm run docs:dev
```

`docs:dev` 和 `docs:build` 都会先调用 `scripts/build-atlas.mjs`。脚本检查完整 HTML、混淆构建标记及 source map，然后将文件复制到 `docs/public/atlas/nvidia/index.html`，交给 VitePress 一起发布。这个步骤不安装子应用依赖，也不运行 React/Vite 构建。原有 Pages 工作流的 `npm install` → `npm run docs:build` 已包含图谱，无须单独发布。

```bash
npm run docs:build
npm run docs:preview
```

默认图谱地址为 `/hello-gpu/atlas/nvidia/`。运行 `EDGEONE=1 npm run docs:build` 时，图谱地址为 `/atlas/nvidia/`。两种部署复制同一份 HTML。

修改 NVIDIA 的 `index.html` 后，在仓库根目录执行 `npm run docs:build-atlas`，再刷新正在运行的教程开发站点，即可预览更新。

## 主题与版本控制

子应用默认深色，并读取、写入与教程相同的 `localStorage` 键 `vitepress-theme-appearance`。从日间教程进入图谱或返回时保持用户已选的主题；主题选择由各页面初始化处理。

NVIDIA 目录只提交 `index.html`。`docs/public/atlas/nvidia/` 是复制生成的站点文件，由 `.gitignore` 排除；它的内容应始终来自 `apps/gpu-atlas/nvidia/index.html`。
