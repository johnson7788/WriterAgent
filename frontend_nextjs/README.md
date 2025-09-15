# 📚 ReviewGen Web UI — 综述生成界面

ReviewGen Web UI 是一个用于**学术综述生成**的前端界面：连接后端 API，支持流式生成、Markdown 预览、会话管理与一键导出，帮助你快速搭建从问题到综述的工作流。

---

## ✨ 功能特性（Highlights）

* **流式生成**：Token-by-token 实时输出，打字机般的生成体验。
* **Markdown 预览**：内置样式良好的 Markdown 渲染，支持代码块、表格、公式等。
* **可配置后端**：通过环境变量一键切换后端 API 地址。
* **轻量状态管理**：Zustand 驱动，简单稳定。
* **动效与细节**：基于 Framer Motion 的细腻动效，交互更顺滑。
* **易部署**：支持本地开发、Docker、Docker Compose 与生产构建。

> 技术栈：Next.js · shadcn/ui · Zustand · Framer Motion · React Markdown

---

## 📖 文档（Documentation）
实现逻辑
1）用户输入要生成综述的主题和选择的语言，点击确定开始生成大纲。
2）前端像后端的/api/outline发送主题和语言，后端开始SSE流式生成markdown格式的大纲内容。
3）前端进行确认，可以进行编辑，或者不编辑，点击确认后，向后端的/api/review发送大纲和主题。
4）后端开始SSE流式生成markdown格式的综述内容。
5）前端用户可以选中某个段落进行扩写，改写等，右上角有下载按钮，用户可以进行下载成docx格式。

## 🚀 快速开始（Quick Start）

### 1. 运行前置（Prerequisites）

* **后端 API 服务**
* **Node.js** `v22.14.0+`
* **pnpm** `v10.6.2+`

### 2. 配置（Configuration）

在项目根目录创建 `.env` 文件，并配置以下环境变量：

* `NEXT_PUBLIC_API_URL`：后端 API 的基础地址（带前缀路径时也可直接填入）。

建议从示例文件开始：

```bash
cp env_example .env
```

示例：

```ini
# .env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 3. 安装依赖（Install）

本项目使用 `pnpm` 作为包管理工具：

```bash
cd web
pnpm install
```

### 4. 本地开发（Development）

> **注意**：请确保后端 API 服务已启动。

启动开发服务器：

```bash
cd web
pnpm dev
```

默认访问地址：`http://localhost:3000`

若后端地址非本机或路径不同，请在 `.env` 中设置 `NEXT_PUBLIC_API_URL`。

### 5. 生产构建与启动（Production）

```bash
cd web
pnpm build
pnpm start
```

---

## 🐳 使用 Docker 部署

### 方式一：Docker CLI

1）准备 `.env`（参见上文配置章节）。

2）构建镜像（你也可以用 `--build-arg` 覆盖 API 地址）：

```bash
docker build --build-arg NEXT_PUBLIC_API_URL=YOUR_API_URL -t reviewgen-web .
```

3）运行容器：

```bash
# 将 reviewgen-web-app 替换成你喜欢的容器名
docker run -d -t -p 3000:3000 --env-file .env --name reviewgen-web-app reviewgen-web

# 停止服务
docker stop reviewgen-web-app
```

### 方式二：Docker Compose

```bash
# 构建镜像
docker compose build

# 启动服务
docker compose up
```

---

## 🧩 常见问题（Troubleshooting）

* **前端能开但接口 404/超时**：检查 `NEXT_PUBLIC_API_URL` 是否正确、后端是否已启动、网络与端口是否可达。
* **CORS 报错**：需要在后端开启跨域（CORS）或配置允许的来源。
* **Node/pnpm 版本不符**：请升级到文档建议的版本，或使用 `nvm`/`corepack` 管理版本。

---

## 🤝 贡献（Contributing）

欢迎提交 Issue 与 PR，一起完善开源的综述生成工作流。

---

## ❤️ 致谢（Acknowledgments）

我们向开源社区致以诚挚的感谢。本项目基于以下优秀开源项目构建：

* [Next.js](https://nextjs.org/) —— 强大的 React 应用框架
* [shadcn/ui](https://ui.shadcn.com/) —— 极简、可组合的 UI 组件
* [Zustand](https://zustand.docs.pmnd.rs/) —— 轻量但强大的状态管理
* [Framer Motion](https://www.framer.com/motion/) —— 出色的动画库
* [React Markdown](https://www.npmjs.com/package/react-markdown) —— 灵活的 Markdown 渲染
* 特别感谢 [SToneX](https://github.com/stonexer) 在 **token-by-token** 可视化效果上的贡献（参考：`./src/core/rehype/rehype-split-words-into-spans.ts`）

这些项目共同构成了 Web UI 的技术底座，开源让协作更有力量。

