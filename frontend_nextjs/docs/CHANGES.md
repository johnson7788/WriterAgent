综述撰写应用的实现逻辑
1）用户输入要生成综述的主题和选择的语言，点击确定开始生成大纲。
2）前端像后端的/api/outline发送主题和语言，后端开始SSE流式生成markdown格式的大纲内容。
3）前端进行确认，可以进行编辑，或者不编辑，点击确认后，向后端的/api/review发送大纲和主题。
4）后端开始SSE流式生成markdown格式的综述内容。
5）前端用户可以选中某个段落进行扩写，改写等，右上角有下载按钮，用户可以进行下载成docx格式。

总体思路
  核心思路是：用一个新的、分步骤的界面替换当前的聊天界面。我们将复用项目中的UI组件（如按钮、输入框）、Markdown渲染器、编辑器组件
  以及处理SSE流式响应的底层代码。

   1. UI改造：将主要的聊天视图 (chat/page.tsx) 从一个对话列表，改造成一个多阶段的视图：
       * 阶段一：显示“主题输入”表单。
       * 阶段二：显示流式生成的“大纲”，并提供编辑和确认功能。
       * 阶段三：显示流式生成的“综述内容”，并提供编辑和下载功能。
   2. API层改造：在 src/core/api/ 目录下，新增处理 /api/outline 和 /api/review 请求的函数。我们可以参考 src/core/api/chat.ts
      的实现，因为它很可能已经处理了SSE流。
   3. 状态管理：我们需要一个新的状态来管理当前的流程（例如：idle, generatingOutline, confirmingOutline, generatingReview,
      done），以及保存主题、大纲和综述内容。

  ---

  文件修改建议

  以下是具体需要关注和修改的文件列表：

  1. UI层 (主要修改)

   * `src/app/chat/page.tsx` 和 `src/app/chat/main.tsx`:
       * 目的: 这是应用的主入口和布局。您需要修改它，用新的工作流组件替换掉当前的聊天组件（如 MessageListView 和 InputBox）。
       * 建议:
           1. 创建一个新的状态来管理当前的生成步骤 (e.g., 'FORM', 'OUTLINE', 'REVIEW')。
           2. 根据这个状态，条件性地渲染不同的组件：
               * 初始状态下，渲染一个新的表单组件（用于输入主题和选择语言）。
               * 生成大纲时，渲染一个只读的Markdown显示组件。
               * 确认大纲时，渲染编辑器组件，并加载大纲内容。
               * 生成综述时，再次渲染只读的Markdown显示组件。
               * 完成后，渲染编辑器组件，并加载综述内容，同时显示下载和编辑按钮。

   * `src/app/chat/components/input-box.tsx`:
       * 目的: 这是当前的聊天输入框。
       * 建议: 这个文件可以被一个新的组件替代。您可以创建一个
         review-request-form.tsx，包含一个文本输入框（用于主题）和一个下拉菜单（用于语言选择），以及一个“生成大纲”的按钮。

   * `src/app/chat/components/message-list-view.tsx` 和 `messages-block.tsx`:
       * 目的: 用于显示对话消息。
       * 建议: 这两个文件不再需要，因为新的应用不是对话形式。您可以用一个更大的容器来展示大纲或综述。

  2. 内容展示与编辑 (主要复用)

   * `src/components/editor/index.tsx`:
       * 目的: 这是一个功能强大的文本编辑器。
       * 建议: 完美复用。您可以用它来：
           1. 让用户编辑和确认生成的大纲。
           2. 在综述生成后，让用户进行段落的扩写、改写和最终编辑。

   * `src/components/deer-flow/markdown.tsx`:
       * 目的: 用于渲染Markdown文本。
       * 建议: 完美复用。在SSE流式生成大纲和综述的过程中，用这个组件来实时显示内容。

   * `src/app/chat/components/research-report-block.tsx`:
       * 目的: 看起来是用来展示最终报告的组件。
       * 建议: 可以改造复用。这个组件的内部逻辑可能很适合用来包装 markdown.tsx 或 editor/index.tsx，作为展示大纲和最终综述的容器。

  3. API通信层 (需要新增)

   * `src/core/api/chat.ts`:
       * 目的: 处理与后端 /api/chat 的通信。
       * 建议: 作为模板。复制此文件创建一个新的 src/core/api/review.ts。

   * `src/core/api/review.ts` (新文件):
       * 目的: 处理与新后端API的通信。
       * 建议: 在这个新文件中创建两个函数：
           1. generateOutline(topic: string, language: string): 向 /api/outline 发送请求并处理SSE流。
           2. generateReview(topic: string, outline: string): 向 /api/review 发送请求并处理SSE流。

   * `src/core/sse/fetch-stream.ts`:
       * 目的: 底层的SSE流处理逻辑。
       * 建议: 直接复用。您在 review.ts 中创建的新函数应该会调用它。

  4. 状态管理 (需要修改)

   * `src/core/store/store.ts`:
       * 目的: 全局状态管理（可能使用Zustand或类似库）。
       * 建议: 在这里添加新的状态来驱动整个流程，例如：
           * generationStep: 'FORM' | 'OUTLINE' | 'REVIEW'
           * topic: string
           * language: string
           * outlineContent: string
           * reviewContent: string

  5. 新功能：下载 DOCX

   * 建议:
       1. 您需要安装一个新的库来将Markdown或HTML转为DOCX，例如 html-to-docx-file-saver 或 mammoth.js。
   1         pnpm install html-to-docx-file-saver
       2. 在展示最终综述的组件中（例如，改造后的 research-report-block.tsx），添加一个“下载”按钮。
       3. 按钮的点击事件会调用一个函数，该函数使用您安装的库，将编辑器中的内容（通常是HTML格式）转换为DOCX文件并触发浏览器下载。

  ---

  行动计划

  我建议按以下步骤进行：

   1. 代码阅读: 我将首先阅读 src/app/chat/main.tsx, src/core/api/chat.ts 和 src/components/editor/index.tsx
      这几个核心文件，以精确验证我的上述假设。
   2. UI改造: 从 src/app/chat/main.tsx 开始，移除聊天组件，并创建一个简单的表单用于第一步。
   3. API集成: 创建 src/core/api/review.ts 并实现 generateOutline 函数，连接到您的新UI。
   4. 内容展示: 将流式返回的大纲内容用 markdown.tsx 组件显示出来。
   5. 迭代完善: 逐步实现大纲编辑、综述生成和下载功能。

  首先，我将阅读您提到的几个核心文件，即 src/app/chat/main.tsx、src/core/api/chat.ts 和
  src/components/editor/index.tsx，以便更精确地确认我的建议。您同意吗？




# 更新日志，调用后端的API地址
/Users/admin/git/AgenticWriter/frontend_nextjs/src/core/api/resolve-service-url.ts

# 综述撰写页面
src/app/review

# 调用后端的代码
src/core/api/review.ts