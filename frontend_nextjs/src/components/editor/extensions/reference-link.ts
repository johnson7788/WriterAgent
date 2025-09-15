import { Mark, markInputRule, markPasteRule, mergeAttributes } from '@tiptap/core';
import { Plugin, PluginKey } from 'prosemirror-state';

export interface ReferenceLinkOptions {
  HTMLAttributes: Record<string, any>;
  inclusive?: boolean;
  openOnClick?: boolean;
  linkOnPaste?: boolean;
  autolink?: boolean;
}

declare module '@tiptap/core' {
  interface Commands<ReturnType> {
    referenceLink: {
      /**
       * Set a reference link mark
       */
      setReferenceLink: (attributes: { href: string }) => ReturnType;
      /**
       * Toggle a reference link mark
       */
      toggleReferenceLink: () => ReturnType;
      /**
       * Unset a reference link mark
       */
      unsetReferenceLink: () => ReturnType;
    };
  }
}

export const referenceLinkRegex = /\[([^\]]*)\]\[(\d+)\]|\[(\d+)\]/g;

export const ReferenceLink = Mark.create<ReferenceLinkOptions>({
  name: 'referenceLink',

  priority: 1000,

  keepOnSplit: false,

  inclusive() {
    return this.options.inclusive ?? false;
  },

  addOptions() {
    return {
      openOnClick: true,
      linkOnPaste: true,
      autolink: true,
      inclusive: false,
      HTMLAttributes: {
        target: '_self',
        rel: 'noopener noreferrer nofollow',
        class: 'reference-link',
      },
      
    };
  },

  addAttributes() {
    return {
      href: {
        default: null,
      },
      'data-author': {
        default: null,
      },
      'data-index': {
        default: null,
      },
    };
  },

  parseHTML() {
    return [{ tag: 'a[data-reference-link]' }];
  },

  renderHTML({ HTMLAttributes }) {
    return ['a', mergeAttributes(this.options.HTMLAttributes, HTMLAttributes, { 'data-reference-link': 'true' }), 0];
  },

  addCommands() {
    return {
      setReferenceLink:
        (attributes) =>
        ({ chain }) => {
          return chain().setMark(this.name, attributes).run();
        },

      toggleReferenceLink:
        () =>
        ({ chain }) => {
          return chain().toggleMark(this.name).run();
        },

      unsetReferenceLink:
        () =>
        ({ chain }) => {
          return chain().unsetMark(this.name).run();
        },
    };
  },

  addPasteRules() {
    return [
      markPasteRule({
        find: referenceLinkRegex,
        type: this.type,
        getAttributes: (match) => {
          const author = match[1];
          const index = match[2] || match[3];
          return {
            href: `#ref-${index}`,
            'data-author': author,
            'data-index': index,
          };
        },
      }),
    ];
  },

  addInputRules() {
    return [
      markInputRule({
        find: referenceLinkRegex,
        type: this.type,
        getAttributes: (match) => {
          const author = match[1];
          const index = match[2] || match[3];
          return {
            href: `#ref-${index}`,
            'data-author': author,
            'data-index': index,
          };
        },
      }),
    ];
  },

  addProseMirrorPlugins() {
    const { type } = this;

    return [
      new Plugin({
        key: new PluginKey('referenceLinkInitialScan'),
        view: (editorView) => {
          const { state } = editorView;
          const { tr } = state;
          let modified = false;

          state.doc.descendants((node, pos) => {
            if (!node.isText) {
              return;
            }

            const text = node.textContent;
            let match;

            while ((match = referenceLinkRegex.exec(text)) !== null) {
              const [fullMatch] = match;
              const author = match[1];
              const index = match[2] || match[3];
              const from = pos + match.index;
              const to = from + fullMatch.length;

              const linkMark = type.create({
                href: `#ref-${index}`,
                'data-author': author,
                'data-index': index,
              });

              const hasMark = state.doc.rangeHasMark(from, to, type);
              if (!hasMark) {
                tr.addMark(from, to, linkMark);
                modified = true;
              }
            }
          });

          if (modified) {
            editorView.dispatch(tr);
          }

          return {};
        },
      }),
    ];
  }
});