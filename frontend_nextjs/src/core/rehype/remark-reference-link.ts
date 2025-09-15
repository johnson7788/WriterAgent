import type { Root, Text } from "mdast";
import type { Parent } from "unist";
import { visit, SKIP } from "unist-util-visit";

export function remarkReferenceLink() {
  return (tree: Root) => {
    visit(tree, "text", ((node: any, index: any, parent: any) => {
      if (typeof node.value !== "string" || index === undefined || !parent) {
        return;
      }

      // Matches [text][1] or [1]
      const citationRegex = /\[([^\]]*)\]\[(\d+)\]|\[(\d+)\]/g;

      const parts: any[] = [];
      let lastIndex = 0;
      let match;

      while ((match = citationRegex.exec(node.value)) !== null) {
        // Add text before the match
        if (match.index > lastIndex) {
          parts.push({
            type: "text",
            value: node.value.slice(lastIndex, match.index),
          });
        }

        const fullMatch = match[0];
        const refId = match[2] ?? match[3];

        parts.push({
          type: "link",
          url: `#ref-${refId}`,
          children: [{ type: "text", value: fullMatch }],
          data: {
            hProperties: {
              className: "reference-link",
            },
          },
        });

        lastIndex = citationRegex.lastIndex;
      }

      // Add text after the last match
      if (lastIndex < node.value.length) {
        parts.push({
          type: "text",
          value: node.value.slice(lastIndex),
        });
      }

      if (parts.length > 1) {
        parent.children.splice(index, 1, ...parts);
        return [SKIP, index + parts.length];
      }
    }) as any);
  };
}