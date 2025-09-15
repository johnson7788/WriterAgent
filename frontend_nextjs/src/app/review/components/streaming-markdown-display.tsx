

"use client";

import { LoadingAnimation } from "~/components/deer-flow/loading-animation";
import { Markdown } from "~/components/deer-flow/markdown";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

interface StreamingMarkdownDisplayProps {
  title: string;
  content: string;
}

export function StreamingMarkdownDisplay({ title, content }: StreamingMarkdownDisplayProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="min-h-[300px]">
        <Markdown animated checkLinkCredibility>
          {content}
        </Markdown>
        <LoadingAnimation className="my-12" />
      </CardContent>
    </Card>
  );
}
